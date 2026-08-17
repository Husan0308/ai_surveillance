from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

LocalKey = tuple[int, int]


@dataclass
class VisualSample:
    jpeg: bytes
    quality: float
    seen_at: float


@dataclass
class VerifyTask:
    key_a: LocalKey
    key_b: LocalKey
    gid_a: int
    gid_b: int
    room_id: int
    image_a: bytes
    image_b: bytes
    submitted_at: float
    reason: str


@dataclass
class VerifyResult:
    key_a: LocalKey
    key_b: LocalKey
    gid_a: int
    gid_b: int
    room_id: int
    verdict: str
    confidence: float
    reason: str
    latency_ms: float
    error: str = ""


class QwenRoomReIDVerifier:
    """Slow, conservative visual judge for same-room peer-camera ReID.

    The fast TAO/NvDCF path remains authoritative for geometry and candidate
    generation. Qwen is only a second opinion for peer cameras in the same room.
    It never creates tracks and never writes directly to NvDCF.

    Two independent Qwen votes are required before a SAME merge or DIFFERENT split.
    This prevents a single VLM response from permanently changing identity state.
    All calls run on a background thread so camera rendering never waits on Qwen.
    """

    def __init__(self) -> None:
        self.enabled = os.environ.get("CAMERA_V2_QWEN_VERIFY", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.url = os.environ.get(
            "CAMERA_V2_QWEN_URL", "http://127.0.0.1:8080/v1/chat/completions"
        ).strip()
        self.model = os.environ.get("CAMERA_V2_QWEN_MODEL", "qwen3-vl-reid").strip()
        self.timeout = max(2.0, float(os.environ.get("CAMERA_V2_QWEN_TIMEOUT", "25")))
        self.min_same_conf = float(os.environ.get("CAMERA_V2_QWEN_SAME_CONF", "0.78"))
        self.min_diff_conf = float(os.environ.get("CAMERA_V2_QWEN_DIFF_CONF", "0.82"))
        self.same_votes_required = max(2, int(os.environ.get("CAMERA_V2_QWEN_SAME_VOTES", "2")))
        self.diff_votes_required = max(2, int(os.environ.get("CAMERA_V2_QWEN_DIFF_VOTES", "2")))
        self.pair_cooldown = max(2.0, float(os.environ.get("CAMERA_V2_QWEN_PAIR_COOLDOWN", "5.0")))
        self.scan_interval = max(0.5, float(os.environ.get("CAMERA_V2_QWEN_SCAN_INTERVAL", "1.0")))
        self.min_peer_reid = float(os.environ.get("CAMERA_V2_QWEN_MIN_PEER_REID", "0.34"))
        self.max_pending = max(1, min(4, int(os.environ.get("CAMERA_V2_QWEN_MAX_PENDING", "2"))))
        self.visual_ttl = max(3.0, float(os.environ.get("CAMERA_V2_QWEN_VISUAL_TTL", "15")))

        self.visuals: dict[LocalKey, deque[VisualSample]] = {}
        self.input_q: queue.Queue[VerifyTask | None] = queue.Queue(maxsize=self.max_pending)
        self.output_q: queue.Queue[VerifyResult] = queue.Queue(maxsize=self.max_pending * 4)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.pending_pairs: set[frozenset[LocalKey]] = set()
        self.last_pair_submit: dict[frozenset[LocalKey], float] = {}
        self.same_votes: dict[frozenset[LocalKey], int] = defaultdict(int)
        self.diff_votes: dict[frozenset[LocalKey], int] = defaultdict(int)
        self.last_scan = 0.0

        self.requests = 0
        self.responses = 0
        self.same = 0
        self.different = 0
        self.uncertain = 0
        self.merges = 0
        self.splits = 0
        self.cannot_links = 0
        self.failed = 0
        self.dropped = 0
        self.last_latency_ms = 0.0
        self.last_error = ""
        self.last_verdict = "none"

        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="camera-v2-qwen-reid", daemon=True)
            self.thread.start()

    @staticmethod
    def _pair_key(a: LocalKey, b: LocalKey) -> frozenset[LocalKey]:
        return frozenset((a, b))

    def remember(self, key: LocalKey, crop_bgr: np.ndarray, quality: float) -> None:
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return
        try:
            import cv2

            h, w = crop_bgr.shape[:2]
            if h < 40 or w < 16:
                return
            # Keep Qwen input small: the verifier is deliberately CPU-side so it
            # cannot steal VRAM from YOLO/NvDCF on the GTX 1050 Ti.
            scale = min(1.0, 320.0 / max(h, 1), 180.0 / max(w, 1))
            if scale < 0.999:
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (max(16, int(round(w * scale))), max(40, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            if not ok:
                return
            sample = VisualSample(bytes(encoded), float(quality), time.monotonic())
            rows = self.visuals.get(key)
            if rows is None:
                rows = deque(maxlen=4)
                self.visuals[key] = rows
            # Avoid filling memory with nearly identical immediate frames.
            if rows and sample.seen_at - rows[-1].seen_at < 0.35:
                if sample.quality > rows[-1].quality:
                    rows[-1] = sample
                return
            rows.append(sample)
        except Exception:
            return

    def _best_montage(self, key: LocalKey, now: float) -> bytes | None:
        rows = [row for row in self.visuals.get(key, ()) if now - row.seen_at <= self.visual_ttl]
        if not rows:
            return None
        rows.sort(key=lambda row: (row.quality, row.seen_at), reverse=True)
        chosen = rows[:2]
        if len(chosen) == 1:
            return chosen[0].jpeg

        try:
            import cv2

            images = []
            for row in chosen:
                arr = np.frombuffer(row.jpeg, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    continue
                h, w = image.shape[:2]
                target_h = 256
                target_w = max(48, min(180, int(round(w * target_h / max(h, 1)))))
                images.append(cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA))
            if not images:
                return None
            if len(images) == 1:
                ok, enc = cv2.imencode(".jpg", images[0], [int(cv2.IMWRITE_JPEG_QUALITY), 84])
                return bytes(enc) if ok else None
            gap = np.full((256, 8, 3), 32, dtype=np.uint8)
            montage = np.concatenate((images[0], gap, images[1]), axis=1)
            ok, enc = cv2.imencode(".jpg", montage, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            return bytes(enc) if ok else None
        except Exception:
            return chosen[0].jpeg

    @staticmethod
    def _dot(a, b) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        return float(sum(float(x) * float(y) for x, y in zip(a, b)))

    def _candidate_pairs(self, manager, now: float) -> list[tuple[float, LocalKey, LocalKey, int, int, int, str]]:
        active = [
            key
            for key in manager.bindings
            if manager._is_active(key, now) and self._best_montage(key, now) is not None
        ]
        by_room: dict[int, list[LocalKey]] = defaultdict(list)
        for key in active:
            by_room[manager.room_of(key[0])].append(key)

        output: list[tuple[float, LocalKey, LocalKey, int, int, int, str]] = []
        for room_id, keys in by_room.items():
            # Only compare peer cameras. Two tracks from the same camera are known
            # different people and are already a hard cannot-link in GlobalReIDManager.
            raw_pairs: list[tuple[float, LocalKey, LocalKey, int, int, str]] = []
            best_for: dict[LocalKey, tuple[float, LocalKey]] = {}
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    if a[0] == b[0]:
                        continue
                    va, _ca, na = manager._track_prototype(a)
                    vb, _cb, nb = manager._track_prototype(b)
                    if va is None or vb is None or min(na, nb) < 2:
                        continue
                    reid = self._dot(va, vb)
                    ga = manager._resolve(manager.bindings[a].global_id)
                    gb = manager._resolve(manager.bindings[b].global_id)
                    same_gid = ga == gb
                    if not same_gid and reid < self.min_peer_reid:
                        continue
                    reason = "same_gid_audit" if same_gid else "peer_candidate"
                    # A same-GID pair is high priority because Qwen can catch an
                    # accidental identity collision before it contaminates memory.
                    priority = (1.4 if same_gid else 1.0) + max(-0.2, reid)
                    raw_pairs.append((priority, a, b, ga, gb, reason))
                    if not same_gid:
                        old = best_for.get(a)
                        if old is None or reid > old[0]:
                            best_for[a] = (reid, b)
                        old = best_for.get(b)
                        if old is None or reid > old[0]:
                            best_for[b] = (reid, a)

            for priority, a, b, ga, gb, reason in raw_pairs:
                if ga != gb:
                    # For different IDs, ask Qwen only for mutual-nearest peer
                    # candidates. This approximates a one-to-one room assignment
                    # and prevents an N^2 VLM request storm in crowded scenes.
                    if best_for.get(a, (-1.0, None))[1] != b or best_for.get(b, (-1.0, None))[1] != a:
                        continue
                output.append((priority, a, b, ga, gb, room_id, reason))

        output.sort(reverse=True, key=lambda row: row[0])
        return output

    def service(self, manager, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic() if now is None else float(now)
        self._consume_results(manager, now)
        if now - self.last_scan < self.scan_interval:
            return
        self.last_scan = now
        if self.input_q.qsize() >= self.max_pending:
            return

        for _priority, a, b, ga, gb, room_id, reason in self._candidate_pairs(manager, now):
            if self.input_q.qsize() >= self.max_pending:
                break
            pair = self._pair_key(a, b)
            if pair in self.pending_pairs:
                continue
            if now - self.last_pair_submit.get(pair, 0.0) < self.pair_cooldown:
                continue
            image_a = self._best_montage(a, now)
            image_b = self._best_montage(b, now)
            if image_a is None or image_b is None:
                continue
            task = VerifyTask(a, b, ga, gb, room_id, image_a, image_b, now, reason)
            try:
                self.input_q.put_nowait(task)
                self.pending_pairs.add(pair)
                self.last_pair_submit[pair] = now
                self.requests += 1
            except queue.Full:
                self.dropped += 1
                break

    def _prompt(self) -> str:
        return (
            "You are a conservative person re-identification verifier. "
            "Image A is one local tracked person (possibly a two-frame montage). "
            "Image B is another local tracked person from the peer camera in the SAME physical room. "
            "Decide whether A and B are the SAME physical person. Ignore background and camera viewpoint. "
            "Compare independent cues: body proportions/build, clothing layout and patterns, trousers, shoes, "
            "hair/head shape, accessories, and face only when clearly visible. Camera angle, pose, lighting and "
            "partial occlusion may differ. If evidence is insufficient or cues conflict, answer uncertain; do not guess. "
            "Return JSON only: {\"verdict\":\"same|different|uncertain\",\"confidence\":0.0}."
        )

    @staticmethod
    def _data_url(jpeg: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

    def _request(self, task: VerifyTask) -> VerifyResult:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt()},
                        {"type": "image_url", "image_url": {"url": self._data_url(task.image_a)}},
                        {"type": "image_url", "image_url": {"url": self._data_url(task.image_b)}},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 48,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
            text = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
            verdict, confidence = self._parse_answer(text)
            return VerifyResult(
                task.key_a,
                task.key_b,
                task.gid_a,
                task.gid_b,
                task.room_id,
                verdict,
                confidence,
                task.reason,
                (time.monotonic() - started) * 1000.0,
            )
        except Exception as exc:
            return VerifyResult(
                task.key_a,
                task.key_b,
                task.gid_a,
                task.gid_b,
                task.room_id,
                "uncertain",
                0.0,
                task.reason,
                (time.monotonic() - started) * 1000.0,
                f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _parse_answer(text: str) -> tuple[str, float]:
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            row = json.loads(cleaned)
            verdict = str(row.get("verdict", "uncertain")).strip().lower()
            confidence = float(row.get("confidence", 0.0))
        except Exception:
            match = re.search(r"\b(same|different|uncertain)\b", cleaned, flags=re.I)
            verdict = match.group(1).lower() if match else "uncertain"
            numbers = re.findall(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", cleaned)
            confidence = float(numbers[-1]) if numbers else 0.0
        if verdict not in {"same", "different", "uncertain"}:
            verdict = "uncertain"
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return verdict, confidence

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.input_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if task is None:
                return
            result = self._request(task)
            try:
                self.output_q.put_nowait(result)
            except queue.Full:
                self.dropped += 1

    @staticmethod
    def _binding_strength(manager, key: LocalKey) -> tuple[int, int, float, int]:
        binding = manager.bindings.get(key)
        if binding is None:
            return (-1, -1, 0.0, 0)
        gid = manager._resolve(binding.global_id)
        profile = manager.profiles.get(gid)
        known = 1 if profile is not None and bool(profile.known_name) else 0
        state_rank = {"provisional": 0, "confirmed": 1, "anchor": 2}.get(binding.state, 0)
        samples = int(profile.sample_count) if profile is not None else 0
        # Earlier track wins a final tie so the ID stays with the original owner.
        age_rank = -float(binding.first_seen)
        return (known, state_rank, age_rank, samples)

    def _same_pair(self, manager, a: LocalKey, b: LocalKey, now: float) -> bool:
        ba = manager.bindings.get(a)
        bb = manager.bindings.get(b)
        if ba is None or bb is None or a[0] == b[0]:
            return False
        if manager.room_of(a[0]) != manager.room_of(b[0]):
            return False
        ga = manager._resolve(ba.global_id)
        gb = manager._resolve(bb.global_id)
        if ga == gb:
            # Qwen confirmation strengthens an existing peer-camera match without
            # touching the global gallery directly; normal ReID commits remain gated.
            if ba.state == "provisional":
                ba.confirm_votes = max(ba.confirm_votes, manager.confirm_votes_required - 1)
            if bb.state == "provisional":
                bb.confirm_votes = max(bb.confirm_votes, manager.confirm_votes_required - 1)
            return True

        keep, move = (a, b)
        if self._binding_strength(manager, b) > self._binding_strength(manager, a):
            keep, move = b, a
        keep_gid = manager._resolve(manager.bindings[keep].global_id)
        manager._switch_binding(manager.bindings[move], move, keep_gid, now, provisional=True)
        self.merges += 1
        return True

    def _split_pair(self, manager, a: LocalKey, b: LocalKey, now: float) -> bool:
        ba = manager.bindings.get(a)
        bb = manager.bindings.get(b)
        if ba is None or bb is None:
            return False
        pair = self._pair_key(a, b)
        manager.cannot_link[pair] = now + 15.0
        self.cannot_links += 1
        ga = manager._resolve(ba.global_id)
        gb = manager._resolve(bb.global_id)
        if ga != gb:
            return True

        keep, move = (a, b)
        if self._binding_strength(manager, b) > self._binding_strength(manager, a):
            keep, move = b, a
        binding = manager.bindings[move]
        vector, color, count = manager._track_prototype(move)
        if vector is None or count < 2:
            return True
        (
            alt_gid,
            _alt_score,
            _second,
            _threshold,
            _reid,
            _color,
            _room,
            _covis,
            _context,
            accepted,
        ) = manager._candidate_decision(vector, color, move, now, exclude_gid=ga)
        if accepted and alt_gid is not None:
            manager._switch_binding(binding, move, int(alt_gid), now, provisional=True)
        else:
            quality = max((item.quality for item in manager._evidence(move)), default=0.5)
            manager._correct_to_new_anchor(
                binding,
                move,
                vector,
                color,
                binding.last_bbox,
                now,
                quality,
            )
        self.splits += 1
        return True

    def _consume_results(self, manager, now: float) -> None:
        while True:
            try:
                result = self.output_q.get_nowait()
            except queue.Empty:
                break
            pair = self._pair_key(result.key_a, result.key_b)
            self.pending_pairs.discard(pair)
            self.responses += 1
            self.last_latency_ms = result.latency_ms
            self.last_error = result.error
            self.last_verdict = result.verdict
            if result.error:
                self.failed += 1
                continue

            if result.verdict == "same" and result.confidence >= self.min_same_conf:
                self.same += 1
                self.same_votes[pair] += 1
                self.diff_votes[pair] = 0
                if self.same_votes[pair] >= self.same_votes_required:
                    if self._same_pair(manager, result.key_a, result.key_b, now):
                        self.same_votes[pair] = 0
            elif result.verdict == "different" and result.confidence >= self.min_diff_conf:
                self.different += 1
                self.diff_votes[pair] += 1
                self.same_votes[pair] = 0
                if self.diff_votes[pair] >= self.diff_votes_required:
                    if self._split_pair(manager, result.key_a, result.key_b, now):
                        self.diff_votes[pair] = 0
            else:
                self.uncertain += 1
                # Uncertain is deliberately a no-op. The fast ReID manager keeps
                # the binding provisional and will collect more visual evidence.

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "requests": self.requests,
            "responses": self.responses,
            "pending": len(self.pending_pairs),
            "same": self.same,
            "different": self.different,
            "uncertain": self.uncertain,
            "merges": self.merges,
            "splits": self.splits,
            "cannot_links": self.cannot_links,
            "failed": self.failed,
            "dropped": self.dropped,
            "latency_ms": self.last_latency_ms,
            "last_verdict": self.last_verdict,
            "error": self.last_error,
            "visual_tracks": len(self.visuals),
        }

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.input_q.put_nowait(None)
            except queue.Full:
                pass
