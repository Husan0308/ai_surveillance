from __future__ import annotations

import json
import os
import time
import urllib.request
from collections import defaultdict

from .qwen_reid_verifier import QwenRoomReIDVerifier, VerifyResult, VerifyTask


class FastQwenRoomReIDVerifier(QwenRoomReIDVerifier):
    """Low-latency peer-camera verifier with tracklet consensus before Qwen.

    The important rule is that a single frame is never identity truth. For each
    same-room camera pair we compare the recent ReID tracklets (all recent crops),
    solve a mutual-best one-to-one pairing, and require that pairing to persist
    over several scans before a high-confidence automatic merge. Qwen is only used
    for the ambiguous middle band and for auditing suspicious already-shared IDs.

    This is intentionally closer to video/person-ReID practice than asking a VLM
    to decide identity from one crop. Geometry hard rules remain above both layers:
    same-camera tracks and simultaneous tracks from different rooms can never merge.
    """

    def __init__(self) -> None:
        self.visual_different_until: dict[frozenset, float] = {}
        self.appearance_votes: dict[frozenset, int] = defaultdict(int)
        self._pair_margins: dict[frozenset, float] = {}
        self.auto_same_min = float(os.environ.get("CAMERA_V2_ROOM_AUTO_SAME", "0.52"))
        self.auto_same_margin = float(os.environ.get("CAMERA_V2_ROOM_AUTO_MARGIN", "0.045"))
        self.auto_same_votes = max(3, int(os.environ.get("CAMERA_V2_ROOM_AUTO_VOTES", "4")))
        self.diff_reid_ceiling = float(os.environ.get("CAMERA_V2_QWEN_DIFF_REID_MAX", "0.38"))
        self.auto_merges = 0
        self.last_pair_score = -1.0
        self.last_pair_margin = -1.0
        super().__init__()
        # General VLM verification is allowed to inspect a wider candidate band,
        # while automatic merges stay much stricter and require temporal consensus.
        self.min_peer_reid = float(os.environ.get("CAMERA_V2_QWEN_MIN_PEER_REID", "0.28"))

    @staticmethod
    def _prompt() -> str:
        return (
            "LEFT side is person A; RIGHT side is person B. They are from two cameras "
            "covering the SAME physical room. Decide if A and B are the SAME real person. "
            "Compare stable cues across all shown frames: body build, clothing layout/patterns, "
            "trousers, shoes, hair/head and accessories. Ignore background, viewpoint, pose and "
            "lighting. If any important cue conflicts or evidence is weak, choose UNCERTAIN. "
            "Reply with exactly one word: SAME, DIFFERENT, or UNCERTAIN."
        )

    def _request(self, task: VerifyTask) -> VerifyResult:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_url(task.image)},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
            text = str(
                data.get("choices", [{}])[0].get("message", {}).get("content", "")
            ).strip().upper()
            if "DIFFERENT" in text:
                verdict, confidence = "different", 1.0
            elif "UNCERTAIN" in text:
                verdict, confidence = "uncertain", 0.0
            elif "SAME" in text:
                verdict, confidence = "same", 1.0
            else:
                verdict, confidence = "uncertain", 0.0
            return VerifyResult(
                task.key_a,
                task.key_b,
                task.gid_a,
                task.gid_b,
                task.room_id,
                verdict,
                confidence,
                task.peer_reid,
                task.reason,
                task.submitted_at,
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
                task.peer_reid,
                task.reason,
                task.submitted_at,
                (time.monotonic() - started) * 1000.0,
                f"{type(exc).__name__}: {exc}",
            )

    def _tracklet_similarity(self, manager, a, b) -> float:
        """Robust cross-camera similarity from many recent crops, not one mean."""
        ea = manager._evidence(a)[-7:]
        eb = manager._evidence(b)[-7:]
        if not ea or not eb:
            return -1.0

        sims = sorted(
            (self._dot(x.vector, y.vector) for x in ea for y in eb),
            reverse=True,
        )
        if not sims:
            return -1.0
        top = sims[: min(5, len(sims))]
        top_mean = sum(top) / len(top)

        va, ca, _na = manager._track_prototype(a)
        vb, cb, _nb = manager._track_prototype(b)
        proto = self._dot(va, vb)
        score = 0.48 * top[0] + 0.34 * top_mean + 0.18 * proto

        # Clothing colour is only a tiny stabilizer. It must never rescue a bad
        # person embedding on its own because workers can wear similar colours.
        if ca and cb:
            color = self._dot(ca, cb)
            score = 0.96 * score + 0.04 * color
        return float(score)

    def _candidate_pairs(self, manager, now: float):
        active = [
            key
            for key in manager.bindings
            if manager._is_active(key, now) and self._track_strip(key, now) is not None
        ]
        by_room = defaultdict(list)
        for key in active:
            by_room[manager.room_of(key[0])].append(key)

        output = []
        self._pair_margins = {}
        for room_id, keys in by_room.items():
            raw = []
            scores_for = defaultdict(list)
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    if a[0] == b[0]:
                        continue
                    score = self._tracklet_similarity(manager, a, b)
                    if score < -0.5:
                        continue
                    ga = manager._resolve(manager.bindings[a].global_id)
                    gb = manager._resolve(manager.bindings[b].global_id)
                    same_gid = ga == gb
                    scores_for[a].append((score, b))
                    scores_for[b].append((score, a))
                    if same_gid:
                        provisional = (
                            manager.bindings[a].state == "provisional"
                            or manager.bindings[b].state == "provisional"
                        )
                        if not provisional and score >= self.audit_same_gid_below:
                            continue
                        reason = "same_gid_audit"
                        priority = 2.0 + score
                    else:
                        if score < self.min_peer_reid:
                            continue
                        reason = "peer_candidate"
                        priority = 1.0 + score
                    raw.append((priority, a, b, ga, gb, score, reason))

            best_for = {}
            margin_for = {}
            for key, rows in scores_for.items():
                rows.sort(reverse=True, key=lambda row: row[0])
                if rows:
                    best_for[key] = rows[0][1]
                    second = rows[1][0] if len(rows) > 1 else -1.0
                    margin_for[key] = rows[0][0] - second if second >= 0.0 else 1.0

            for priority, a, b, ga, gb, score, reason in raw:
                if ga != gb and (best_for.get(a) != b or best_for.get(b) != a):
                    continue
                pair = self._pair_key(a, b)
                self._pair_margins[pair] = min(
                    float(margin_for.get(a, 0.0)), float(margin_for.get(b, 0.0))
                )
                output.append((priority, a, b, ga, gb, room_id, score, reason))

        output.sort(reverse=True, key=lambda row: row[0])
        return output

    def _service_tracklet_consensus(self, manager, now: float) -> None:
        seen_pairs = set()
        for _priority, a, b, ga, gb, _room, score, _reason in self._candidate_pairs(manager, now):
            pair = self._pair_key(a, b)
            seen_pairs.add(pair)
            margin = float(self._pair_margins.get(pair, 0.0))
            self.last_pair_score = float(score)
            self.last_pair_margin = margin

            if ga == gb:
                self.appearance_votes[pair] = 0
                continue
            if self.visual_different_until.get(pair, 0.0) >= now:
                self.appearance_votes[pair] = 0
                continue

            if score >= self.auto_same_min and margin >= self.auto_same_margin:
                self.appearance_votes[pair] += 1
                if self.appearance_votes[pair] >= self.auto_same_votes:
                    if self._same_pair(manager, a, b, now):
                        self.auto_merges += 1
                    self.appearance_votes[pair] = 0
            else:
                self.appearance_votes[pair] = max(0, self.appearance_votes[pair] - 1)

        for pair in list(self.appearance_votes):
            if pair not in seen_pairs:
                self.appearance_votes[pair] = max(0, self.appearance_votes[pair] - 1)
                if self.appearance_votes[pair] == 0:
                    self.appearance_votes.pop(pair, None)

    def service(self, manager, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        # First let stable specialized ReID tracklets solve the easy one-to-one
        # room pairs. Then the base service queues Qwen only for unresolved pairs.
        self._consume_results(manager, now)
        if now - self.last_scan >= self.scan_interval:
            self._service_tracklet_consensus(manager, now)
        super().service(manager, now)

    def _same_pair(self, manager, a, b, now: float) -> bool:
        pair = self._pair_key(a, b)
        if self.visual_different_until.get(pair, 0.0) >= now:
            return False

        ba = manager.bindings.get(a)
        bb = manager.bindings.get(b)
        if ba is None or bb is None or a[0] == b[0]:
            return False
        if manager.room_of(a[0]) != manager.room_of(b[0]):
            return False

        # Only peer-camera appearance blocks may be cleared here. Same-camera and
        # cross-room impossibilities never reach this method.
        manager.cannot_link.pop(pair, None)
        return super()._same_pair(manager, a, b, now)

    def _split_pair(self, manager, a, b, now: float) -> bool:
        pair = self._pair_key(a, b)
        score = self._tracklet_similarity(manager, a, b)
        # A 2B VLM is not a person-ReID authority. If specialized multi-frame ReID
        # still says the pair is reasonably compatible, Qwen DIFFERENT only vetoes
        # future auto-merge briefly; it does not tear an existing Global ID apart.
        self.visual_different_until[pair] = now + 30.0
        if score > self.diff_reid_ceiling:
            return False
        return super()._split_pair(manager, a, b, now)

    def snapshot(self) -> dict:
        row = super().snapshot()
        row.update(
            {
                "auto_merges": self.auto_merges,
                "appearance_vote_pairs": len(self.appearance_votes),
                "last_pair_score": self.last_pair_score,
                "last_pair_margin": self.last_pair_margin,
            }
        )
        return row
