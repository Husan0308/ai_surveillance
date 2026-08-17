from __future__ import annotations

import json
import time
import urllib.request

from .qwen_reid_verifier import QwenRoomReIDVerifier, VerifyResult, VerifyTask


class FastQwenRoomReIDVerifier(QwenRoomReIDVerifier):
    """Low-latency policy layer on top of the conservative Qwen verifier.

    The base verifier owns visual memory, mutual-best candidate generation, queues,
    two-vote consensus and stale-result protection. This subclass only changes two
    things that matter on the GTX 1050 Ti deployment:

    1) ask Qwen for a single classification token instead of JSON/confidence text;
    2) allow two fresh Qwen SAME votes to override a temporary TAO appearance
       cannot-link between peer cameras in the same physical room.

    Qwen still cannot override same-camera or cross-room geometry because candidate
    generation never calls _same_pair for those impossible pairs.
    """

    def __init__(self) -> None:
        self.visual_different_until: dict[frozenset, float] = {}
        super().__init__()

    @staticmethod
    def _prompt() -> str:
        return (
            "LEFT side is person A; RIGHT side is person B. They come from two cameras "
            "covering the SAME physical room. Decide if A and B are the SAME real person. "
            "Compare body build, clothing layout/patterns, trousers, shoes, hair/head and "
            "accessories. Ignore background, camera angle, pose and lighting. If evidence is "
            "not sufficient, choose UNCERTAIN. Reply with exactly one word: SAME, DIFFERENT, "
            "or UNCERTAIN."
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

    def _same_pair(self, manager, a, b, now: float) -> bool:
        pair = self._pair_key(a, b)
        # A recent Qwen DIFFERENT decision is stronger than a subsequent TAO-only
        # candidate. Wait for the visual exclusion window to expire before merging.
        if self.visual_different_until.get(pair, 0.0) >= now:
            return False

        ba = manager.bindings.get(a)
        bb = manager.bindings.get(b)
        if ba is None or bb is None or a[0] == b[0]:
            return False
        if manager.room_of(a[0]) != manager.room_of(b[0]):
            return False

        # The base GlobalReIDManager may have temporarily cannot-linked this peer
        # pair solely because TAO cross-view similarity was low. Two independent
        # Qwen SAME votes are precisely the higher-level evidence intended to repair
        # that case, so clear that appearance-only block before the merge.
        manager.cannot_link.pop(pair, None)
        return super()._same_pair(manager, a, b, now)

    def _split_pair(self, manager, a, b, now: float) -> bool:
        pair = self._pair_key(a, b)
        self.visual_different_until[pair] = now + 30.0
        return super()._split_pair(manager, a, b, now)
