from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class _EvidenceSample:
    crop_bgr: np.ndarray
    embedding: np.ndarray
    local_track: str
    captured_ns: int
    quality: float


class V11RoomEvidenceWriterV1:
    """Debug-only multi-shot crop writer for manually auditing Room Identity fusion.

    The production matcher remains untouched. This writer stores a small rolling
    gallery per camera identity, compares every cross-camera sample pair with the
    already-computed L2-normalized ReID embeddings, writes a pair-score matrix and a
    side-by-side comparison sheet, and labels weak individual crops as SUSPECT.
    """

    def __init__(
        self,
        root: str,
        *,
        max_per_member: int = 4,
        min_interval_sec: float = 0.75,
        max_cached_members: int = 64,
        cache_per_member: int = 6,
        suspect_score: float = 0.60,
        audit_max_score: float = 0.68,
        audit_top2_mean: float = 0.62,
        audit_min_support: int = 2,
    ) -> None:
        text = str(root).strip()
        self.enabled = bool(text)
        self.root = Path(text).expanduser() if text else None
        self.max_per_member = max(1, min(20, int(max_per_member)))
        self.min_interval_ns = int(max(0.1, float(min_interval_sec)) * 1_000_000_000.0)
        self.max_cached_members = max(8, int(max_cached_members))
        self.cache_per_member = max(self.max_per_member, min(12, int(cache_per_member)))
        self.suspect_score = max(0.0, min(1.0, float(suspect_score)))
        self.audit_max_score = max(0.0, min(1.0, float(audit_max_score)))
        self.audit_top2_mean = max(0.0, min(1.0, float(audit_top2_mean)))
        self.audit_min_support = max(1, int(audit_min_support))
        self._cache: OrderedDict[
            tuple[str, str], list[_EvidenceSample]
        ] = OrderedDict()
        self._saved: dict[tuple[str, str, str], int] = {}
        self._last_saved_ns: dict[tuple[str, str, str], int] = {}
        self.write_ok = 0
        self.write_fail = 0
        self.audit_pass = 0
        self.audit_suspect = 0
        if self.enabled and self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        row = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(row))
        if not np.isfinite(norm) or norm <= 1e-9:
            raise ValueError("invalid ReID embedding for room evidence")
        return row / norm

    def remember(
        self,
        *,
        camera_id: str,
        camera_identity: str,
        local_track: str,
        crop_bgr: np.ndarray,
        embedding: np.ndarray,
        quality: float,
        captured_ns: int,
    ) -> None:
        if not self.enabled or not camera_identity or crop_bgr.size == 0:
            return
        key = (str(camera_id), str(camera_identity))
        sample = _EvidenceSample(
            crop_bgr=np.ascontiguousarray(crop_bgr.copy()),
            embedding=self._normalize(embedding),
            local_track=str(local_track),
            captured_ns=int(captured_ns),
            quality=max(0.0, min(1.0, float(quality))),
        )
        rows = self._cache.setdefault(key, [])
        rows.append(sample)
        if len(rows) > self.cache_per_member:
            del rows[:-self.cache_per_member]
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_cached_members:
            self._cache.popitem(last=False)

    def _best_peer_score(
        self,
        member: tuple[str, str],
        sample: _EvidenceSample,
        members: set[tuple[str, str]],
    ) -> float:
        best = -1.0
        for peer in members:
            if peer == member or peer[0] == member[0]:
                continue
            for peer_sample in self._cache.get(peer, []):
                best = max(best, float(np.dot(sample.embedding, peer_sample.embedding)))
        return best

    def _pair_rows(
        self,
        members: set[tuple[str, str]],
    ) -> list[tuple[float, tuple[str, str], int, _EvidenceSample, tuple[str, str], int, _EvidenceSample]]:
        rows = []
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                if left[0] == right[0]:
                    continue
                for left_index, left_sample in enumerate(self._cache.get(left, []), start=1):
                    for right_index, right_sample in enumerate(self._cache.get(right, []), start=1):
                        score = float(np.dot(left_sample.embedding, right_sample.embedding))
                        rows.append(
                            (
                                score,
                                left,
                                left_index,
                                left_sample,
                                right,
                                right_index,
                                right_sample,
                            )
                        )
        rows.sort(key=lambda row: row[0], reverse=True)
        return rows

    def _write_pair_audit(
        self,
        *,
        room_identity: str,
        members: set[tuple[str, str]],
    ) -> list[dict[str, object]]:
        if not self.enabled or self.root is None:
            return []
        folder = self.root / str(room_identity)
        folder.mkdir(parents=True, exist_ok=True)
        pair_rows = self._pair_rows(members)

        pair_path = folder / "pair_scores.tsv"
        with pair_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "score\tleft_camera\tleft_identity\tleft_track\tleft_ns\t"
                "right_camera\tright_identity\tright_track\tright_ns\n"
            )
            for score, left, _li, left_sample, right, _ri, right_sample in pair_rows:
                handle.write(
                    f"{score:.6f}\t{left[0]}\t{left[1]}\t{left_sample.local_track}\t"
                    f"{left_sample.captured_ns}\t{right[0]}\t{right[1]}\t"
                    f"{right_sample.local_track}\t{right_sample.captured_ns}\n"
                )

        grouped: dict[
            tuple[tuple[str, str], tuple[str, str]], list[float]
        ] = {}
        for score, left, _li, _ls, right, _ri, _rs in pair_rows:
            grouped.setdefault((left, right), []).append(score)

        audits: list[dict[str, object]] = []
        summary_path = folder / "audit.tsv"
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "status\tleft_camera\tleft_identity\tright_camera\tright_identity\t"
                "max\ttop2_mean\tmedian\tsupport\tpairs\n"
            )
            for (left, right), scores in sorted(grouped.items()):
                ranked = sorted(scores, reverse=True)
                max_score = ranked[0]
                top2_mean = float(np.mean(ranked[: min(2, len(ranked))]))
                median = float(np.median(ranked))
                support = sum(score >= self.suspect_score for score in ranked)
                status = "PASS" if (
                    max_score >= self.audit_max_score
                    and top2_mean >= self.audit_top2_mean
                    and support >= self.audit_min_support
                ) else "SUSPECT"
                audits.append(
                    {
                        "status": status,
                        "left": left,
                        "right": right,
                        "max": max_score,
                        "top2_mean": top2_mean,
                        "median": median,
                        "support": support,
                        "pairs": len(ranked),
                    }
                )
                handle.write(
                    f"{status}\t{left[0]}\t{left[1]}\t{right[0]}\t{right[1]}\t"
                    f"{max_score:.6f}\t{top2_mean:.6f}\t{median:.6f}\t"
                    f"{support}\t{len(ranked)}\n"
                )

        self.audit_pass += sum(row["status"] == "PASS" for row in audits)
        self.audit_suspect += sum(row["status"] == "SUSPECT" for row in audits)
        self._write_comparison_sheet(folder=folder, pair_rows=pair_rows)
        return audits

    @staticmethod
    def _fit_thumb(image: np.ndarray, width: int, height: int) -> np.ndarray:
        import cv2

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if image.size == 0:
            return canvas
        scale = min(width / image.shape[1], height / image.shape[0])
        new_w = max(1, int(round(image.shape[1] * scale)))
        new_h = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    def _write_comparison_sheet(
        self,
        *,
        folder: Path,
        pair_rows: list[tuple[float, tuple[str, str], int, _EvidenceSample, tuple[str, str], int, _EvidenceSample]],
    ) -> None:
        if not pair_rows:
            return
        try:
            import cv2

            chosen = pair_rows[: min(4, len(pair_rows))]
            thumb_w, thumb_h, header_h = 220, 300, 34
            canvas = np.zeros(
                (len(chosen) * (thumb_h + header_h), thumb_w * 2, 3),
                dtype=np.uint8,
            )
            for row_index, (score, left, _li, left_sample, right, _ri, right_sample) in enumerate(chosen):
                y0 = row_index * (thumb_h + header_h)
                left_thumb = self._fit_thumb(left_sample.crop_bgr, thumb_w, thumb_h)
                right_thumb = self._fit_thumb(right_sample.crop_bgr, thumb_w, thumb_h)
                canvas[y0 + header_h : y0 + header_h + thumb_h, :thumb_w] = left_thumb
                canvas[y0 + header_h : y0 + header_h + thumb_h, thumb_w:] = right_thumb
                text = (
                    f"{left[0]}/{left_sample.local_track} <-> "
                    f"{right[0]}/{right_sample.local_track} score={score:.3f}"
                )
                cv2.putText(
                    canvas,
                    text,
                    (6, y0 + 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(folder / "comparison.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception:
            self.write_fail += 1

    def _write_one(
        self,
        *,
        room_id: str,
        room_identity: str,
        member: tuple[str, str],
        members: set[tuple[str, str]],
        force: bool,
    ) -> bool:
        if not self.enabled or self.root is None:
            return False
        cached_rows = self._cache.get(member, [])
        if not cached_rows:
            return False
        sample = cached_rows[-1]
        camera_id, camera_identity = member
        save_key = (str(room_identity), camera_id, camera_identity)
        count = self._saved.get(save_key, 0)
        if count >= self.max_per_member:
            return False
        previous_ns = self._last_saved_ns.get(save_key, 0)
        if not force and previous_ns and sample.captured_ns - previous_ns < self.min_interval_ns:
            return False

        best_peer_score = self._best_peer_score(member, sample, members)
        label = "OK" if best_peer_score >= self.suspect_score else "SUSPECT"
        folder = self.root / str(room_identity)
        folder.mkdir(parents=True, exist_ok=True)
        index = count + 1
        safe_track = sample.local_track.replace("/", "_")
        filename = (
            f"{label}__{camera_id}__{camera_identity}__{index:02d}__"
            f"{safe_track}__peer{best_peer_score:.3f}.jpg"
        )
        path = folder / filename

        try:
            import cv2

            ok = bool(cv2.imwrite(str(path), sample.crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]))
        except Exception:
            ok = False
        if not ok:
            self.write_fail += 1
            return False

        manifest = folder / "manifest.tsv"
        if not manifest.exists():
            manifest.write_text(
                "room\troom_identity\tcamera\tcamera_identity\tlocal_track\t"
                "quality\tbest_peer_score\tlabel\tfile\n",
                encoding="utf-8",
            )
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{room_id}\t{room_identity}\t{camera_id}\t{camera_identity}\t"
                f"{sample.local_track}\t{sample.quality:.4f}\t{best_peer_score:.6f}\t"
                f"{label}\t{filename}\n"
            )
        self._saved[save_key] = index
        self._last_saved_ns[save_key] = sample.captured_ns
        self.write_ok += 1
        return True

    def capture_room(
        self,
        *,
        room_id: str,
        room_identity: str,
        members: set[tuple[str, str]],
        current_member: tuple[str, str],
    ) -> dict[str, object]:
        if not self.enabled or len(members) < 2:
            return {"saved": [], "audits": []}
        saved: list[str] = []
        # Backfill one cached crop for every fused camera identity immediately so a
        # newly-created folder already contains evidence from both cameras.
        for member in sorted(members):
            save_key = (str(room_identity), member[0], member[1])
            if self._saved.get(save_key, 0) == 0 and self._write_one(
                room_id=room_id,
                room_identity=room_identity,
                member=member,
                members=members,
                force=True,
            ):
                saved.append(f"{member[0]}/{member[1]}")

        # Then collect a few temporally separated crops from the currently observed
        # member. This is bounded by max_per_member and does not grow indefinitely.
        if self._write_one(
            room_id=room_id,
            room_identity=room_identity,
            member=current_member,
            members=members,
            force=False,
        ):
            saved.append(f"{current_member[0]}/{current_member[1]}")

        audits = self._write_pair_audit(room_identity=room_identity, members=members)
        return {"saved": saved, "audits": audits}

    def snapshot(self) -> dict[str, int]:
        return {
            "enabled": int(self.enabled),
            "cached_members": len(self._cache),
            "write_ok": self.write_ok,
            "write_fail": self.write_fail,
            "audit_pass": self.audit_pass,
            "audit_suspect": self.audit_suspect,
        }
