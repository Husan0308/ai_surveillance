from __future__ import annotations

"""Strict person-only post-processing for RF-DETR.

This ports the useful parts of the earlier stable detection stack without pulling
its service/UI architecture back into Camera V2: exact person-class filtering,
valid-geometry checks and a second duplicate/containment suppression pass.

The RF-DETR checkpoint normally exposes COCO category id 1 for ``person``.  When
human-readable class names are available they are authoritative and only the
literal normalized name ``person`` is accepted.  If names are unavailable, the
fallback ids are explicit/configurable instead of treating both ids 0 and 1 as
persons.
"""

import math
import os
from dataclasses import dataclass

import numpy as np


def _parse_ids(raw: str) -> tuple[int, ...]:
    output: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value not in output:
            output.append(value)
    return tuple(output) or (1,)


def _area(box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection(a, b) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a, b) -> float:
    acx = (float(a[0]) + float(a[2])) * 0.5
    acy = (float(a[1]) + float(a[3])) * 0.5
    bcx = (float(b[0]) + float(b[2])) * 0.5
    bcy = (float(b[1]) + float(b[3])) * 0.5
    scale = max(20.0, math.sqrt(max(_area(a), _area(b), 1.0)))
    return math.hypot(acx - bcx, acy - bcy) / scale


@dataclass(frozen=True)
class PersonFilterStats:
    raw: int
    class_rejected: int
    geometry_rejected: int
    duplicate_rejected: int
    kept: int
    class_mode: str
    raw_ids: tuple[int, ...]
    raw_names: tuple[str, ...]


class PersonCandidateFilter:
    def __init__(self) -> None:
        self.person_ids = _parse_ids(os.environ.get("CAMERA_V2_PERSON_CLASS_IDS", "1"))
        self.duplicate_iou = float(os.environ.get("CAMERA_V2_PERSON_DEDUP_IOU", "0.58"))
        self.duplicate_containment = float(
            os.environ.get("CAMERA_V2_PERSON_DEDUP_CONTAINMENT", "0.84")
        )
        self.duplicate_center = float(
            os.environ.get("CAMERA_V2_PERSON_DEDUP_CENTER", "0.40")
        )
        # Deliberately conservative. Far-away or seated people remain valid;
        # only degenerate/tiny RF-DETR fragments are removed here.
        self.min_width = float(os.environ.get("CAMERA_V2_PERSON_MIN_WIDTH", "6"))
        self.min_height = float(os.environ.get("CAMERA_V2_PERSON_MIN_HEIGHT", "10"))
        self.min_area = float(os.environ.get("CAMERA_V2_PERSON_MIN_AREA", "80"))

    @staticmethod
    def _class_names(detections, count: int) -> np.ndarray | None:
        data = getattr(detections, "data", None)
        if not isinstance(data, dict):
            return None
        names = data.get("class_name")
        if names is None:
            return None
        names = np.asarray(names).astype(str)
        if len(names) != count:
            return None
        return np.char.lower(np.char.strip(names))

    def filter(self, detections, max_det: int) -> tuple[list[tuple[list[float], float]], PersonFilterStats]:
        xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32)
        confidence = np.asarray(getattr(detections, "confidence", []), dtype=np.float32)
        class_id = np.asarray(getattr(detections, "class_id", []), dtype=np.int64)

        valid_shape = xyxy.ndim == 2 and xyxy.shape[-1:] == (4,)
        if not valid_shape or len(xyxy) != len(confidence) or len(xyxy) != len(class_id):
            stats = PersonFilterStats(
                raw=int(len(xyxy) if xyxy.ndim else 0),
                class_rejected=0,
                geometry_rejected=int(len(xyxy) if xyxy.ndim else 0),
                duplicate_rejected=0,
                kept=0,
                class_mode="invalid",
                raw_ids=tuple(),
                raw_names=tuple(),
            )
            return [], stats

        raw_count = len(xyxy)
        names = self._class_names(detections, raw_count)
        if names is not None:
            person_mask = names == "person"
            class_mode = "name"
            raw_names = tuple(sorted(set(str(v) for v in names.tolist())))
        else:
            person_mask = np.isin(class_id, np.asarray(self.person_ids, dtype=np.int64))
            class_mode = "id"
            raw_names = tuple()

        raw_ids = tuple(sorted(set(int(v) for v in class_id.tolist())))
        class_indices = np.flatnonzero(person_mask)
        class_rejected = raw_count - len(class_indices)

        candidates: list[tuple[list[float], float]] = []
        geometry_rejected = 0
        for idx in class_indices:
            box = [float(v) for v in xyxy[int(idx)]]
            score = float(confidence[int(idx)])
            if not math.isfinite(score) or not all(math.isfinite(v) for v in box):
                geometry_rejected += 1
                continue
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if (
                score <= 0.0
                or width < self.min_width
                or height < self.min_height
                or width * height < self.min_area
            ):
                geometry_rejected += 1
                continue
            candidates.append((box, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        kept: list[tuple[list[float], float]] = []
        duplicate_rejected = 0
        for box, score in candidates:
            duplicate = False
            for other, _other_score in kept:
                if _iou(box, other) >= self.duplicate_iou or (
                    _containment(box, other) >= self.duplicate_containment
                    and _center_distance(box, other) <= self.duplicate_center
                ):
                    duplicate = True
                    break
            if duplicate:
                duplicate_rejected += 1
                continue
            kept.append((box, score))
            if len(kept) >= max(1, int(max_det)):
                break

        stats = PersonFilterStats(
            raw=raw_count,
            class_rejected=int(class_rejected),
            geometry_rejected=int(geometry_rejected),
            duplicate_rejected=int(duplicate_rejected),
            kept=len(kept),
            class_mode=class_mode,
            raw_ids=raw_ids,
            raw_names=raw_names,
        )
        return kept, stats
