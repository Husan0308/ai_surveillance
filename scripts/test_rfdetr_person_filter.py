from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.person_candidate_filter import PersonCandidateFilter


def fake(xyxy, confidence, class_id, names=None):
    data = {} if names is None else {"class_name": np.asarray(names, dtype=object)}
    return SimpleNamespace(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        class_id=np.asarray(class_id, dtype=np.int64),
        data=data,
    )


def main() -> int:
    # Human-readable class names are authoritative even when category ids differ.
    named = fake(
        [
            [10, 10, 100, 220],       # person, strongest
            [14, 14, 96, 216],        # nested duplicate person
            [200, 100, 350, 190],     # seated/wide person
            [380, 80, 470, 170],      # chair/non-person
            [500, 100, 503, 106],     # tiny invalid person fragment
        ],
        [0.91, 0.79, 0.62, 0.95, 0.88],
        [0, 0, 0, 1, 0],
        ["person", "person", "person", "chair", "person"],
    )
    filt = PersonCandidateFilter()
    rows, stats = filt.filter(named, 40)
    assert stats.class_mode == "name", stats
    assert stats.raw == 5, stats
    assert stats.class_rejected == 1, stats
    assert stats.geometry_rejected == 1, stats
    assert stats.duplicate_rejected == 1, stats
    assert len(rows) == 2, (rows, stats)
    # Wide seated person must survive; no standing-person aspect-ratio rule exists.
    assert any((box[2] - box[0]) > (box[3] - box[1]) for box, _ in rows), rows

    # When class names are unavailable, fallback is explicit COCO person id 1.
    os.environ.pop("CAMERA_V2_PERSON_CLASS_IDS", None)
    fallback_filter = PersonCandidateFilter()
    fallback = fake(
        [[10, 10, 100, 220], [200, 20, 290, 220]],
        [0.93, 0.92],
        [0, 1],
        None,
    )
    rows2, stats2 = fallback_filter.filter(fallback, 40)
    assert stats2.class_mode == "id", stats2
    assert fallback_filter.person_ids == (1,), fallback_filter.person_ids
    assert len(rows2) == 1, (rows2, stats2)
    assert stats2.class_rejected == 1, stats2

    print(
        "RFDETR_PERSON_FILTER_TEST=PASS "
        "name_person_only=1 fallback_id1_only=1 dedup=1 tiny_reject=1 seated_wide_kept=1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
