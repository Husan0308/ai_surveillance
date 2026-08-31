from __future__ import annotations

import threading
from collections import OrderedDict

from .step4_reid_gallery_v1 import GalleryViewV1
from .step4_reid_pair_scorer_v1 import GalleryPairScoreV1, score_gallery_pair_v1


# Step3 and Step4 run in the same process.  Step3's shadow scorer sees changed
# galleries first; Step4's matcher follows after a short phase delay.  Keep the
# exact immutable Step3 result so Step4 consumes evidence instead of recomputing
# the same pair.  The key contains every sample sequence plus embedding identity,
# so any gallery replacement/diversity update naturally creates a cache miss.
_PAIR_CACHE_MAX = 4096
_PAIR_CACHE_LOCK = threading.Lock()
_PAIR_CACHE: OrderedDict[
    tuple[
        tuple[str, str, tuple[tuple[int, int], ...]],
        tuple[str, str, tuple[tuple[int, int], ...]],
    ],
    GalleryPairScoreV1,
] = OrderedDict()


def _view_fingerprint(
    view: GalleryViewV1,
) -> tuple[str, str, tuple[tuple[int, int], ...]]:
    return (
        str(view.camera_id),
        str(view.local_track_id),
        tuple(
            (int(sample.sample_sequence), id(sample.embedding))
            for sample in view.samples
        ),
    )


def score_gallery_pair_step3_cached_v1(
    first: GalleryViewV1,
    second: GalleryViewV1,
) -> GalleryPairScoreV1:
    """Return the authoritative Step3 score, reusing it across shadow consumers.

    Cache misses call ``score_gallery_pair_v1`` unchanged.  Therefore all score
    fields, insufficient/invalid semantics and the fixed robust-score formula are
    exactly Step3's existing implementation; only duplicate evaluation is removed.
    """

    key = (_view_fingerprint(first), _view_fingerprint(second))
    with _PAIR_CACHE_LOCK:
        cached = _PAIR_CACHE.get(key)
        if cached is not None:
            _PAIR_CACHE.move_to_end(key)
            return cached

    result = score_gallery_pair_v1(
        [sample.embedding for sample in first.samples],
        [sample.embedding for sample in second.samples],
    )

    with _PAIR_CACHE_LOCK:
        existing = _PAIR_CACHE.get(key)
        if existing is not None:
            _PAIR_CACHE.move_to_end(key)
            return existing
        _PAIR_CACHE[key] = result
        _PAIR_CACHE.move_to_end(key)
        while len(_PAIR_CACHE) > _PAIR_CACHE_MAX:
            _PAIR_CACHE.popitem(last=False)
    return result
