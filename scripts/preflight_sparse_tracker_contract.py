#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fail(message: str) -> int:
    print(f"SPARSE_TRACKER_PREFLIGHT=FAIL {message}", flush=True)
    return 2


def main() -> int:
    source_path = ROOT / "services/camera_v2/native_sparse_tracker_contract.c"
    try:
        source = source_path.read_text(encoding="utf-8")
    except Exception as exc:
        return fail(f"native source unreadable: {type(exc).__name__}: {exc}")

    for required in (
        "camera_v2_mark_batch_infer_done",
        "frame_meta->bInferDone = TRUE",
        "batch_meta->frame_meta_list",
    ):
        if required not in source:
            return fail(f"native contract missing: {required}")

    try:
        from services.camera_v2.sparse_tracker_contract import (
            ensure_sparse_tracker_bridge,
            install_sparse_tracker_contract,
        )

        library = ensure_sparse_tracker_bridge()
        install_sparse_tracker_contract()
        from services.camera_v2.person_tracking_final import CameraPersonTrackingFinal
    except Exception as exc:
        return fail(f"bridge unavailable: {type(exc).__name__}: {exc}")

    if not getattr(CameraPersonTrackingFinal, "_sparse_tracker_contract_installed", False):
        return fail("CameraPersonTrackingFinal probe wrapper was not installed")

    print(
        "SPARSE_TRACKER_PREFLIGHT=PASS "
        f"library={library} contract=bInferDone-all-mux-frames "
        "fresh-rfdetr-metadata=existing-probe nvdcf=per-frame",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
