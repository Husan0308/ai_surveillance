from services.gpu.runtime.live_health import SourceFpsMonitor


def test_native_freshness_uses_oldest_required_stage():
    import json

    mon = SourceFpsMonitor(
        min_fps=5.0,
        max_age_sec=15.0,
        min_consecutive=2,
    )

    now_ns = 100_000_000_000

    for source_id in range(6):
        decode_age = 16 if source_id == 0 else 1

        sample = {
            "source_id": source_id,
            "camera_id": f"CAM-{source_id + 1:02d}",
            "monotonic_ns": now_ns,
            "fatal_metadata": False,
            "stages": [
                {
                    "count": 20,
                    "last_ns": now_ns - decode_age * 1_000_000_000,
                },
                {},
                {},
                {
                    "count": 20,
                    "last_ns": now_ns - 1_000_000_000,
                },
            ],
        }

        assert mon.ingest(
            "SIX_PROGRESS " + json.dumps(sample),
            now=1.0,
        )

    snap = mon.snapshot(now=1.0)

    assert snap["camera_stage_age_seconds"]["CAM-01"] == 16.0
    assert snap["camera_stage_age_seconds"]["CAM-02"] == 1.0
    assert snap["fresh"] is False

    # Future stage timestamps must remain invalid.
    sample["source_id"] = 0
    sample["camera_id"] = "CAM-01"
    sample["stages"][0]["last_ns"] = now_ns + 1

    assert not mon.ingest(
        "SIX_PROGRESS " + json.dumps(sample),
        now=2.0,
    )
