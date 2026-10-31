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


def _generation_progress(source_id, second, generation, count):
    import json

    now_ns = second * 1_000_000_000
    record = {
        "source_id": source_id,
        "camera_id": f"CAM-{source_id + 1:02d}",
        "source_generation": generation,
        "monotonic_ns": now_ns,
        "fatal_metadata": False,
        "stages": [
            {"count": count, "last_ns": now_ns - 100_000_000},
            {},
            {},
            {"count": count, "last_ns": now_ns - 100_000_000},
        ],
    }
    return "SIX_PROGRESS " + json.dumps(record)


def test_generation_transition_rebaselines_counters():
    monitor = SourceFpsMonitor(min_fps=5.0, min_consecutive=2)
    # Two positive deltas after baseline establish readiness for six sources.
    for second in (1, 2, 3):
        for source_id in range(6):
            assert monitor.ingest(
                _generation_progress(source_id, second, 0, second * 10),
                now=float(second),
            )
    assert monitor.snapshot(now=3.0)["ready"]

    # Source 0 reconnects and its native counters restart at one.
    assert monitor.ingest(_generation_progress(0, 4, 1, 1), now=4.0)
    snapshot = monitor.snapshot(now=4.0)
    assert not snapshot["ready"]
    assert snapshot["camera_progress"]["CAM-01"]["source_generation"] == 1
    assert snapshot["camera_progress"]["CAM-01"]["decode_fps"] == 0.0
    assert snapshot["camera_progress"]["CAM-01"]["consecutive_good"] == 0

    # A delayed older-generation record cannot replace the new baseline.
    assert not monitor.ingest(_generation_progress(0, 5, 0, 999), now=5.0)
    assert monitor.snapshot(now=5.0)["camera_progress"]["CAM-01"]["decode_count"] == 1

    assert monitor.ingest(_generation_progress(0, 5, 1, 11), now=5.0)
    assert not monitor.snapshot(now=5.0)["ready"]
    assert monitor.ingest(_generation_progress(0, 6, 1, 21), now=6.0)
    assert monitor.snapshot(now=6.0)["ready"]


def test_counter_reset_and_invalid_generation_fail_closed():
    monitor = SourceFpsMonitor(min_fps=5.0, min_consecutive=2)
    for second in (1, 2, 3):
        for source_id in range(6):
            assert monitor.ingest(
                _generation_progress(source_id, second, 2, second * 10),
                now=float(second),
            )
    assert monitor.snapshot(now=3.0)["ready"]

    # An unexplained counter reset is never counted as forward progress.
    assert monitor.ingest(_generation_progress(0, 4, 2, 1), now=4.0)
    assert not monitor.snapshot(now=4.0)["ready"]
    assert monitor.snapshot(now=4.0)["camera_progress"]["CAM-01"]["decode_fps"] == 0.0

    for invalid in (-1, True, "3"):
        assert not monitor.ingest(_generation_progress(0, 5, invalid, 100), now=5.0)
    assert not monitor.ingest(_generation_progress(0, 3, 2, 100), now=5.0)
    assert not monitor.ingest(_generation_progress(0, 5, None, 100), now=5.0)
    assert monitor.snapshot(now=5.0)["camera_progress"]["CAM-01"]["decode_count"] == 1

    assert monitor.ingest(_generation_progress(0, 5, 2, 11), now=5.0)
    assert not monitor.snapshot(now=5.0)["ready"]
    assert monitor.ingest(_generation_progress(0, 6, 2, 21), now=6.0)
    assert monitor.snapshot(now=6.0)["ready"]
