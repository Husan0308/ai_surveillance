import json
from pathlib import Path

from scripts.dev_room_mv3dt.run_room_pair import (
    deepstream_command,
    live_config,
    apply_test_camera_latency_overrides,
    apply_test_streammux_timeout,
    record_test_streammux_timeout_override,
)


def test_live_room_pair_staging_uses_each_camera_latency(tmp_path: Path) -> None:
    config = tmp_path / "config_deepstream.txt"
    config.write_text(
        "[source0]\n"
        "type=3\n"
        "enable=1\n"
        "cudadec-memtype=0\n"
        "gpu-id=0\n"
        "num-sources=1\n"
        "uri=file:///workspace/inputs/videos/cam_00.mp4\n"
        "[source1]\n"
        "type=3\n"
        "enable=1\n"
        "cudadec-memtype=0\n"
        "gpu-id=0\n"
        "num-sources=1\n"
        "uri=file:///workspace/inputs/videos/cam_01.mp4\n"
        "[streammux]\n"
        "live-source=0\n"
    )

    live_config(
        tmp_path,
        {"CAM-01": "rtsp://cam01.example/live", "CAM-04": "rtsp://cam04.example/live"},
        {"CAM-01": 20, "CAM-04": 80},
    )

    staged = config.read_text()
    assert "[source0]" in staged and "latency=20" in staged
    assert "[source1]" in staged and "latency=80" in staged
    assert staged.count("drop-on-latency=1") == 2
    assert "live-source=1" in staged
    assert "rtsp://cam01.example/live" in staged
    assert "rtsp://cam04.example/live" in staged


def test_cam04_decoder_surface_test_override_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    monkeypatch.setenv("MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES", "4")

    command = deepstream_command(
        tmp_path, tmp_path / "binary", "ds-image", "live", "latency-test"
    )

    assert "MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES=4" in command


def test_native_deepstream_latency_probe_is_opt_in(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    monkeypatch.setenv("MV3DT_CAPTURE_NATIVE_LATENCY", "1")

    command = deepstream_command(
        tmp_path, tmp_path / "binary", "ds-image", "live", "latency-test"
    )

    assert "MV3DT_NATIVE_LATENCY_LOG=/workspace/experiments/logs/probe/native_decoder_latency.jsonl" in command


def test_cam04_decoder_surface_test_override_rejects_invalid_value(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    monkeypatch.setenv("MV3DT_TEST_CAM04_DECODER_EXTRA_SURFACES", "56")

    try:
        deepstream_command(
            tmp_path, tmp_path / "binary", "ds-image", "live", "latency-test"
        )
    except ValueError as exc:
        assert "must be 0..55" in str(exc)
    else:
        raise AssertionError("out-of-range decoder surface override was accepted")


def test_cam04_rtsp_latency_override_is_experiment_only(monkeypatch) -> None:
    original = {"CAM-01": 20, "CAM-04": 80}
    monkeypatch.setenv("MV3DT_TEST_CAM04_RTSP_LATENCY_MS", "50")

    assert apply_test_camera_latency_overrides(original) == {"CAM-01": 20, "CAM-04": 50}
    assert original == {"CAM-01": 20, "CAM-04": 80}


def test_cam04_rtsp_latency_override_rejects_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("MV3DT_TEST_CAM04_RTSP_LATENCY_MS", "0")

    try:
        apply_test_camera_latency_overrides({"CAM-01": 20, "CAM-04": 80})
    except ValueError as exc:
        assert "must be 1..150" in str(exc)
    else:
        raise AssertionError("out-of-range CAM-04 source latency override was accepted")


def test_streammux_timeout_override_only_changes_staged_config(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config_deepstream.txt"
    original = "[source0]\nenable=1\n[source1]\nenable=1\n[streammux]\nbatch-size=2\nbatched-push-timeout=-1\n"
    config.write_text(original)
    monkeypatch.setenv("MV3DT_TEST_STREAMMUX_BATCHED_PUSH_TIMEOUT", "25000")

    assert apply_test_streammux_timeout(tmp_path) == 25000
    assert "batched-push-timeout=25000" in config.read_text()
    assert "batched-push-timeout=-1" not in config.read_text()


def test_streammux_timeout_override_rejects_out_of_range(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config_deepstream.txt").write_text("[streammux]\nbatched-push-timeout=-1\n")
    monkeypatch.setenv("MV3DT_TEST_STREAMMUX_BATCHED_PUSH_TIMEOUT", "50001")
    try:
        apply_test_streammux_timeout(tmp_path)
    except ValueError as exc:
        assert "1..50000" in str(exc)
    else:
        raise AssertionError("out-of-range streammux timeout was accepted")


def test_streammux_timeout_metadata_supports_production_no_experiment_run(tmp_path: Path) -> None:
    record_test_streammux_timeout_override(tmp_path, 25000, None, False)
    metadata = json.loads((tmp_path / "experiment_overrides.json").read_text())
    assert metadata["experiment"] is None
    assert metadata["production_profile_modified"] is False
    assert metadata["streammux_batched_push_timeout_us_test"] == 25000


def test_cam04_analytics_queue_override_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    monkeypatch.setenv("MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS", "24")

    command = deepstream_command(
        tmp_path, tmp_path / "binary", "ds-image", "live", "latency-test"
    )

    assert "MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS=24" in command


def test_cam04_analytics_queue_override_rejects_invalid_value(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "config_deepstream.txt").write_text(
        "[source0]\nenable=1\n[source1]\nenable=1\n"
    )
    (tmp_path / "config_msgconv.txt").write_text(
        "[sensor0]\nid=CAM-01\n[sensor1]\nid=CAM-04\n"
    )
    monkeypatch.setenv("MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS", "65")

    try:
        deepstream_command(
            tmp_path, tmp_path / "binary", "ds-image", "live", "latency-test"
        )
    except ValueError as exc:
        assert "must be 4..64" in str(exc)
    else:
        raise AssertionError("out-of-range CAM-04 analytics queue override was accepted")
