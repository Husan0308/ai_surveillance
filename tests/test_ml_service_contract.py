from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_camera_config_has_no_codec_knob() -> None:
    config = source("services/ml_service/app/config.py")
    worker = source("services/ml_service/app/camera_worker.py")
    probe = source("scripts/probe_cameras.py")

    assert "codec:" not in config
    assert ".codec" not in worker
    assert ".codec" not in probe
    assert "_camera_uri" in config
    assert 'f"{camera_id.replace(\'-\', \'_\')}_URI"' in config


def test_capture_is_owned_by_nvurisrcbin() -> None:
    capture = source("services/ml_service/app/deepstream/capture.py")

    for required in (
        'backend = "deepstream-nvurisrcbin"',
        '"nvurisrcbin"',
        '"rtsp-reconnect-attempts=-1"',
        'f"rtsp-reconnect-interval={reconnect_interval}"',
        'f"select-rtp-protocol={rtp_protocol}"',
        'f"cudadec-memtype={int(c.cudadec_memtype)}"',
        '"disable-audio=true"',
        '"drop=true"',
        '"max-buffers=1"',
    ):
        assert required in capture

    assert "rtph264depay" not in capture
    assert "rtph265depay" not in capture
    assert "nvv4l2decoder" not in capture


def test_deepstream_settings_validate_decoder_memory() -> None:
    config = source("services/ml_service/app/config.py")
    cameras = source("config/cameras.yaml")

    assert "cudadec_memtype: int" in config
    assert "cudadec_memtype not in {0, 1, 2}" in config
    assert "cudadec_memtype: 0" in cameras


def test_ml_preflight_exists() -> None:
    preflight = source("scripts/preflight_ml_service.py")
    for plugin in ("nvurisrcbin", "nvvideoconvert", "appsink"):
        assert plugin in preflight
    assert 'print("ML_PREFLIGHT=PASS"' in preflight
