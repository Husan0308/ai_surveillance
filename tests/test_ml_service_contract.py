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


def test_capture_is_nvurisrcbin_latest_only_and_gpu_scaled_before_host_copy() -> None:
    capture = source("services/ml_service/app/deepstream/capture.py")

    for required in (
        'backend = "deepstream-nvurisrcbin"',
        '"nvurisrcbin"',
        '"rtsp-reconnect-attempts=-1"',
        'f"rtsp-reconnect-interval={reconnect_interval}"',
        'f"select-rtp-protocol={rtp_protocol}"',
        'f"cudadec-memtype={int(c.cudadec_memtype)}"',
        '"disable-audio=true"',
        '"leaky=downstream"',
        '"drop=true"',
        '"max-buffers=1"',
        '"nvvideoconvert"',
        'f"video/x-raw,width={int(c.display_width)},height={int(c.display_height)},"',
    ):
        assert required in capture

    assert "shmsink" not in capture
    assert "rtph264depay" not in capture
    assert "rtph265depay" not in capture
    assert "nvv4l2decoder" not in capture


def test_proven_camera_latency_and_clear_display_profile_are_configured() -> None:
    config = source("services/ml_service/app/config.py")
    cameras = source("config/cameras.yaml")

    assert "latency_ms: int | None" in config
    assert "camera.latency_ms or self.ds.latency_ms" in source("services/ml_service/app/camera_worker.py")
    assert cameras.count("latency_ms: 20") == 3
    assert cameras.count("latency_ms: 80") == 3
    assert "width: 960" in cameras
    assert "height: 540" in cameras
    assert "fps: 20" in cameras
    assert "jpeg_quality: 88" in cameras


def test_deepstream_settings_validate_decoder_memory() -> None:
    config = source("services/ml_service/app/config.py")
    cameras = source("config/cameras.yaml")

    assert "cudadec_memtype: int" in config
    assert "cudadec_memtype not in {0, 1, 2}" in config
    assert "cudadec_memtype: 0" in cameras


def test_mmap_presentation_is_sigbus_safe_and_jpeg_is_on_demand() -> None:
    pipeline = source("services/ml_service/app/deepstream/pipeline.py")
    mmap_publisher = source("services/ml_service/app/mmap_publisher.py")
    mmap_transport = source("shared/mmap_frame.py")
    safe_writer = source("shared/safe_mmap_frame.py")
    fallback = source("services/ml_service/app/fallback_jpeg.py")

    assert "MmapFramePublisher" in pipeline
    assert "OnDemandJpegPublisher" in pipeline
    assert "self.mmap_publishers" in pipeline
    assert "event-driven-mmap-latest-only" in mmap_publisher
    assert "SigbusSafeMmapFrameWriter" in mmap_publisher
    assert "MmapFrameReader" in mmap_transport
    assert "os.replace" in safe_writer
    assert "posix_fallocate" in safe_writer
    assert "on-demand-fallback" in fallback


def test_ml_preflight_exists() -> None:
    preflight = source("scripts/preflight_ml_service.py")
    for plugin in ("nvurisrcbin", "nvvideoconvert", "appsink"):
        assert plugin in preflight
    assert "shmsink" not in preflight
    assert "presentation=mmap" in preflight
    assert 'print("ML_PREFLIGHT=PASS"' in preflight


def test_person_detector_is_process_isolated_and_person_only() -> None:
    detector = source("services/ml_service/app/detector.py")
    main = source("services/ml_service/app/main.py")
    launcher = source("scripts/run_ml_service.sh")
    preflight = source("scripts/preflight_person_detection.py")

    assert 'mp.get_context("spawn")' in detector
    assert 'name="person-detector-cuda"' in detector
    assert '"classes": [0]' in detector
    assert '"isolation"] = "spawn-process"' in detector

    for forbidden in (
        "insightface",
        "fastreid",
        "torchreid",
        "BYTETracker",
        "GlobalIdentity",
    ):
        assert forbidden not in detector

    assert "from services.ml_service.app.deepstream.pipeline import DeepStreamRuntime" in main
    assert "runtime: Any | None = None" in main
    assert "PERSON_DETECT_PREFLIGHT_WARNING" in launcher
    assert "torch.cuda.get_arch_list()" in preflight
    assert "torch.cuda.get_device_capability" in preflight
    assert "PERSON_DETECT_CUDA_KERNEL=PASS" in preflight
