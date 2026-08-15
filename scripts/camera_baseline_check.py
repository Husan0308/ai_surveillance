from __future__ import annotations

import json
import urllib.error
import urllib.request


URL = "http://127.0.0.1:8001/health"
CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def fetch_health() -> dict:
    with urllib.request.urlopen(URL, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def stat(stage: dict, key: str, percentile: str = "p95") -> float:
    value = stage.get(key) or {}
    if isinstance(value, dict):
        return f(value.get(percentile))
    return 0.0


def main() -> int:
    try:
        payload = fetch_health()
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read {URL}: {exc}")
        return 2

    print(f"mode={payload.get('mode')} profile={payload.get('profile')}")
    print(f"online={payload.get('online')}/{payload.get('total')}")
    print(f"display={payload.get('display')}")
    print()

    cameras = payload.get("cameras") or {}
    publishers = payload.get("publishers") or {}
    failed = False

    for camera_id in CAMERA_IDS:
        camera = cameras.get(camera_id) or {}
        publisher = publishers.get(camera_id) or {}
        runtime = camera.get("source_runtime") or {}
        stage = camera.get("capture_stage") or {}

        backend = str(camera.get("capture_backend") or "")
        online = bool(camera.get("online"))
        fps = f(camera.get("source_fps"))
        age_ms = f(camera.get("last_frame_age_ms"), 999999)
        lag_ms = camera.get("pipeline_lag_ms")
        lag_text = "n/a" if lag_ms is None else f"{f(lag_ms):.0f}ms"
        reconnects = int(camera.get("reconnects") or 0)
        read_failures = int(camera.get("read_failures") or 0)
        publish_rate = f(publisher.get("publish_rate"))
        transport_ms = f(publisher.get("last_transport_ms"))
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        rtsp_transport = str(runtime.get("transport") or "-")
        rtsp_latency = int(runtime.get("latency_ms") or 0)
        extra_surfaces = int(runtime.get("decoder_extra_surfaces") or 0)
        map_copy_p95 = stat(stage, "map_copy_ms")
        frame_interval_p95 = stat(stage, "frame_interval_ms")
        stage_lag_p95 = stat(stage, "pipeline_lag_ms")
        queue_buffers = stage.get("current_postdecode_queue_buffers")
        last_warning = str(stage.get("last_bus_warning") or "")
        last_error = str(stage.get("last_bus_error") or "")

        warnings = []
        if not online:
            warnings.append("OFFLINE")
        if backend != "deepstream-nvurisrcbin":
            warnings.append(f"backend={backend or 'none'}")
        if width != TARGET_WIDTH or height != TARGET_HEIGHT:
            warnings.append(f"RES={width}x{height}")
        if fps < 14.0:
            warnings.append("LOW_SOURCE_FPS")
        if age_ms > 300.0:
            warnings.append("STALE_FRAME")
        if reconnects > 0:
            warnings.append("RECONNECTS")
        if read_failures > 5:
            warnings.append("READ_FAILURES")
        if publish_rate < 14.0:
            warnings.append("LOW_PUBLISH_RATE")
        if transport_ms > 12.0:
            warnings.append("SLOW_MMAP_WRITE")
        if map_copy_p95 > 8.0:
            warnings.append("SLOW_GPU_CPU_COPY")
        if frame_interval_p95 > 100.0:
            warnings.append("JITTERY_DECODE")
        if queue_buffers is not None and int(queue_buffers) > 1:
            warnings.append("QUEUE_BACKLOG")
        if last_error:
            warnings.append("GST_ERROR")

        if warnings:
            failed = True

        print(
            f"{camera_id} {width}x{height} online={online} "
            f"backend={backend or '-'} rtp={rtsp_transport}/{rtsp_latency}ms "
            f"surf={extra_surfaces} src={fps:.1f}fps age={age_ms:.0f}ms "
            f"lag={lag_text} stageLag95={stage_lag_p95:.1f}ms "
            f"frame95={frame_interval_p95:.1f}ms map95={map_copy_p95:.1f}ms "
            f"mmap={transport_ms:.1f}ms publish={publish_rate:.1f}fps "
            f"q={queue_buffers} reconnects={reconnects} readfail={read_failures} "
            f"{'OK' if not warnings else 'WARN:' + ','.join(warnings)}"
        )
        if last_warning:
            print(f"  gst-warning: {last_warning}")
        if last_error:
            print(f"  gst-error: {last_error}")

    resources = payload.get("service_resources") or {}
    print()
    print(f"service_resources={resources}")
    print(
        "NOTE: this command measures the legacy CPU-BGR+mmap diagnostic path. "
        "For the true display baseline run: "
        "python -m services.frontend.core_v1.deepstream_gpu_wall"
    )

    if failed:
        print("\nMMAP DIAGNOSTIC: bottleneck/warnings detected; keep AI disabled.")
        return 1

    print("\nMMAP DIAGNOSTIC: healthy, but GPU wall is still the preferred display path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
