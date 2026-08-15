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

        backend = str(camera.get("capture_backend") or "")
        online = bool(camera.get("online"))
        fps = f(camera.get("source_fps"))
        age_ms = f(camera.get("last_frame_age_ms"), 999999)
        lag_ms = camera.get("pipeline_lag_ms")
        lag_text = "n/a" if lag_ms is None else f"{f(lag_ms):.0f}ms"
        reconnects = int(camera.get("reconnects") or 0)
        publish_rate = f(publisher.get("publish_rate"))
        transport_ms = f(publisher.get("last_transport_ms"))
        width = int(camera.get("width") or 0)
        height = int(camera.get("height") or 0)
        rtsp_transport = str(runtime.get("transport") or "-")
        rtsp_latency = int(runtime.get("latency_ms") or 0)
        extra_surfaces = int(runtime.get("decoder_extra_surfaces") or 0)

        warnings = []
        if not online:
            warnings.append("OFFLINE")
        if backend != "deepstream-nvurisrcbin":
            warnings.append(f"backend={backend or 'none'}")
        if width != TARGET_WIDTH or height != TARGET_HEIGHT:
            warnings.append(f"RES={width}x{height}")
        if rtsp_transport != "tcp":
            warnings.append(f"RTP={rtsp_transport}")
        if fps < 14.0:
            warnings.append("LOW_SOURCE_FPS")
        if age_ms > 250.0:
            warnings.append("STALE_FRAME")
        if lag_ms is not None and f(lag_ms) > 500.0:
            warnings.append("HIGH_PIPELINE_LAG")
        if publish_rate < 14.0:
            warnings.append("LOW_PUBLISH_RATE")
        if transport_ms > 20.0:
            warnings.append("SLOW_MMAP_WRITE")

        if warnings:
            failed = True

        print(
            f"{camera_id} {width}x{height} "
            f"online={online} backend={backend or '-'} "
            f"rtsp={rtsp_transport}/{rtsp_latency}ms surf={extra_surfaces} "
            f"src={fps:.1f}fps age={age_ms:.0f}ms lag={lag_text} "
            f"reconnects={reconnects} publish={publish_rate:.1f}fps "
            f"mmap={transport_ms:.1f}ms "
            f"{'OK' if not warnings else 'WARN:' + ','.join(warnings)}"
        )

    resources = payload.get("service_resources") or {}
    print()
    print(f"service_resources={resources}")

    if failed:
        print("\nBASELINE: NOT CLEAN — keep AI disabled and fix camera path first.")
        return 1

    print("\nBASELINE: CLEAN — 720p DeepStream camera path is ready for detection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
