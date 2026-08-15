from __future__ import annotations

from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ml_service.app.config import load_settings


def _rtsp_request(uri: str, method: str, cseq: int = 1) -> tuple[str, dict[str, str]]:
    parsed = urlsplit(uri)
    host = parsed.hostname
    if not host:
        raise RuntimeError("RTSP URI has no host")
    port = parsed.port or 554
    request = (
        f"{method} {uri} RTSP/1.0\r\n"
        f"CSeq: {cseq}\r\n"
        "User-Agent: ai-surveillance-rtsp-probe/1.0\r\n"
        "Accept: application/sdp\r\n"
        "\r\n"
    ).encode("ascii", errors="strict")

    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.settimeout(3.0)
        sock.sendall(request)
        data = bytearray()
        while b"\r\n\r\n" not in data and len(data) < 65536:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data.extend(chunk)

    if not data:
        raise RuntimeError("server closed connection without an RTSP response")

    head = bytes(data).split(b"\r\n\r\n", 1)[0].decode("latin-1", errors="replace")
    lines = head.split("\r\n")
    status = lines[0] if lines else ""
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return status, headers


def main() -> int:
    settings = load_settings()
    failures = 0
    print("=== RTSP server control-plane probe ===")
    print("This does not start NVDEC; it only asks the RTSP server for OPTIONS/DESCRIBE.\n")

    for camera in settings.cameras:
        print(f"[RTSP] {camera.camera_id} {camera.uri} auth_configured={'yes' if camera.username else 'no'}")
        try:
            options_status, _ = _rtsp_request(camera.uri, "OPTIONS", 1)
            describe_status, describe_headers = _rtsp_request(camera.uri, "DESCRIBE", 2)
            auth_header = describe_headers.get("www-authenticate", "")
            auth_scheme = auth_header.split(None, 1)[0] if auth_header else ""
            print(f"       OPTIONS  {options_status}")
            print(f"       DESCRIBE {describe_status}")
            if describe_status.startswith("RTSP/1.0 401"):
                print(f"       AUTH REQUIRED scheme={auth_scheme or 'unknown'}")
                if not camera.username:
                    print("       FIX: configure SURVEILLANCE_RTSP_USERNAME/PASSWORD in .env")
            elif describe_status.startswith("RTSP/1.0 404"):
                print("       FIX: RTSP channel/path is not present on this recorder")
            elif describe_status.startswith("RTSP/1.0 200"):
                print("       RTSP path is reachable without an auth challenge")
            else:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"       ERROR: {type(exc).__name__}: {exc}")
        print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
