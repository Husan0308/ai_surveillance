#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.shared.camera_config import load_settings


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_node(root: ET.Element, name: str) -> ET.Element | None:
    for node in root.iter():
        if _local(node.tag) == name:
            return node
    return None


def _find_text(root: ET.Element, name: str) -> str:
    node = _find_node(root, name)
    return (node.text or "").strip() if node is not None else ""


def _substream_id(uri: str) -> str:
    token = urllib.parse.urlparse(uri).path.rstrip("/").rsplit("/", 1)[-1]
    if not token.isdigit() or len(token) < 3:
        raise ValueError(f"cannot derive Hikvision channel from URI: {uri}")
    return f"{token[:-1]}2"


class _SingleAttemptDigestAuthHandler(urllib.request.HTTPDigestAuthHandler):
    """Allow one authenticated Digest attempt for each HTTP request.

    Hikvision Illegal Login Lock can block an IP after only a handful of bad
    attempts. urllib keeps the retry counter on the handler instance, so reset it
    after a successful response while still stopping a rejected credential after
    one authenticated retry.
    """

    def http_error_401(self, req, fp, code, msg, hdrs):  # type: ignore[override]
        if getattr(self, "retried", 0) >= 1:
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "digest auth failed after one credential attempt",
                hdrs,
                fp,
            )
        return super().http_error_401(req, fp, code, msg, hdrs)

    def http_response(self, req, response):  # type: ignore[override]
        self.retried = 0
        return response

    https_response = http_response


def _digest_opener(base_url: str, username: str, password: str) -> urllib.request.OpenerDirector:
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, base_url, username, password)
    return urllib.request.build_opener(_SingleAttemptDigestAuthHandler(mgr))


def _request(
    opener: urllib.request.OpenerDirector,
    url: str,
    method: str,
    timeout: float,
    payload: bytes | None = None,
) -> bytes:
    headers = {
        "Accept": "application/xml",
        "User-Agent": "ai-surveillance-hikvision-stream-config/1.3",
    }
    if payload is not None:
        headers["Content-Type"] = "application/xml; charset=UTF-8"
    req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    with opener.open(req, timeout=timeout) as resp:
        return resp.read()


def _register_default_namespace(root: ET.Element) -> None:
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0][1:]
        if namespace:
            ET.register_namespace("", namespace)


def _fps(raw: str) -> float:
    value = float(raw)
    return value / 100.0 if value >= 100.0 else value


def _response_status(payload: bytes) -> tuple[str, str, str, str, str]:
    if not payload:
        return "-", "-", "-", "-", "-"
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return "-", "non-xml", "-", "-", "-"
    return (
        _find_text(root, "statusCode") or "-",
        _find_text(root, "statusString") or "-",
        _find_text(root, "subStatusCode") or "-",
        _find_text(root, "errorCode") or "-",
        _find_text(root, "errorMsg") or "-",
    )


def _write_http_error(channel: str, stamp: str, exc: urllib.error.HTTPError) -> int:
    try:
        body = exc.read()
    except Exception:
        body = b""

    status_code, status_string, sub_status, error_code, error_msg = _response_status(body)
    error_path = Path(f"/tmp/hikvision_{channel}_put_error_{stamp}.xml")
    if body:
        error_path.write_bytes(body)

    print(
        "HIKVISION_STREAM_SET_HTTP_ERROR "
        f"http={exc.code} reason={exc.reason} status_code={status_code} "
        f"status={status_string} sub_status={sub_status} error_code={error_code} "
        f"error_msg={error_msg} body_bytes={len(body)} "
        f"body_path={error_path if body else '-'}",
        flush=True,
    )

    normalized = " ".join(
        value.lower() for value in (status_string, sub_status, error_msg) if value and value != "-"
    )
    if exc.code == 403:
        if "lowprivilege" in normalized or "no permission" in normalized:
            diagnosis = "authenticated-but-insufficient-remote-configuration-privilege"
            next_step = "use-admin-or-enable-remote-parameters-settings-and-remote-camera-management"
        elif "notsupport" in normalized or "not support" in normalized:
            diagnosis = "endpoint-write-not-supported-or-proxied-channel"
            next_step = "configure-the-underlying-ip-camera-directly-or-via-nvr-virtual-host"
        elif "permission" in normalized or "forbidden" in normalized or "authorization" in normalized:
            diagnosis = "authenticated-but-write-permission-denied"
            next_step = "use-admin-or-enable-remote-parameters-settings"
        elif "invalid operation" in normalized:
            diagnosis = "invalid-operation-without-specific-substatus"
            next_step = "check-user-remote-configuration-permissions-then-device-endpoint-support"
        else:
            diagnosis = "authenticated-get-ok-but-put-forbidden"
            next_step = "check-admin-remote-parameters-permission-and-response-body"
        print(
            "HIKVISION_STREAM_SET_DIAG "
            f"diagnosis={diagnosis} next={next_step} "
            "note=GET-succeeded-so-credentials-are-valid-for-read",
            flush=True,
        )
    return 3


def _write_get_auth_error(exc: urllib.error.HTTPError, stage: str) -> int:
    if exc.code != 401:
        raise exc
    print(
        "HIKVISION_STREAM_SET_AUTH_ERROR http=401 "
        f"stage={stage} attempts=1 action=abort "
        "reason=credential-rejected-or-illegal-login-lock",
        flush=True,
    )
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely inspect or update one Hikvision substream. Dry-run is the default."
    )
    parser.add_argument("--camera", default="CAM-02", help="Camera id from config/cameras.yaml")
    parser.add_argument("--fps", type=float, default=20.0, help="Requested frame rate in fps")
    parser.add_argument("--gov", type=int, default=None, help="Optional GovLength; omit to leave unchanged")
    parser.add_argument("--apply", action="store_true", help="Actually PUT the modified XML to the device")
    args = parser.parse_args()

    settings = load_settings()
    camera = next((row for row in settings.cameras if row.camera_id == args.camera), None)
    if camera is None:
        raise SystemExit(f"HIKVISION_STREAM_SET_FAIL camera={args.camera} reason=not-configured")

    parsed = urllib.parse.urlparse(camera.uri)
    host = os.getenv("HIKVISION_ISAPI_HOST", parsed.hostname or "").strip()
    scheme = os.getenv("HIKVISION_ISAPI_SCHEME", "http").strip().lower()
    port = int(os.getenv("HIKVISION_ISAPI_PORT", "80" if scheme == "http" else "443"))
    timeout = max(1.0, float(os.getenv("HIKVISION_ISAPI_TIMEOUT_SEC", "6")))
    base_url = f"{scheme}://{host}:{port}"
    channel = _substream_id(camera.uri)
    url = f"{base_url}/ISAPI/Streaming/channels/{channel}"

    if not camera.username:
        raise SystemExit("HIKVISION_STREAM_SET_FAIL reason=missing-username")

    opener = _digest_opener(base_url, camera.username, camera.password)
    try:
        before_bytes = _request(opener, url, "GET", timeout)
    except urllib.error.HTTPError as exc:
        return _write_get_auth_error(exc, "initial-get")

    root = ET.fromstring(before_bytes)
    _register_default_namespace(root)

    fps_node = _find_node(root, "maxFrameRate")
    if fps_node is None:
        raise SystemExit("HIKVISION_STREAM_SET_FAIL reason=maxFrameRate-not-found")

    before_fps_raw = (fps_node.text or "").strip()
    before_fps = _fps(before_fps_raw)
    requested_fps_raw = str(int(round(args.fps * 100.0)))

    before_gov = _find_text(root, "GovLength") or "-"
    gov_node = _find_node(root, "GovLength")

    print(
        f"HIKVISION_STREAM_SET_PLAN camera={camera.camera_id} channel={channel} "
        f"codec={_find_text(root, 'videoCodecType') or '-'} "
        f"size={_find_text(root, 'videoResolutionWidth') or '-'}x{_find_text(root, 'videoResolutionHeight') or '-'} "
        f"fps={before_fps:.2f}->{args.fps:.2f} raw={before_fps_raw}->{requested_fps_raw} "
        f"gov={before_gov}->{args.gov if args.gov is not None else before_gov} "
        f"apply={int(args.apply)} password_logged=0 auth=digest-single-attempt-per-request",
        flush=True,
    )

    fps_node.text = requested_fps_raw
    if args.gov is not None:
        if gov_node is None:
            raise SystemExit("HIKVISION_STREAM_SET_FAIL reason=GovLength-not-found")
        gov_node.text = str(args.gov)

    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    if not args.apply:
        print("HIKVISION_STREAM_SET_DRY_RUN status=OK no_device_change=1", flush=True)
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path(f"/tmp/hikvision_{channel}_before_{stamp}.xml")
    backup.write_bytes(before_bytes)
    print(f"HIKVISION_STREAM_SET_BACKUP path={backup}", flush=True)

    try:
        response = _request(opener, url, "PUT", timeout, payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return _write_get_auth_error(exc, "put")
        return _write_http_error(channel, stamp, exc)

    if response:
        status_code, status_string, sub_status, error_code, error_msg = _response_status(response)
        print(
            "HIKVISION_STREAM_SET_PUT "
            f"status_code={status_code} status={status_string} sub_status={sub_status} "
            f"error_code={error_code} error_msg={error_msg}",
            flush=True,
        )

    try:
        verify_bytes = _request(opener, url, "GET", timeout)
    except urllib.error.HTTPError as exc:
        return _write_get_auth_error(exc, "verify-get")
    verify = ET.fromstring(verify_bytes)
    after_fps_raw = _find_text(verify, "maxFrameRate") or "-"
    after_gov = _find_text(verify, "GovLength") or "-"
    after_codec = _find_text(verify, "videoCodecType") or "-"

    fps_ok = after_fps_raw == requested_fps_raw
    gov_ok = args.gov is None or after_gov == str(args.gov)
    print(
        f"HIKVISION_STREAM_SET_VERIFY camera={camera.camera_id} channel={channel} codec={after_codec} "
        f"maxFrameRate={after_fps_raw} fps={_fps(after_fps_raw):.2f} gov={after_gov} "
        f"fps_ok={int(fps_ok)} gov_ok={int(gov_ok)}",
        flush=True,
    )
    if not (fps_ok and gov_ok):
        print("HIKVISION_STREAM_SET_RESULT status=FAIL backup_available=1", flush=True)
        return 2

    print("HIKVISION_STREAM_SET_RESULT status=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
