#!/usr/bin/env python3
from __future__ import annotations

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


def _find_text(root: ET.Element, name: str) -> str:
    for node in root.iter():
        if _local(node.tag) == name:
            return (node.text or "").strip()
    return ""


def _find_node(root: ET.Element, name: str) -> ET.Element | None:
    for node in root.iter():
        if _local(node.tag) == name:
            return node
    return None


def _substream_id(uri: str) -> str:
    path = urllib.parse.urlparse(uri).path.rstrip("/")
    token = path.rsplit("/", 1)[-1]
    if not token.isdigit() or len(token) < 3:
        raise ValueError(f"cannot derive Hikvision channel from URI path: {path}")
    return f"{token[:-1]}2"


def _fps(raw: str) -> str:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return "-"
    if value >= 100.0:
        value /= 100.0
    return f"{value:.2f}"


class _SingleAttemptDigestAuthHandler(urllib.request.HTTPDigestAuthHandler):
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


def _digest_opener(base_url: str, username: str, password: str) -> urllib.request.OpenerDirector:
    mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, base_url, username, password)
    return urllib.request.build_opener(_SingleAttemptDigestAuthHandler(mgr))


def _get_xml(opener: urllib.request.OpenerDirector, url: str, timeout: float) -> ET.Element:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/xml", "User-Agent": "ai-surveillance-stream-probe/1.1"},
    )
    with opener.open(req, timeout=timeout) as resp:
        payload = resp.read()
    return ET.fromstring(payload)


def _cap_text(root: ET.Element | None, name: str) -> str:
    if root is None:
        return "-"
    node = _find_node(root, name)
    if node is None:
        return "-"
    attrs = []
    for key in ("opt", "min", "max", "def"):
        if key in node.attrib:
            attrs.append(f"{key}={node.attrib[key]}")
    value = (node.text or "").strip()
    if value:
        attrs.insert(0, f"value={value}")
    return ",".join(attrs) if attrs else "present"


def main() -> int:
    settings = load_settings()
    first = settings.cameras[0]
    parsed = urllib.parse.urlparse(first.uri)
    host = os.getenv("HIKVISION_ISAPI_HOST", parsed.hostname or "").strip()
    if not host:
        raise SystemExit("HIKVISION_STREAM_PROBE_FAIL reason=no-host")
    scheme = os.getenv("HIKVISION_ISAPI_SCHEME", "http").strip().lower()
    port = int(os.getenv("HIKVISION_ISAPI_PORT", "80" if scheme == "http" else "443"))
    timeout = max(1.0, float(os.getenv("HIKVISION_ISAPI_TIMEOUT_SEC", "6")))
    base_url = f"{scheme}://{host}:{port}"

    username = first.username
    password = first.password
    if not username:
        raise SystemExit("HIKVISION_STREAM_PROBE_FAIL reason=missing-username")

    opener = _digest_opener(base_url, username, password)
    print(
        f"HIKVISION_STREAM_PROBE target={host}:{port} scheme={scheme} auth=digest-single-attempt "
        f"cameras={len(settings.cameras)} password_logged=0",
        flush=True,
    )

    failures = 0
    rows: list[dict[str, str]] = []
    for camera in settings.cameras:
        channel = _substream_id(camera.uri)
        config_url = f"{base_url}/ISAPI/Streaming/channels/{channel}"
        caps_url = f"{config_url}/capabilities"
        try:
            cfg = _get_xml(opener, config_url, timeout)
        except urllib.error.HTTPError as exc:
            failures += 1
            if exc.code == 401:
                print(
                    f"HIKVISION_STREAM_PROBE_AUTH_ERROR camera={camera.camera_id} id={channel} "
                    "http=401 attempts=1 action=abort reason=credential-rejected-or-illegal-login-lock "
                    "note=do-not-repeat-until-lock-duration-has-expired",
                    flush=True,
                )
                return 4
            print(
                f"HIKVISION_STREAM {camera.camera_id} id={channel} status=HTTP_{exc.code}",
                flush=True,
            )
            continue
        except Exception as exc:
            failures += 1
            print(
                f"HIKVISION_STREAM {camera.camera_id} id={channel} status=ERROR "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            continue

        try:
            caps = _get_xml(opener, caps_url, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                print(
                    f"HIKVISION_STREAM_PROBE_AUTH_ERROR camera={camera.camera_id} id={channel} "
                    "stage=capabilities http=401 attempts=1 action=abort",
                    flush=True,
                )
                return 4
            caps = None
        except Exception:
            caps = None

        svc_node = None
        for node in cfg.iter():
            if _local(node.tag) == "SVC":
                svc_node = node
                break
        svc_enabled = _find_text(svc_node, "enabled") if svc_node is not None else ""

        row = {
            "camera": camera.camera_id,
            "id": channel,
            "codec": _find_text(cfg, "videoCodecType") or "-",
            "width": _find_text(cfg, "videoResolutionWidth") or "-",
            "height": _find_text(cfg, "videoResolutionHeight") or "-",
            "max_fps_raw": _find_text(cfg, "maxFrameRate") or "-",
            "max_fps": _fps(_find_text(cfg, "maxFrameRate")),
            "gov": _find_text(cfg, "GovLength") or "-",
            "key_ms": _find_text(cfg, "keyFrameInterval") or "-",
            "rate_control": _find_text(cfg, "videoQualityControlType") or "-",
            "cbr_kbps": _find_text(cfg, "constantBitRate") or "-",
            "vbr_max_kbps": _find_text(cfg, "vbrUpperCap") or "-",
            "vbr_min_kbps": _find_text(cfg, "vbrLowerCap") or "-",
            "svc": svc_enabled or "-",
            "smoothing": _find_text(cfg, "smoothing") or "-",
            "fps_cap": _cap_text(caps, "maxFrameRate"),
            "gov_cap": _cap_text(caps, "GovLength"),
        }
        rows.append(row)
        print(
            "HIKVISION_STREAM "
            f"{row['camera']} id={row['id']} status=OK codec={row['codec']} "
            f"size={row['width']}x{row['height']} max_fps={row['max_fps']} raw={row['max_fps_raw']} "
            f"gov={row['gov']} key_ms={row['key_ms']} rate={row['rate_control']} "
            f"cbr_kbps={row['cbr_kbps']} vbr={row['vbr_min_kbps']}..{row['vbr_max_kbps']} "
            f"svc={row['svc']} smoothing={row['smoothing']} fps_cap=<{row['fps_cap']}> "
            f"gov_cap=<{row['gov_cap']}>",
            flush=True,
        )

    cam02 = next((row for row in rows if row["camera"] == "CAM-02"), None)
    peers = [row for row in rows if row["camera"] != "CAM-02"]
    if cam02 is not None and peers:
        for field in ("codec", "width", "height", "max_fps", "gov", "key_ms", "rate_control", "svc", "smoothing"):
            peer_values = sorted({row[field] for row in peers})
            if cam02[field] not in peer_values or len(peer_values) > 1:
                print(
                    f"HIKVISION_STREAM_DIFF field={field} CAM-02={cam02[field]} "
                    f"peers={','.join(peer_values)}",
                    flush=True,
                )

    if failures:
        print(f"HIKVISION_STREAM_PROBE_RESULT status=PARTIAL failures={failures}", flush=True)
        return 2
    print("HIKVISION_STREAM_PROBE_RESULT status=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
