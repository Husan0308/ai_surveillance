from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


@dataclass(frozen=True)
class QwenVerdict:
    verdict: str
    confidence: float
    cues: dict
    latency_ms: float
    error: str = ""


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_json_object(text: str) -> dict:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    try:
        data = json.loads(value)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(value[start : end + 1])
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def _decode_jpeg(blob: bytes) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def build_comparison_sheet(old_jpegs: list[bytes], new_jpegs: list[bytes]) -> bytes:
    """Make one compact OLD-vs-NEW image so VLM serving only receives one image."""
    old = [img for img in (_decode_jpeg(x) for x in old_jpegs[:3]) if img is not None]
    new = [img for img in (_decode_jpeg(x) for x in new_jpegs[:3]) if img is not None]
    if not old or not new:
        raise ValueError("Qwen comparison requires at least one OLD and one NEW crop")

    cell_w, cell_h = 160, 320
    header = 36
    cols = 3
    canvas = np.zeros((header * 2 + cell_h * 2, cell_w * cols, 3), dtype=np.uint8)
    canvas[:] = (14, 20, 27)

    cv2.putText(canvas, "OLD GLOBAL ID", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 238, 245), 2, cv2.LINE_AA)
    second_y = header + cell_h
    cv2.putText(canvas, "NEW TRACK", (10, second_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 238, 245), 2, cv2.LINE_AA)

    def place(images: list[np.ndarray], row_y: int) -> None:
        for index in range(cols):
            x = index * cell_w
            if index >= len(images):
                cv2.rectangle(canvas, (x + 3, row_y + 3), (x + cell_w - 4, row_y + cell_h - 4), (42, 53, 64), 1)
                continue
            image = images[index]
            h, w = image.shape[:2]
            scale = min((cell_w - 8) / max(1, w), (cell_h - 8) / max(1, h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            ox = x + (cell_w - nw) // 2
            oy = row_y + (cell_h - nh) // 2
            canvas[oy : oy + nh, ox : ox + nw] = resized
            cv2.putText(canvas, str(index + 1), (x + 8, row_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 200), 2, cv2.LINE_AA)

    place(old, header)
    place(new, second_y + header)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("could not encode Qwen comparison sheet")
    return encoded.tobytes()


class QwenReIdVerifier:
    """OpenAI-compatible Qwen-VL verifier; always called outside the hot path."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        raw_url = str(os.environ.get("QWEN_REID_URL", cfg.get("qwen_url", ""))).strip()
        if raw_url.endswith("/v1"):
            raw_url += "/chat/completions"
        elif raw_url and not raw_url.endswith("/chat/completions"):
            raw_url = raw_url.rstrip("/") + "/v1/chat/completions"
        self.url = raw_url
        self.model = str(os.environ.get("QWEN_REID_MODEL", cfg.get("qwen_model", "qwen3-vl"))).strip()
        self.api_key = str(os.environ.get("QWEN_REID_API_KEY", cfg.get("qwen_api_key", ""))).strip()
        self.timeout = max(0.5, float(os.environ.get("QWEN_REID_TIMEOUT_SEC", cfg.get("qwen_timeout_sec", 3.0))))
        self.enabled = bool(self.url) and str(os.environ.get("QWEN_REID_ENABLED", cfg.get("qwen_enabled", "1"))).lower() not in {"0", "false", "no", "off"}
        self.calls = 0
        self.same = 0
        self.different = 0
        self.uncertain = 0
        self.errors = 0
        self.last_latency_ms = 0.0
        self.last_error = ""

    @staticmethod
    def _prompt() -> str:
        return (
            "You are a conservative visual verifier for person re-identification in CCTV. "
            "The image is a 2-row contact sheet: OLD GLOBAL ID on top, NEW TRACK on bottom. "
            "Judge whether both rows show the SAME physical person despite camera angle, pose, "
            "partial occlusion, scale and lighting. Compare clothing colors/patterns, trousers, "
            "shoes, backpack/accessories, body build, hair/head appearance, and any visible face. "
            "Do not use background/location as identity evidence. If evidence is weak, conflicting, "
            "or crops are too poor, choose UNCERTAIN. False SAME is worse than UNCERTAIN. "
            "Return ONLY one JSON object with schema: "
            '{"verdict":"SAME|DIFFERENT|UNCERTAIN","confidence":0.0,"cues":{"upper":"match|different|uncertain","lower":"match|different|uncertain","shoes":"match|different|uncertain","accessories":"match|different|uncertain","body":"match|different|uncertain","face":"match|different|uncertain"}}'
        )

    def verify(self, payload: dict) -> QwenVerdict:
        if not self.enabled:
            return QwenVerdict("UNCERTAIN", 0.0, {}, 0.0, "qwen-disabled")
        started = time.perf_counter()
        try:
            sheet = build_comparison_sheet(payload.get("old_jpegs") or [], payload.get("new_jpegs") or [])
            data_uri = "data:image/jpeg;base64," + base64.b64encode(sheet).decode("ascii")
            body = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": self._prompt()},
                    ],
                }],
                "temperature": 0.0,
                "max_tokens": 180,
            }
            raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(self.url, data=raw, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8", errors="replace"))
            content = response_data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
            parsed = _parse_json_object(str(content))
            verdict = str(parsed.get("verdict", "UNCERTAIN")).upper().strip()
            if verdict not in {"SAME", "DIFFERENT", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            try:
                confidence = _clamp(float(parsed.get("confidence", 0.0)))
            except Exception:
                confidence = 0.0
            cues = parsed.get("cues") if isinstance(parsed.get("cues"), dict) else {}
            error = "" if parsed else "invalid-json"
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError, OSError) as exc:
            verdict, confidence, cues = "UNCERTAIN", 0.0, {}
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            verdict, confidence, cues = "UNCERTAIN", 0.0, {}
            error = f"{type(exc).__name__}: {exc}"

        latency = (time.perf_counter() - started) * 1000.0
        self.calls += 1
        self.last_latency_ms = latency
        self.last_error = error
        if error:
            self.errors += 1
        if verdict == "SAME": self.same += 1
        elif verdict == "DIFFERENT": self.different += 1
        else: self.uncertain += 1
        return QwenVerdict(verdict, confidence, cues, latency, error)

    def metrics(self) -> dict:
        return {
            "enabled": self.enabled,
            "url_configured": bool(self.url),
            "model": self.model,
            "calls": self.calls,
            "same": self.same,
            "different": self.different,
            "uncertain": self.uncertain,
            "errors": self.errors,
            "last_latency_ms": self.last_latency_ms,
            "last_error": self.last_error,
            "timeout_sec": self.timeout,
        }
