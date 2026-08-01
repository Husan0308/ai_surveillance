"""
GeminiEngine — Gemini 2.5 Flash integratsiyasi.
Uzoqdagi yuzlarni tahlil qilish, kiyim/holat tavsifi, yuz sifati baholash.
"""
import os
import time
import threading
import base64
import json
import numpy as np

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiEngine:
    def __init__(self, config=None):
        self.config = config or {}
        gem_cfg = self.config.get("gemini", {})
        self.enabled = gem_cfg.get("enabled", False)
        self.model_name = gem_cfg.get("model", "gemini-2.5-flash")
        self.api_key = gem_cfg.get("api_key") or os.environ.get("GEMINI_API_KEY")
        self.min_face_score = gem_cfg.get("min_face_score", 0.5)
        self.max_calls_per_min = gem_cfg.get("max_calls_per_min", 30)
        self.model = None
        self._lock = threading.Lock()
        self._call_times = []

        if self.enabled and GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"[GeminiEngine] ✅ {self.model_name} yuklandi", flush=True)
            except Exception as e:
                print(f"[GeminiEngine] ❌ Xato: {e}", flush=True)
                self.enabled = False
        else:
            if not GEMINI_AVAILABLE:
                print("[GeminiEngine] ⚠ google-generativeai o'rnatilmagan", flush=True)
            if not self.api_key:
                print("[GeminiEngine] ⚠ API key yo'q", flush=True)

    def _rate_limit(self):
        now = time.time()
        with self._lock:
            self._call_times = [t for t in self._call_times if now - t < 60]
            if len(self._call_times) >= self.max_calls_per_min:
                return False
            self._call_times.append(now)
            return True

    def _encode_image(self, image_bgr):
        import cv2
        _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_person(self, crop_bgr, face_score=0.0):
        """Odam crop ini tahlil qilish → {clothing, posture, direction, face_quality, confidence}"""
        if not self.enabled or self.model is None or crop_bgr is None or crop_bgr.size == 0:
            return None
        if not self._rate_limit():
            return None
        try:
            image_b64 = self._encode_image(crop_bgr)
            prompt = (
                f"Bu rasmda odam ko'rinmoqda. Yuz aniqlash darajasi: {face_score:.2f}.\n"
                "JSON formatda qaytaring:\n"
                '{"clothing": "kiyim tavsifi", "posture": "holat", '
                '"direction": "old/orqa/chap/o\'ng", "face_quality": "good/medium/poor", '
                '"confidence": 0.0}\n'
                "Faqat JSON, boshqa matn yo'q."
            )
            response = self.model.generate_content([
                {"mime_type": "image/jpeg", "data": image_b64},
                prompt
            ])
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            result = json.loads(text.strip())
            print(f"[GeminiEngine] ✅ {result.get('clothing','?')} | {result.get('direction','?')}", flush=True)
            return result
        except Exception as e:
            print(f"[GeminiEngine] ⚠ Tahlil xatosi: {e}", flush=True)
            return None

    def describe_snapshot(self, image_bgr, event_type=""):
        """Snapshot uchun AI tavsifi (event log ga izoh)"""
        if not self.enabled or self.model is None or not self._rate_limit():
            return ""
        try:
            image_b64 = self._encode_image(image_bgr)
            prompt = f"Bu kuzatuv kamerasi rasmi. Hodisa: {event_type}. 1-2 gap bilan tavsiflang."
            response = self.model.generate_content([
                {"mime_type": "image/jpeg", "data": image_b64},
                prompt
            ])
            return response.text.strip()
        except Exception as e:
            print(f"[GeminiEngine] ⚠ Tavsif xatosi: {e}", flush=True)
            return ""

    def assess_face_quality(self, face_crop_bgr):
        """Yuz sifati: tanish uchun yetarlimi?"""
        if not self.enabled or self.model is None or not self._rate_limit():
            return {"quality": "unknown", "sufficient": True}
        try:
            image_b64 = self._encode_image(face_crop_bgr)
            prompt = (
                "Bu yuz rasmi. Tanish uchun yetarlimi?\n"
                'JSON: {"quality": "good/medium/poor", "sufficient": true/false, "reason": "sabab"}\n'
                "Faqat JSON."
            )
            response = self.model.generate_content([
                {"mime_type": "image/jpeg", "data": image_b64},
                prompt
            ])
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception as e:
            print(f"[GeminiEngine] ⚠ Sifat xatosi: {e}", flush=True)
            return {"quality": "unknown", "sufficient": True}
