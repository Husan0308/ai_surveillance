"""
Body Re-Identification Engine
Yuz ko'rinmaganda, tana/kiyim orqali odamni tanish.
HSV color histogram + spatial weighting ishlatadi.
"""
import cv2
import numpy as np
from typing import Optional, Tuple

class BodyReIDEngine:
    """
    Tana crop dan HSV histogram yaratadi va solishtiradi.
    
    Afzalliklari:
    - CPU da tez ishlaydi (~1ms per crop)
    - Yoritish o'zgarishiga chidamli (HSV)
    - Orqadan/yon tomondan ham ishlaydi
    """
    
    def __init__(self, config=None):
        # ServiceManager uchun zarur atributlar
        self.enabled = True
        self.available = True
        
        # Histogram parametrlari (config dan yoki default)
        cfg = config or {}
        self.h_bins = int(cfg.get("ai.reid.h_bins", 30))
        self.s_bins = int(cfg.get("ai.reid.s_bins", 32))
        self.v_bins = int(cfg.get("ai.reid.v_bins", 16))
        
        # Histogram o'lchami: 30 × 32 × 16 = 15,360
        self.hist_size = [self.h_bins, self.s_bins, self.v_bins]
        self.h_ranges = [0, 180]
        self.s_ranges = [0, 256]
        self.v_ranges = [0, 256]
        self.ranges = self.h_ranges + self.s_ranges + self.v_ranges
        self.channels = [0, 1, 2]
        
        print(f"[BodyReID] ✅ Engine initialized: {self.h_bins}×{self.s_bins}×{self.v_bins}", flush=True)
    
    def extract_features(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop dan HSV histogram olish.
        
        Args:
            crop_bgr: BGR formatdagi odamning crop qilingan rasmi
        
        Returns:
            Normalized histogram (1D array) yoki None
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        
        h, w = crop_bgr.shape[:2]
        if h < 30 or w < 20:  # Juda kichik crop
            return None
        
        try:
            # BGR → HSV
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            
            # Spatial weighting: yuqori qism (bosh/ko'ylak) muhimroq
            # Crop ni 3 qismga bo'lish: bosh, tana, oyoq
            head_h = h // 4
            body_h = h // 2
            legs_h = h - head_h - body_h
            
            # Har qism uchun alohida histogram
            head_hist = cv2.calcHist([hsv[:head_h, :]], self.channels, None, 
                                     self.hist_size, self.ranges)
            body_hist = cv2.calcHist([hsv[head_h:head_h+body_h, :]], self.channels, None,
                                     self.hist_size, self.ranges)
            legs_hist = cv2.calcHist([hsv[head_h+body_h:, :]], self.channels, None,
                                     self.hist_size, self.ranges)
            
            # Vaznlar: bosh=0.4, tana=0.4, oyoq=0.2
            combined = head_hist * 0.4 + body_hist * 0.4 + legs_hist * 0.2
            
            # Normalize
            cv2.normalize(combined, combined, 0, 1, cv2.NORM_MINMAX)
            
            # 1D array ga aylantirish
            return combined.flatten()
            
        except Exception as e:
            print(f"[BodyReID] ⚠ extract error: {e}", flush=True)
            return None
    
    def compute_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Ikki histogram orasidagi o'xshashlikni hisoblash.
        
        Usullar:
        1. Correlation (0.0 to 1.0)
        2. Chi-Squared (pastroq = yaxshiroq)
        3. Bhattacharyya distance
        
        Returns: 0.0 (umuman o'xshash emas) to 1.0 (butunlay o'xshash)
        """
        if hist1 is None or hist2 is None:
            return 0.0
        
        if len(hist1) != len(hist2):
            return 0.0
        
        try:
            # Correlation — eng ishonchli
            corr = cv2.compareHist(hist1.reshape(-1, 1).astype(np.float32),
                                   hist2.reshape(-1, 1).astype(np.float32),
                                   cv2.HISTCMP_CORREL)
            
            # Correlation -1 dan 1 gacha, biz 0-1 ga o'tkazamiz
            similarity = (corr + 1.0) / 2.0
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"[BodyReID] ⚠ similarity error: {e}", flush=True)
            return 0.0


# Eski nomlar uchun alias (backward compatibility)
ReIDEngine = BodyReIDEngine


# ============================ HYBRID ENGINE ================================
class HybridReIDEngine:
    """Deep (ResNet50) primary, HSV fallback. Bir xil API."""

    def __init__(self, config=None):
        # Circular import dan qochish: to'g'ridan-to'g'ri fayldan yuklash
        import importlib.util, os
        _p = os.path.join(os.path.dirname(__file__), "deep_reid.py")
        _spec = importlib.util.spec_from_file_location("_deep_reid_mod", _p)
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        DeepReIDEngine = _m.DeepReIDEngine
        self.deep = DeepReIDEngine(config)
        self.hsv = BodyReIDEngine(config)
        self.enabled = True
        self.available = self.deep.available or self.hsv.available
        self.backend = "deep" if self.deep.available else "hsv"
        print(f"[ReID] 🧠 Hybrid engine: backend={self.backend} "
              f"(deep={self.deep.available}, hsv={self.hsv.available})", flush=True)

    def extract_features(self, crop_bgr):
        if self.backend == "deep":
            f = self.deep.extract_features(crop_bgr)
            if f is not None:
                return f
            # deep muvaffaqiyatsiz → hsv fallback
            return self.hsv.extract_features(crop_bgr)
        return self.hsv.extract_features(crop_bgr)

    def compute_similarity(self, a, b):
        if self.backend == "deep":
            return self.deep.compute_similarity(a, b)
        return self.hsv.compute_similarity(a, b)


# ServiceManager ReIDEngine(config) chaqiradi → Hybrid ishlatamiz
ReIDEngine = HybridReIDEngine
