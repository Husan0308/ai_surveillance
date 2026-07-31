"""
Deep Re-Identification Engine.
Primary: OSNet-x1_0 Market-1501 (haqiqiy ReID, kiyim/tana tekstura).
Fallback: OSNet ImageNet -> ResNet50 ImageNet.
"""
import cv2
import numpy as np
import os
import torch
import torch.nn as nn

try:
    from torchvision import transforms as T
    _TV_OK = True
except Exception:
    _TV_OK = False


class DeepReIDEngine:
    def __init__(self, config=None):
        self.enabled = True
        self.available = False
        self.backend = "none"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not _TV_OK:
            print("[DeepReID] torchvision yo'q", flush=True)
            return

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 1) OSNet (Market-1501 fine-tune yoki ImageNet)
        try:
            import torchreid
            mkt = os.path.expanduser("~/.cache/torch/checkpoints/osnet_x1_0_market1501.pth")
            if os.path.exists(mkt):
                model = torchreid.models.build_model(name="osnet_x1_0", num_classes=751, loss="softmax", pretrained=False)
                ckpt = torch.load(mkt, map_location="cpu")
                sd = ckpt.get("state_dict", ckpt)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model.load_state_dict(sd, strict=False)
                self.backend = "osnet_market1501"
            else:
                model = torchreid.models.build_model(name="osnet_x1_0", num_classes=1000, loss="softmax", pretrained=True)
                self.backend = "osnet_imagenet"
            self.model = model.eval().to(self.device)
            self.available = True
            print(f"[DeepReID] {self.backend} | device={self.device} | dim=512", flush=True)
            return
        except Exception as e:
            print(f"[DeepReID] OSNet yo'q ({e}) -> ResNet50 fallback", flush=True)

        # 2) ResNet50 ImageNet fallback
        try:
            import torchvision.models as models
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(base.children())[:-1]).eval().to(self.device)
            self.backend = "resnet50"
            self.available = True
            print(f"[DeepReID] resnet50 fallback | dim=2048", flush=True)
        except Exception as e:
            print(f"[DeepReID] init error: {e}", flush=True)

    @torch.no_grad()
    def extract_features(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 30 or w < 20:
            return None
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            if self.backend.startswith("osnet"):
                try:
                    feat = self.model(tensor, return_feats=True)
                except TypeError:
                    feat = self.model(tensor)
                if isinstance(feat, (list, tuple)):
                    feat = feat[0]
            else:
                feat = self.model(tensor).squeeze()
            feat = feat.view(feat.size(0), -1) if feat.dim() > 1 else feat.unsqueeze(0)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception:
            return None

    def compute_similarity(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(np.clip(np.dot(f1, f2), 0.0, 1.0))
