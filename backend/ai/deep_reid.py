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
from backend.core.gpu_utils import resolve_torch_device

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
        self.device = torch.device(resolve_torch_device("auto"))

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
        res = self.extract_features_batch([crop_bgr])
        return res[0] if res else None

    @torch.no_grad()
    def extract_features_batch(self, crops_bgr):
        if not crops_bgr:
            return []

        out = [None for _ in crops_bgr]
        tensors = []
        valid_indices = []

        for i, crop in enumerate(crops_bgr):
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < 30 or w < 20:
                continue
            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                t = self.transform(rgb)
                tensors.append(t)
                valid_indices.append(i)
            except Exception:
                continue

        if not tensors:
            return out

        try:
            batch_tensor = torch.stack(tensors).to(self.device)
            use_cuda = getattr(self.device, "type", str(self.device)) == "cuda"
            with torch.amp.autocast('cuda', enabled=use_cuda):
                if self.backend.startswith("osnet"):
                    try:
                        feats = self.model(batch_tensor, return_feats=True)
                    except TypeError:
                        feats = self.model(batch_tensor)
                    if isinstance(feats, (list, tuple)):
                        feats = feats[0]
                else:
                    feats = self.model(batch_tensor)

            feats = feats.view(feats.size(0), -1)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            feats_np = feats.cpu().numpy().astype(np.float32)

            for row_idx, orig_idx in enumerate(valid_indices):
                out[orig_idx] = feats_np[row_idx]

        except Exception as e:
            print(f"[DeepReID] batch extract error: {e}", flush=True)

        return out

    def compute_similarity(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return float(np.clip(np.dot(f1, f2), 0.0, 1.0))
