"""Canonical shared InsightFace detection/embedding implementation."""
import threading
import numpy as np
from contextlib import nullcontext
from services.ml_service.pipeline.gpu_coordinator import gpu_coordinator

class InsightFaceEngine:
    def __init__(self, config):
        face=config.get("face",{});self.enabled=bool(face.get("enabled",True));self.lock=threading.Lock();self.available=False
        self.device=str(face.get("device","cpu"));self.model_name=str(face.get("model","buffalo_l"));raw=face.get("det_size",(320,320))
        self.det_size=tuple(raw) if isinstance(raw,(list,tuple)) else (int(raw),int(raw));self.app=None
        if not self.enabled:return
        from insightface.app import FaceAnalysis
        import onnxruntime
        cuda=self.device.startswith("cuda") and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        self.uses_cuda=cuda;providers=["CUDAExecutionProvider","CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
        self.app=FaceAnalysis(name=self.model_name,allowed_modules=["detection","recognition"],providers=providers)
        self.app.prepare(ctx_id=0 if cuda else -1,det_size=self.det_size);self.available=True

    @staticmethod
    def normalize(embedding):
        if embedding is None:return None
        value=np.asarray(embedding,np.float32);norm=np.linalg.norm(value)
        return value/norm if norm else None

    def detect(self,bgr,need_embedding=True):
        if not self.available or bgr is None:return []
        image=np.ascontiguousarray(bgr,dtype=np.uint8)
        gate=gpu_coordinator.secondary("FACE") if self.uses_cuda else nullcontext()
        with gate,self.lock:faces=self.app.get(image)
        output=[]
        for face in faces:
            raw_embedding=getattr(face,"normed_embedding",None)
            if raw_embedding is None:raw_embedding=getattr(face,"embedding",None)
            embedding=self.normalize(raw_embedding)
            if need_embedding and embedding is None:continue
            output.append({"bbox":[float(v) for v in face.bbox],"det_score":float(face.det_score),"embedding":embedding,
                           "landmarks":getattr(face,"kps",None),"pose":getattr(face,"pose",None)})
        return output

    def warmup(self):
        if self.available:self.detect(np.zeros((self.det_size[1],self.det_size[0],3),np.uint8))

    def close(self):self.app=None;self.available=False
