#!/usr/bin/env python3
"""Deterministic six-image detector-only benchmark for release diagnosis."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from shared.config import project_config
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.preprocessing import BatchPreprocessor
from services.ml_service.detection.person_detector import UltralyticsBackend

def stats(values):
 a=np.asarray(values,float);return {f"p{p}":float(np.percentile(a,p)) for p in (50,90,95,99)}|{"mean":float(a.mean()),"count":len(a)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--iterations",type=int,default=120);parser.add_argument("--warmup",type=int,default=20);args=parser.parse_args()
 cfg=project_config();size=tuple(cfg["ai"]["detector"]["imgsz"]);rng=np.random.default_rng(20260809)
 frames=[];now=time.time()
 for i in range(6):
  image=rng.integers(0,256,(720,1280,3),dtype=np.uint8);frames.append(FramePacket(f"FIXED-{i+1}",i,now,now,image,1280,720,now))
 batch=BatchOutput(1,now,tuple(frames));prepared=BatchPreprocessor(size).prepare(batch);backend=UltralyticsBackend(cfg,6)
 values=[]
 for i in range(args.warmup+args.iterations):
  _,timing=backend.infer(prepared)
  if i>=args.warmup:values.append(timing)
 print("shape",[6,3,*size],"dtype",str(backend.torch.float16 if backend.half else backend.torch.float32),"device",backend.device,"model_instances",1)
 for key in ("gpu_inference_ms","model_forward_wall_ms","h2d_ms","h2d_wall_ms","postprocess_ms"):print(key,json.dumps(stats([x[key] for x in values]),sort_keys=True))
 print("first10_gpu_ms",[round(x["gpu_inference_ms"],3) for x in values[:10]])
 backend.close()
if __name__=="__main__":main()
