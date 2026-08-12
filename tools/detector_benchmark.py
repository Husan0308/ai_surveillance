#!/usr/bin/env python3
"""Deterministic six-image detector-only benchmark for release diagnosis."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from shared.config import project_config
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.preprocessing import BatchPreprocessor
from services.ml_service.detection.person_detector import UltralyticsBackend

def stats(values):
 a=np.asarray(values,float);return {f"p{p}":float(np.percentile(a,p)) for p in (50,90,95,99)}|{"mean":float(a.mean()),"count":len(a)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--iterations",type=int,default=120);parser.add_argument("--warmup",type=int,default=20);parser.add_argument("--batch-size",type=int,choices=(1,2,3,4,6),default=6);parser.add_argument("--output",type=Path);args=parser.parse_args()
 cfg=project_config();size=tuple(cfg["ai"]["detector"]["imgsz"]);now=time.time()
 paths=sorted((ROOT/"data/detection_diagnostic/current").glob("CAM-*_A_original.jpg"))[:args.batch_size]
 if len(paths)<args.batch_size:raise SystemExit("saved real-camera benchmark set is incomplete")
 frames=[]
 for i,path in enumerate(paths):
  image=cv2.imread(str(path));height,width=image.shape[:2];frames.append(FramePacket(path.name.split("_",1)[0],i,now,now,image,width,height,now))
 batch=BatchOutput(1,now,tuple(frames));prepared=BatchPreprocessor(size).prepare(batch);backend=UltralyticsBackend(cfg,args.batch_size)
 values=[]
 for i in range(args.warmup+args.iterations):
  _,timing=backend.infer(prepared)
  if i>=args.warmup:values.append(timing)
 result={"shape":[args.batch_size,3,*size],"dtype":str(backend.torch.float16 if backend.half else backend.torch.float32),"device":str(backend.device),"model_instances":1,"timings":{}}
 print("shape",result["shape"],"dtype",result["dtype"],"device",result["device"],"model_instances",1)
 for key in ("gpu_inference_ms","model_forward_wall_ms","h2d_ms","h2d_wall_ms","postprocess_ms"):
  summary=stats([x[key] for x in values]);summary["per_image_p50"]=summary["p50"]/args.batch_size;summary["processed_fps_p50"]=args.batch_size*1000/summary["p50"];result["timings"][key]=summary;print(key,json.dumps(summary,sort_keys=True))
 result["first10_gpu_ms"]=[round(x["gpu_inference_ms"],3) for x in values[:10]]
 print("first10_gpu_ms",result["first10_gpu_ms"])
 if args.output:
  args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix(args.output.suffix+".tmp");temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");temporary.replace(args.output)
 backend.close()
if __name__=="__main__":main()
