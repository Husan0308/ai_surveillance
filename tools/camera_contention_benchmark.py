#!/usr/bin/env python3
"""Controlled NVDEC/nvvideoconvert contention matrix with a fixed detector batch."""
from __future__ import annotations
import argparse,json,sqlite3,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from shared.config import project_config,camera_config
from shared.settings import ServiceSettings
from services.ml_service.cameras.config import _normalize
from services.ml_service.cameras.manager import CameraManager
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.preprocessing import BatchPreprocessor
from services.ml_service.detection.person_detector import UltralyticsBackend

def percentile(v,p):return float(np.percentile(np.asarray(v,float),p))
def canonical():
 defaults={str(x['id']):x for x in camera_config().get('cameras',[])}
 with sqlite3.connect(ServiceSettings.from_env().database_path) as db:rows=db.execute("select id,data from api_resources where resource='cameras' order by id").fetchall()
 return [_normalize({'id':cid,**json.loads(raw)},defaults) for cid,raw in rows]
def gpu():
 out=subprocess.check_output(['nvidia-smi','--query-gpu=pstate,clocks.sm,clocks.mem,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip();return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--iterations",type=int,default=40);p.add_argument("--warmup",type=int,default=8);p.add_argument("--lower-substreams",action="store_true");args=p.parse_args();cfg=project_config();size=tuple(cfg['ai']['detector']['imgsz']);now=time.time();rng=np.random.default_rng(7)
 frames=tuple(FramePacket(f'FIXED-{i}',i,now,now,rng.integers(0,256,(720,1280,3),dtype=np.uint8),1280,720,now) for i in range(6));prepared=BatchPreprocessor(size).prepare(BatchOutput(1,now,frames));backend=UltralyticsBackend(cfg,6);manager=CameraManager();manager.start();cameras=canonical()
 if args.lower_substreams:
  bootstrap={str(x["id"]):x for x in camera_config().get("cameras",[])}
  cameras=[({**item,**{k:bootstrap[item["id"]][k] for k in ("source","codec","resolution")}} if item["id"] in ("CAM-03","CAM-06") else item) for item in cameras]
 try:
  for count in ((6,) if args.lower_substreams else (0,1,2,4,6)):
   manager.configure(cameras[:count]);deadline=time.time()+8
   while count and time.time()<deadline and sum(x['online'] for x in manager.metrics().values())<count:time.sleep(.2)
   online=sum(x['online'] for x in manager.metrics().values());values=[];walls=[]
   for i in range(args.warmup+args.iterations):
    _,timing=backend.infer(prepared)
    if i>=args.warmup:values.append(timing['gpu_inference_ms']);walls.append(timing['model_forward_wall_ms'])
   print(json.dumps({'camera_pipelines':count,'online':online,'gpu_forward_p50':percentile(values,50),'gpu_forward_p95':percentile(values,95),'wall_forward_p50':percentile(walls,50),"lower_substreams":args.lower_substreams,"gpu_state":gpu()}),flush=True)
 finally:manager.shutdown();backend.close()
if __name__=='__main__':main()
