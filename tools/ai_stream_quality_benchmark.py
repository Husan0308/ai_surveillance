#!/usr/bin/env python3
"""Paired production-detector comparison for explicitly configured main/AI streams."""
from __future__ import annotations
import argparse,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from shared.config import camera_config,project_config
from shared.settings import ServiceSettings
from services.ml_service.cameras.config import _normalize
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.cameras.gstreamer import GStreamerCapture
from services.ml_service.detection.person_detector import PersonDetector
from services.ml_service.pipeline.batch import BatchOutput

def records():
 defaults={str(x['id']):x for x in camera_config().get('cameras',[])}
 with sqlite3.connect(ServiceSettings.from_env().database_path) as db:rows=db.execute("select id,data from api_resources where resource='cameras' order by id").fetchall()
 return [_normalize({'id':cid,**json.loads(raw)},defaults) for cid,raw in rows]
def role_config(item,role):
 source=item['display_source'] if role=='main' else item['ai_source'];codec=item.get('display_codec') if role=='main' else item.get('ai_codec')
 return {**item,'source':source,'codec':codec or item['codec']}
def summary(result,packet):
 ds=result.results[0].detections;conf=[d.confidence for d in ds];small=[d for d in ds if (d.bbox_xyxy[3]-d.bbox_xyxy[1])/packet.height<.15]
 return {'count':len(ds),'mean_confidence':float(np.mean(conf)) if conf else 0.0,'small_person_count':len(small),'resolution':f'{packet.width}x{packet.height}','detections':[{'confidence':round(d.confidence,4),'bbox':[round(v,1) for v in d.bbox_xyxy]} for d in ds]}
def main():
 p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=3);a=p.parse_args();detector=PersonDetector(project_config(),max_frame_age_ms=10000,max_batch_size=2);output=[];frame_id=0
 try:
  for item in records():
   if item.get('ai_source')==item.get('display_source'):
    output.append({'camera':item['id'],'status':'same_stream','selected_ai_source':item['ai_source']});continue
   captures={role:GStreamerCapture(role_config(item,role)) for role in ('main','ai')}
   try:
    for sample in range(a.samples):
     for role in ('main','ai'):
      ok,frame=captures[role].read()
      if not ok:output.append({'camera':item['id'],'role':role,'sample':sample+1,'error':'capture_failed'});continue
      now=time.time();frame_id+=1;h,w=frame.shape[:2];packet=FramePacket(f"{item['id']}-{role}",frame_id,now,now,frame,w,h,now);result=detector.process_batch(BatchOutput(frame_id,now,(packet,)))
      output.append({'camera':item['id'],'role':role,'sample':sample+1,**summary(result,packet)})
   finally:
    for capture in captures.values():capture.release()
 finally:detector.close()
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
