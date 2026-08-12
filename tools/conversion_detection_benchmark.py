#!/usr/bin/env python3
"""Saved-camera quality guard for candidate early-resize conversion paths."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from shared.config import project_config
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.detection.person_detector import PersonDetector

def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);intersection=max(0,x2-x1)*max(0,y2-y1)
 return intersection/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection,1e-9)

def main():
 cfg=project_config();paths=sorted((ROOT/"data/detection_diagnostic/current").glob("CAM-*_A_original.jpg"));now=time.time();original=[];scaled=[];dimensions={}
 for index,path in enumerate(paths):
  image=cv2.imread(str(path));height,width=image.shape[:2];camera=path.name.split("_",1)[0];dimensions[camera]=(width,height)
  original.append(FramePacket(camera,index,now,now,image,width,height,now))
  if width>1280 or height>720:image=cv2.resize(image,(1280,720),interpolation=cv2.INTER_LINEAR)
  sh,sw=image.shape[:2];scaled.append(FramePacket(camera,index+100,now,now,image,sw,sh,now))
 detector=PersonDetector(cfg,max_frame_age_ms=10000,max_batch_size=6)
 try:
  first=detector.process_batch(BatchOutput(1,now,tuple(original)));second=detector.process_batch(BatchOutput(2,now,tuple(scaled)))
  by_first={item.camera_id:item for item in first.results};by_second={item.camera_id:item for item in second.results};report={};total_a=total_b=matched=0;ious=[];confidence_delta=[]
  for camera in sorted(by_first):
   width,height=dimensions[camera];a=list(by_first[camera].detections);b=[]
   sw=next(packet.width for packet in scaled if packet.camera_id==camera);sh=next(packet.height for packet in scaled if packet.camera_id==camera)
   for item in by_second[camera].detections:
    x1,y1,x2,y2=item.bbox_xyxy;b.append((item,(x1*width/sw,y1*height/sh,x2*width/sw,y2*height/sh)))
   pairs=[];remaining=set(range(len(b)))
   for item in a:
    choices=[(iou(item.bbox_xyxy,b[j][1]),j) for j in remaining]
    if choices:
     score,j=max(choices);remaining.remove(j)
     if score>=.5:pairs.append((score,abs(item.confidence-b[j][0].confidence)))
   total_a+=len(a);total_b+=len(b);matched+=len(pairs);ious.extend(x[0] for x in pairs);confidence_delta.extend(x[1] for x in pairs)
   report[camera]={"current_count":len(a),"early_resize_count":len(b),"matched_iou_0_5":len(pairs),"min_matched_iou":min((x[0] for x in pairs),default=None),"max_confidence_delta":max((x[1] for x in pairs),default=None)}
  report["summary"]={"current_count":total_a,"early_resize_count":total_b,"matched_iou_0_5":matched,"mean_iou":float(np.mean(ious)) if ious else None,"max_confidence_delta":max(confidence_delta,default=None),"recall_equivalent":total_a==total_b==matched}
  print(json.dumps(report,indent=2,sort_keys=True))
 finally:detector.close()
if __name__=="__main__":main()
