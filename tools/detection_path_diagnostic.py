#!/usr/bin/env python3
"""Capture one real frame per camera and save bounded detector-path evidence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from shared.config import project_config
from services.ml_service.cameras.config import load_camera_configs
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.cameras.manager import CameraManager
from services.ml_service.detection.person_detector import PersonDetector
from services.ml_service.pipeline.batch import BatchOutput

def capture(timeout):
 configs=load_camera_configs();manager=CameraManager();manager.configure(configs);manager.start();packets={}
 try:
  deadline=time.time()+timeout
  while time.time()<deadline and len(packets)<len(configs):
   for cid,buffer in manager.buffers().items():
    packet=buffer.take()
    if packet is not None:packets[cid]=packet
   time.sleep(.05)
  metrics=manager.metrics()
 finally:manager.shutdown()
 return [packets[x] for x in sorted(packets)],metrics

def overlay(frame,detections):
 image=frame.copy()
 for detection in detections:
  x1,y1,x2,y2=(int(round(v)) for v in detection.bbox_xyxy);cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2);cv2.putText(image,f"person {detection.confidence:.2f}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
 return image

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="data/detection_diagnostic/current");p.add_argument("--timeout",type=float,default=25);p.add_argument("--reuse",action="store_true",help="analyze existing *_original.jpg files without opening cameras");a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.reuse:
  packets=[];metrics={}
  for index,path in enumerate((x for x in sorted(out.glob("*_original.jpg")) if x.name.count("_")==1),1):
   frame=cv2.imread(str(path));height,width=frame.shape[:2];cid=path.name.split("_",1)[0];packets.append(FramePacket(cid,index,time.time(),time.time(),frame,width,height,time.time()));metrics[cid]={"backend":"captured-real-frame"}
 else:packets,metrics=capture(a.timeout)
 if not packets:raise SystemExit("No real camera frames captured")
 now=time.time();packets=[FramePacket(x.camera_id,x.frame_id,now,now,x.frame,x.width,x.height,now) for x in packets];detector=PersonDetector(project_config(),max_frame_age_ms=60000,max_batch_size=len(packets))
 try:
  batch=BatchOutput(1,now,tuple(packets));prepared=detector.preprocessor.prepare(batch);result=detector.process_batch(batch);backend=detector.backend
  official=backend.model.predict([x.frame for x in packets],classes=[0],conf=backend.conf,iou=backend.iou,max_det=backend.max_det,imgsz=backend.imgsz,device=backend.device,half=backend.half,verbose=False)
  report={"model_names":backend.model.names,"person_class_id":next((k for k,v in backend.model.names.items() if v=="person"),None),"cameras":[]}
  for packet,tensor,custom,reference in zip(packets,prepared.images_nchw,result.results,official):
   cid=packet.camera_id;model_view=np.ascontiguousarray(tensor.transpose(1,2,0)[...,::-1]);rows=np.empty((0,6)) if reference.boxes is None else reference.boxes.data.cpu().numpy()
   cv2.imwrite(str(out/f"{cid}_A_original.jpg"),packet.frame);cv2.imwrite(str(out/f"{cid}_B_model_input.jpg"),model_view);cv2.imwrite(str(out/f"{cid}_D_filtered_overlay.jpg"),overlay(packet.frame,custom.detections))
   item={"camera_id":cid,"ai_resolution":[packet.width,packet.height],"decoder":metrics[cid].get("backend"),"tensor_shape":list(tensor.shape),"tensor_dtype":str(tensor.dtype),"tensor_contiguous":bool(tensor.flags.c_contiguous),"custom_count":len(custom.detections),"custom_confidences":[round(x.confidence,5) for x in custom.detections],"official_count":len(rows),"official_confidences":[round(float(x[4]),5) for x in rows]};(out/f"{cid}_C_raw_summary.json").write_text(json.dumps(item,indent=2));report["cameras"].append(item)
  (out/"report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 finally:detector.close()
if __name__=="__main__":main()
