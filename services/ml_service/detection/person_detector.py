"""Permanent, dynamically batched person-only YOLO detector."""
from collections import deque
from pathlib import Path
import json
import threading
import time
import numpy as np
from services.ml_service.detection.schemas import Detection, CameraDetectionResult, DetectionBatchResult
from services.ml_service.metrics.detector_metrics import DetectorBatchMetrics, DetectorMetrics
from services.ml_service.pipeline.batch import BatchOutput
from services.ml_service.pipeline.preprocessing import BatchPreprocessor, original_bbox
from services.ml_service.pipeline.gpu_coordinator import gpu_coordinator
from services.ml_service.cameras.frame import FramePacket
from services.ml_service.detection.roi import ROIRecoveryScheduler,bbox_anchor,crop_rectangle,fuse_detections,map_crop_bbox,point_in_polygon,source_polygon
from shared.logging import get_logger
log = get_logger(__name__)

def _suppress_exact_person_duplicates(rows):
    """Defensive NMS for virtually identical end-to-end object queries only."""
    kept=[]
    for row in sorted(rows,key=lambda value:float(value[4]),reverse=True):
        box=row[:4];area=max(1.0,float((box[2]-box[0])*(box[3]-box[1])));duplicate=False
        for other in kept:
            candidate=other[:4];other_area=max(1.0,float((candidate[2]-candidate[0])*(candidate[3]-candidate[1])));inter=max(0.0,float(min(box[2],candidate[2])-max(box[0],candidate[0])))*max(0.0,float(min(box[3],candidate[3])-max(box[1],candidate[1])));union=area+other_area-inter;iou=inter/union;containment=inter/min(area,other_area);scale=min(area,other_area)/max(area,other_area)
            acx,acy=float((box[0]+box[2])*.5),float((box[1]+box[3])*.5);bcx,bcy=float((candidate[0]+candidate[2])*.5),float((candidate[1]+candidate[3])*.5);center=((acx-bcx)**2+(acy-bcy)**2)**.5/max(1.0,min(area,other_area)**.5)
            if iou>=.97 and containment>=.985 and scale>=.95 and center<=.03:duplicate=True;break
        if not duplicate:kept.append(row)
    return np.asarray(kept,dtype=np.float32).reshape((-1,rows.shape[1])) if kept else np.empty((0,rows.shape[1]),np.float32)

def filter_end2end_predictions(array,conf,classes,max_det):
    """Filter native end-to-end output and remove only exact query duplicates."""
    allowed=None if classes is None else np.asarray(classes,dtype=np.int64)
    output=[]
    for rows in array:
        selected=rows[rows[:,4]>float(conf)]
        if allowed is not None:selected=selected[np.isin(selected[:,5].astype(np.int64,copy=False),allowed)]
        selected=_suppress_exact_person_duplicates(selected)
        output.append(np.ascontiguousarray(selected[:max_det,:6],dtype=np.float32))
    return output

class UltralyticsBackend:
    coordinates_original = True
    def __init__(self, config, max_batch_size=6):
        import torch
        from ultralytics import YOLO
        cfg = config.get("ai", {}).get("detector", {})
        self.torch = torch; requested = cfg.get("device", "auto")
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else str(requested)
        self.conf, self.iou = float(cfg.get("conf", .3)), float(cfg.get("iou", .45))
        self.max_det, self.half = int(cfg.get("max_det", 50)), bool(cfg.get("half", False))
        raw_size = cfg.get("imgsz", (448, 800)); self.imgsz = tuple(raw_size) if isinstance(raw_size, (list, tuple)) else (raw_size, raw_size)
        self.model = YOLO(cfg.get("model", "models/yolo26n.pt")); self.model.to(self.device);self._runtime_lock=threading.Lock();self._gpu_inflight=0;self._max_gpu_inflight=0;self._gpu_batches_completed=0;self._last_gpu_launch=None;self._last_gpu_end=None;self._pinned_staging={};self._device_staging={}
        dummy = np.zeros((self.imgsz[0], self.imgsz[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            self.model.predict([dummy] * max(1, int(max_batch_size)), classes=[0], conf=self.conf,
                iou=self.iou, max_det=self.max_det, imgsz=self.imgsz, device=self.device,
                half=self.half, verbose=False)
        log.info("PersonDetector warmup complete: device=%s max_batch=%d", self.device, max_batch_size)

    def infer(self, prepared):
        torch, predictor = self.torch, self.model.predictor
        from ultralytics.utils import nms
        cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        dtype = torch.float16 if getattr(predictor.model, "fp16", False) else torch.float32
        with torch.inference_mode():
            source=torch.from_numpy(prepared.images_nchw);numpy_to_torch_ms=0.0;shape=tuple(source.shape)
            if cuda:
                stream=torch.cuda.current_stream(self.device);host=self._pinned_staging.get(shape);pinned_reused=host is not None
                phase=time.perf_counter()
                if host is None:host=torch.empty(shape,dtype=torch.uint8,pin_memory=True);self._pinned_staging[shape]=host
                host.copy_(source);host_prepare_ms=(time.perf_counter()-phase)*1000;pin_memory_ms=0.0 if pinned_reused else host_prepare_ms
                device_key=(shape,dtype);tensor=self._device_staging.get(device_key);device_reused=tensor is not None
                if tensor is None:tensor=torch.empty(shape,dtype=dtype,device=self.device);self._device_staging[device_key]=tensor
                hs,he=torch.cuda.Event(True),torch.cuda.Event(True);phase=time.perf_counter();hs.record(stream);tensor.copy_(host,non_blocking=True);he.record(stream);h2d_enqueue_ms=(time.perf_counter()-phase)*1000;he.synchronize();h2d_wall_ms=(time.perf_counter()-phase)*1000;tensor.mul_(1/255.0);ins,ine=torch.cuda.Event(True),torch.cuda.Event(True);ins.record(stream)
            else:
                phase=time.perf_counter();tensor=source.to(self.device,dtype=dtype).mul_(1/255.0);h2d_wall_ms=(time.perf_counter()-phase)*1000;h2d_enqueue_ms=h2d_wall_ms;host_prepare_ms=0.0;pin_memory_ms=0.0;pinned_reused=device_reused=False
            infer_wall = time.perf_counter()
            with self._runtime_lock:
                launch_gap_ms=0.0 if self._last_gpu_launch is None else (infer_wall-self._last_gpu_launch)*1000;gpu_idle_ms=0.0 if self._last_gpu_end is None else (infer_wall-self._last_gpu_end)*1000;self._last_gpu_launch=infer_wall;self._gpu_inflight+=1;self._max_gpu_inflight=max(self._max_gpu_inflight,self._gpu_inflight)
            try:
                predictions = predictor.inference(tensor)
                if cuda:ine.record(stream);ine.synchronize()
            finally:
                with self._runtime_lock:self._gpu_inflight-=1;self._gpu_batches_completed+=1;self._last_gpu_end=time.perf_counter()
            infer_wall_ms = (time.perf_counter() - infer_wall) * 1000
            end2end=bool(getattr(predictor.model,"end2end",False))
            if end2end:
                raw=predictions[0] if isinstance(predictions,(list,tuple)) else predictions
                phase=time.perf_counter();host_predictions=raw.detach().cpu().numpy();tensor_to_cpu_ms=(time.perf_counter()-phase)*1000
                phase=time.perf_counter();boxes=filter_end2end_predictions(host_predictions,predictor.args.conf,predictor.args.classes,predictor.args.max_det);nms_ms=(time.perf_counter()-phase)*1000
                results_ms=0.0;camera_copy_ms=[];self.coordinates_original=False
            else:
                phase=time.perf_counter();preds=nms.non_max_suppression(predictions,predictor.args.conf,predictor.args.iou,
                    predictor.args.classes,predictor.args.agnostic_nms,max_det=predictor.args.max_det,
                    nc=0 if predictor.args.task=="detect" else len(predictor.model.names),end2end=False,rotated=False)
                nms_ms=(time.perf_counter()-phase)*1000
                phase=time.perf_counter();results=predictor.construct_results(preds,tensor,[p.frame for p in prepared.batch.frames]);results_ms=(time.perf_counter()-phase)*1000
                boxes=[];camera_copy_ms=[]
                for result in results:
                    copy_started=time.perf_counter();boxes.append(np.empty((0,6),np.float32) if result.boxes is None or len(result.boxes)==0 else result.boxes.data.detach().cpu().numpy());camera_copy_ms.append((time.perf_counter()-copy_started)*1000)
                tensor_to_cpu_ms=sum(camera_copy_ms);self.coordinates_original=True
            post_ms=nms_ms+results_ms+tensor_to_cpu_ms
        return boxes, {"h2d_ms": hs.elapsed_time(he) if cuda else 0.0,
                       "gpu_inference_ms": ins.elapsed_time(ine) if cuda else infer_wall_ms,"time_since_previous_gpu_launch_ms":launch_gap_ms,"gpu_idle_between_batches_ms":gpu_idle_ms,"cuda_wall_event_delta_ms":infer_wall_ms-(ins.elapsed_time(ine) if cuda else infer_wall_ms),
                       "postprocess_ms":post_ms,"numpy_to_torch_ms":numpy_to_torch_ms,"pin_memory_ms":pin_memory_ms,
                       "h2d_wall_ms":h2d_wall_ms,"h2d_enqueue_ms":h2d_enqueue_ms,"host_prepare_ms":host_prepare_ms,"pinned_reused":pinned_reused,"device_reused":device_reused,"pinned_storage_id":int(host.data_ptr()) if cuda else 0,"device_storage_id":int(tensor.data_ptr()) if cuda else 0,"tensor_batch_size":int(tensor.shape[0]),"model_output_batch_size":len(boxes),"model_forward_wall_ms":infer_wall_ms,"nms_ms":nms_ms,
                       "results_construction_ms":results_ms,"tensor_to_cpu_ms":tensor_to_cpu_ms,
                       "camera_tensor_to_cpu_ms":camera_copy_ms}

    def runtime_snapshot(self):
        with self._runtime_lock:return {"gpu_inflight_batches":self._gpu_inflight,"max_gpu_inflight_batches":self._max_gpu_inflight,"gpu_batches_completed":self._gpu_batches_completed}
    def close(self): self.model = None

class PersonDetector:
    def __init__(self, config, backend=None, max_frame_age_ms=None, max_batch_size=6):
        cfg = config.get("ai", {}).get("detector", {}); raw = cfg.get("imgsz", (448, 800))
        self.input_size = tuple(raw) if isinstance(raw, (list, tuple)) else (raw, raw)
        self.max_frame_age_ms = float(max_frame_age_ms or config.get("ai", {}).get("max_frame_age_ms", 120))
        self.max_det = int(cfg.get("max_det", 50)); self.min_w = float(cfg.get("min_box_width", 3)); self.min_h = float(cfg.get("min_box_height", 6))
        self.low_conf = float(cfg.get("low_conf_size_threshold", .22)); self.low_w = float(cfg.get("low_conf_min_width", 12)); self.low_h = float(cfg.get("low_conf_min_height", 36))
        self.preprocessor = BatchPreprocessor(self.input_size)
        roi_cfg=config.get("ai",{}).get("roi_recovery",{});self.roi=ROIRecoveryScheduler(roi_cfg.get("discovery_interval_ms",2000),roi_cfg.get("urgent_interval_ms",1000),roi_cfg.get("max_task_age_ms",500))
        self.backend = backend or UltralyticsBackend(config, max_batch_size)
        self.metrics = DetectorMetrics(); self._lock = threading.Lock();self._runtime_lock=threading.Lock();self._batches_started=0;self._batches_completed=0;self._batches_pending=0;self._max_batches_pending=0;self.diagnostic_context_provider=None
        self._seen, self._seen_order = set(), deque(maxlen=10000)
        self._raw_ambiguous_pairs=0;self._software_duplicates_suppressed=0;self._active_track_hints={};self._ambiguous_crops_saved=0;self._ambiguous_last_saved={};self._ambiguous_dir=Path("data/diagnostics/ambiguous_detections")

    def process_batch(self, batch: BatchOutput):
        detector_entry_monotonic=time.monotonic();started = time.time(); accepted = [];validation_started=time.perf_counter()
        with self._runtime_lock:self._batches_pending+=1;self._max_batches_pending=max(self._max_batches_pending,self._batches_pending)
        with self._lock:
            with self._runtime_lock:self._batches_pending-=1;self._batches_started+=1
            now = time.time()
            for packet in batch.frames:
                key = (packet.camera_id, packet.frame_id)
                if key in self._seen: self.metrics.duplicate_inference_prevented += 1; continue
                if (now - packet.receive_timestamp) * 1000 > self.max_frame_age_ms:
                    self.metrics.stale_drops_before_inference += 1; continue
                if len(self._seen_order) == self._seen_order.maxlen: self._seen.discard(self._seen_order[0])
                self._seen.add(key); self._seen_order.append(key); accepted.append(packet)
            if not accepted:
                with self._runtime_lock:self._batches_completed+=1
                return DetectionBatchResult(batch.batch_id, started, time.time(), ())
            validation_ms=(time.perf_counter()-validation_started)*1000
            fresh = BatchOutput(batch.batch_id, batch.created_timestamp, tuple(accepted),batch.build_started_monotonic,batch.build_completed_monotonic); wall = time.perf_counter()
            prepared = self.preprocessor.prepare(fresh)
            gate_before=gpu_coordinator.snapshot();gate_request=time.monotonic()
            with gpu_coordinator.primary("YOLO") as admission:raw_results, timing = self.backend.infer(prepared)
            parse = time.perf_counter();camera_results=[];filter_ms=rescale_ms=result_build_ms=0.0
            for packet, transform, rows in zip(fresh.frames, prepared.transforms, raw_results):
                detections=[];phase=time.perf_counter();camera_rescale_ms=0.0
                for row in rows:
                    if len(row) < 6 or int(row[5]) != 0: continue
                    bbox = tuple(float(v) for v in row[:4])
                    if not getattr(self.backend, "coordinates_original", False):
                        rescale_started=time.perf_counter();bbox=original_bbox(bbox,transform);value=(time.perf_counter()-rescale_started)*1000;rescale_ms+=value;camera_rescale_ms+=value
                    else:
                        x1, y1, x2, y2 = bbox; bbox = (min(max(x1, 0), packet.width), min(max(y1, 0), packet.height), min(max(x2, 0), packet.width), min(max(y2, 0), packet.height))
                    width, height, confidence = bbox[2] - bbox[0], bbox[3] - bbox[1], float(row[4])
                    if width < self.min_w or height < self.min_h: continue
                    if confidence < self.low_conf and (width < self.low_w or height < self.low_h): continue
                    ratio = width / max(height, 1)
                    if ratio < .08 or ratio > 8: continue
                    detections.append(Detection(bbox,confidence,detection_source="FULL_FRAME",detection_id=f"{packet.camera_id}:{packet.frame_id}:FULL:{len(detections)}"))
                    if len(detections) >= self.max_det: break
                self._audit_ambiguous_pairs(packet,detections)
                detections=self._suppress_contained_person_duplicates(packet.camera_id,detections)
                filter_ms+=max(0.0,(time.perf_counter()-phase)*1000-camera_rescale_ms)
                phase=time.perf_counter();camera_results.append(CameraDetectionResult(packet.camera_id,packet.frame_id,packet.capture_timestamp,packet.receive_timestamp,tuple(detections),packet.capture_monotonic,packet.width,packet.height));result_build_ms+=(time.perf_counter()-phase)*1000
            runtime=self.runtime_snapshot();self.roi.update_pressure(timing.get("gpu_inference_ms",0),len(accepted),6,runtime.get("detector_batches_pending",0))
            camera_results=self._recover_roi(fresh,tuple(camera_results))
            parse_ms = (time.perf_counter() - parse) * 1000; completed = time.time(); wall_ms = (time.perf_counter() - wall) * 1000
            phases={"frame_validation":validation_ms,"preprocess_total":prepared.preprocess_ms,"postprocess_total":float(timing.get("postprocess_ms",0))+parse_ms,"image_resize":prepared.resize_ms,"letterbox":prepared.letterbox_ms,
                "bgr_to_rgb":prepared.bgr_to_rgb_ms,"numpy_stacking":prepared.numpy_stack_ms,
                "numpy_to_torch":float(timing.get("numpy_to_torch_ms",0)),"pinned_memory":float(timing.get("pin_memory_ms",0)),"host_preparation":float(timing.get("host_prepare_ms",0)),
                "h2d_enqueue":float(timing.get("h2d_enqueue_ms",0)),"h2d_wall_wait":float(timing.get("h2d_wall_ms",0)),"cpu_to_gpu":float(timing.get("h2d_ms",0)),"model_forward":float(timing.get("gpu_inference_ms",0)),"model_forward_wall":float(timing.get("model_forward_wall_ms",0)),"cuda_wall_event_delta":float(timing.get("cuda_wall_event_delta_ms",0)),"batch_period":float(timing.get("time_since_previous_gpu_launch_ms",0)),"time_since_previous_gpu_launch":float(timing.get("time_since_previous_gpu_launch_ms",0)),"gpu_idle_between_batches":float(timing.get("gpu_idle_between_batches_ms",0)),"gpu_gate_wait":float(admission.get("wait_ms",(time.monotonic()-gate_request)*1000)),"scheduler_wait":max(0.0,(detector_entry_monotonic-(batch.build_completed_monotonic or detector_entry_monotonic))*1000),"batch_build":max(0.0,(batch.build_completed_monotonic-batch.build_started_monotonic)*1000) if batch.build_started_monotonic else 0.0,
                "nms":float(timing.get("nms_ms",0)),"results_construction":float(timing.get("results_construction_ms",0)),
                "tensor_to_cpu":float(timing.get("tensor_to_cpu_ms",0)),"bbox_filtering":max(0.0,filter_ms),
                "coordinate_rescaling":rescale_ms,"camera_result_construction":result_build_ms,"pure_detector_wall":wall_ms}
            phases.update({"pinned_reused":float(bool(timing.get("pinned_reused"))),"device_reused":float(bool(timing.get("device_reused"))),"pinned_storage_id":float(timing.get("pinned_storage_id",0)),"device_storage_id":float(timing.get("device_storage_id",0))})
            audit={"timestamp":started,"batch_id":batch.batch_id,"batch_size":len(accepted),"camera_ids":tuple(p.camera_id for p in accepted),"gpu_owner_before":gate_before.get("active"),"gpu_admission":dict(admission),"timing":dict(phases)}
            if float(timing.get("gpu_inference_ms",0))>=1000 and self.diagnostic_context_provider:
                try:audit["runtime"]=self.diagnostic_context_provider()
                except Exception as exc:audit["runtime_error"]=str(exc)
            self.metrics.record(DetectorBatchMetrics(batch.batch_id,len(accepted),tuple(p.camera_id for p in accepted),len(batch.frames),int(timing.get("tensor_batch_size",len(accepted))),int(timing.get("model_output_batch_size",len(raw_results))),started,completed,prepared.preprocess_ms,prepared.cpu_pack_ms,float(timing.get("h2d_ms",0)),float(timing.get("gpu_inference_ms",0)),float(timing.get("postprocess_ms",0)),parse_ms,wall_ms,(completed-min(p.capture_timestamp for p in accepted))*1000,phases,{packet.camera_id:(completed-packet.capture_timestamp)*1000 for packet in accepted}),audit)
            with self._runtime_lock:self._batches_completed+=1
            return DetectionBatchResult(batch.batch_id, started, completed, tuple(camera_results))

    def configure_rois(self,cameras):self.roi.configure(cameras)

    def update_track_hints(self,tracking):
        self.roi.update_track_hints(tracking)
        self._active_track_hints={camera.camera_id:tuple(track.bbox for track in camera.tracks if track.confirmed and not track.misses) for camera in tracking.results}


    def roi_snapshot(self):return self.roi.snapshot()

    @staticmethod
    def _box_iou(a,b):
        x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(1.0,(a[2]-a[0])*(a[3]-a[1]));bb=max(1.0,(b[2]-b[0])*(b[3]-b[1]));return inter/(aa+bb-inter)

    def _two_track_support(self,camera_id,a,b):
        hints=self._active_track_hints.get(camera_id,())
        if len(hints)<2:return False
        best_a=sorted(((self._box_iou(a,h),index) for index,h in enumerate(hints)),reverse=True)
        best_b=sorted(((self._box_iou(b,h),index) for index,h in enumerate(hints)),reverse=True)
        return any(score_a>=.35 and score_b>=.35 and index_a!=index_b for score_a,index_a in best_a for score_b,index_b in best_b)

    def _suppress_contained_person_duplicates(self,camera_id,detections):
        output=[]
        for candidate in detections:
            duplicate_index=None
            for index,item in enumerate(output):
                geometry=self._pair_geometry(item.bbox_xyxy,candidate.bbox_xyxy)
                contained=geometry["containment"]>=.88 and geometry["center_distance"]<=.28 and geometry["scale_ratio"]>=.25
                if contained and not (min(item.confidence,candidate.confidence)>=.45 and self._two_track_support(camera_id,item.bbox_xyxy,candidate.bbox_xyxy)):duplicate_index=index;break
            if duplicate_index is None:output.append(candidate);continue
            existing=output[duplicate_index];ea=max(1.0,(existing.bbox_xyxy[2]-existing.bbox_xyxy[0])*(existing.bbox_xyxy[3]-existing.bbox_xyxy[1]));ca=max(1.0,(candidate.bbox_xyxy[2]-candidate.bbox_xyxy[0])*(candidate.bbox_xyxy[3]-candidate.bbox_xyxy[1]))
            if ca>ea and candidate.confidence>=existing.confidence*.45 or candidate.confidence>existing.confidence and not (ea>ca and existing.confidence>=candidate.confidence*.45):output[duplicate_index]=candidate
            self._software_duplicates_suppressed+=1
        return output

    @staticmethod
    def _pair_geometry(a,b):
        ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b;inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1));aa=max(1.0,(ax2-ax1)*(ay2-ay1));bb=max(1.0,(bx2-bx1)*(by2-by1));union=aa+bb-inter;iou=inter/union;containment=inter/min(aa,bb);ac=((ax1+ax2)*.5,(ay1+ay2)*.5);bc=((bx1+bx2)*.5,(by1+by2)*.5);scale=max(1.0,min(aa,bb)**.5);center=((ac[0]-bc[0])**2+(ac[1]-bc[1])**2)**.5/scale
        return {"iou":iou,"containment":containment,"center_distance":center,"scale_ratio":min(aa,bb)/max(aa,bb)}

    def _audit_ambiguous_pairs(self,packet,detections):
        pairs=[]
        for index,a in enumerate(detections):
            for b in detections[index+1:]:
                geometry=self._pair_geometry(a.bbox_xyxy,b.bbox_xyxy)
                if geometry["iou"]>=.70 or geometry["containment"]>=.88 and geometry["center_distance"]<=.25:pairs.append((a,b,geometry))
        self._raw_ambiguous_pairs+=len(pairs)
        if not pairs or self._ambiguous_crops_saved>=24 or packet.capture_timestamp-self._ambiguous_last_saved.get(packet.camera_id,0)<1.0:return
        try:
            import cv2
            a,b,geometry=pairs[0];boxes=(a.bbox_xyxy,b.bbox_xyxy);x1=max(0,int(min(box[0] for box in boxes))-24);y1=max(0,int(min(box[1] for box in boxes))-24);x2=min(packet.width,int(max(box[2] for box in boxes))+24);y2=min(packet.height,int(max(box[3] for box in boxes))+24);crop=packet.frame[y1:y2,x1:x2].copy()
            if crop.size==0:return
            colors=((0,255,255),(255,0,255))
            for color,box,item in zip(colors,boxes,(a,b)):cv2.rectangle(crop,(int(box[0])-x1,int(box[1])-y1),(int(box[2])-x1,int(box[3])-y1),color,2);cv2.putText(crop,f"{item.confidence:.2f}",(int(box[0])-x1,max(14,int(box[1])-y1)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
            self._ambiguous_dir.mkdir(parents=True,exist_ok=True);stem=f"{packet.camera_id}_{packet.frame_id}_{int(packet.capture_timestamp*1000)}";cv2.imwrite(str(self._ambiguous_dir/f"{stem}.jpg"),crop);payload={"camera":packet.camera_id,"frame_id":packet.frame_id,"source_timestamp":packet.capture_timestamp,"a":{"id":a.detection_id,"bbox":a.bbox_xyxy,"confidence":a.confidence},"b":{"id":b.detection_id,"bbox":b.bbox_xyxy,"confidence":b.confidence},**geometry};(self._ambiguous_dir/f"{stem}.json").write_text(json.dumps(payload,indent=2));self._ambiguous_crops_saved+=1;self._ambiguous_last_saved[packet.camera_id]=packet.capture_timestamp
        except Exception as exc:log.warning("Ambiguous detection crop save failed: %s",exc)

    def _recover_roi(self,batch,camera_results):
        selected=self.roi.select(batch.frames,camera_results)
        if selected is None:return camera_results
        packet,roi,_urgent=selected;x1,y1,x2,y2=crop_rectangle(roi,packet.width,packet.height)
        if x2<=x1 or y2<=y1:return camera_results
        crop=packet.frame[y1:y2,x1:x2];roi_packet=FramePacket(packet.camera_id,packet.frame_id,packet.capture_timestamp,packet.receive_timestamp,crop,x2-x1,y2-y1,packet.scheduler_selected_timestamp,packet.capture_monotonic);roi_batch=BatchOutput(batch.batch_id,batch.created_timestamp,(roi_packet,));started=time.perf_counter();prepared=self.preprocessor.prepare(roi_batch)
        with gpu_coordinator.roi("ROI_RECOVERY"):raw_results,_timing=self.backend.infer(prepared)
        detections=[];polygon=source_polygon(roi,packet.width,packet.height)
        for row in raw_results[0]:
            if len(row)<6 or int(row[5])!=0:continue
            bbox=tuple(float(v) for v in row[:4])
            if not getattr(self.backend,"coordinates_original",False):bbox=original_bbox(bbox,prepared.transforms[0])
            bbox=map_crop_bbox(bbox,(x1,y1));width,height,confidence=bbox[2]-bbox[0],bbox[3]-bbox[1],float(row[4])
            if width<self.min_w or height<self.min_h or confidence<self.low_conf and (width<self.low_w or height<self.low_h):continue
            if not point_in_polygon(bbox_anchor(bbox),polygon):continue
            ratio=width/max(height,1)
            if .08<=ratio<=8:detections.append(Detection(bbox,confidence,detection_source="ROI_RECOVERY",detection_id=f"{packet.camera_id}:{packet.frame_id}:ROI:{len(detections)}"))
        output=[];recovered=duplicates=0
        for camera in camera_results:
            if camera.camera_id!=packet.camera_id:output.append(camera);continue
            fused=fuse_detections(camera.detections,detections);duplicates=max(0,len(camera.detections)+len(detections)-len(fused));recovered=max(0,len(fused)-len(camera.detections));output.append(CameraDetectionResult(camera.camera_id,camera.frame_id,camera.capture_timestamp,camera.receive_timestamp,fused,camera.capture_monotonic,camera.source_width,camera.source_height))
        self.roi.record((time.perf_counter()-started)*1000,recovered,duplicates);return tuple(output)

    def runtime_snapshot(self):
        with self._runtime_lock:item={"detector_batches_started":self._batches_started,"detector_batches_completed":self._batches_completed,"detector_batches_pending":self._batches_pending,"max_detector_batches_pending":self._max_batches_pending,"raw_ambiguous_person_pairs":self._raw_ambiguous_pairs,"software_person_duplicates_suppressed":self._software_duplicates_suppressed,"ambiguous_ground_truth_crops_saved":self._ambiguous_crops_saved}
        backend=getattr(self.backend,"runtime_snapshot",None);return {**item,**(backend() if backend else {"gpu_inflight_batches":0,"max_gpu_inflight_batches":0})}
    def close(self):
        close = getattr(self.backend, "close", None)
        if close: close()
