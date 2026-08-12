from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .runtime_metrics import process_metrics

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    camera_id: str
    frame_id: int
    frame_captured_monotonic: float
    produced_monotonic: float
    boxes: tuple[PersonBox, ...]


class LatestDetectionStore:
    """Exactly one newest detector result per camera."""
    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, DetectionResult] = {}

    def put(self, result: DetectionResult) -> None:
        with self._lock:
            previous = self._results.get(result.camera_id)
            if previous is None or result.frame_id >= previous.frame_id:
                self._results[result.camera_id] = result

    def get(self, camera_id: str) -> DetectionResult | None:
        with self._lock:
            return self._results.get(camera_id)

    def snapshot(self):
        with self._lock:
            return dict(self._results)


def _normalize_imgsz(value, default=(448, 704)):
    value = value if value is not None else default
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return int(value)


def _predict_kwargs(config: dict):
    return {
        "imgsz": _normalize_imgsz(config.get("imgsz", [448, 704])),
        "conf": float(config.get("conf", 0.06)),
        "iou": float(config.get("iou", 0.50)),
        "classes": [0],
        "max_det": max(1, int(config.get("max_det", 50))),
        "device": str(config.get("device", "cuda:0")),
        "quantize": config.get("quantize", 32),
        "verbose": False,
    }


def _roi_predict_kwargs(config: dict, base_kwargs: dict):
    roi_cfg = dict(config.get("roi_second_pass") or {})
    kwargs = dict(base_kwargs)
    kwargs.update({
        "imgsz": _normalize_imgsz(roi_cfg.get("imgsz", [512, 640]), (512, 640)),
        "conf": float(roi_cfg.get("conf", max(0.03, float(base_kwargs["conf"]) * 0.75))),
        "iou": float(roi_cfg.get("iou", base_kwargs["iou"])),
        "max_det": max(1, int(roi_cfg.get("max_det", 20))),
    })
    return kwargs


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a, b):
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b):
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a, b):
    acx = (a[0] + a[2]) * 0.5; acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5; bcy = (b[1] + b[3]) * 0.5
    scale = max(20.0, max(_area(a), _area(b)) ** 0.5)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def _box_center_inside(box, bounds):
    x1, y1, x2, y2 = bounds
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _deduplicate_boxes(boxes, iou_threshold: float, containment_threshold: float = 0.84, center_threshold: float = 0.40):
    ordered = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    for candidate in ordered:
        if any(
            _iou(candidate, existing) >= iou_threshold
            or (_containment(candidate, existing) >= containment_threshold and _center_distance(candidate, existing) <= center_threshold)
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _raw_prediction_boxes(prediction):
    pred_boxes = getattr(prediction, "boxes", None)
    if pred_boxes is None or not len(pred_boxes):
        return []
    xyxy = pred_boxes.xyxy.detach().cpu().tolist()
    confs = pred_boxes.conf.detach().cpu().tolist()
    return [(float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(conf)) for c, conf in zip(xyxy, confs)]


def _map_full_boxes(prediction, source_w: int, source_h: int, resized_shape):
    resized_h, resized_w = resized_shape
    sx = float(source_w) / max(1.0, float(resized_w))
    sy = float(source_h) / max(1.0, float(resized_h))
    return [(x1*sx, y1*sy, x2*sx, y2*sy, conf) for x1,y1,x2,y2,conf in _raw_prediction_boxes(prediction)]


def _inverse_rotate_box(box, rotation: int, original_w: int, original_h: int):
    x1, y1, x2, y2, conf = box
    rotation = int(rotation) % 360
    if rotation == 0:
        return box
    if rotation == 90:
        ox1, oy1, ox2, oy2 = y1, original_h-x2, y2, original_h-x1
    elif rotation == 270:
        ox1, oy1, ox2, oy2 = original_w-y2, x1, original_w-y1, x2
    elif rotation == 180:
        ox1, oy1, ox2, oy2 = original_w-x2, original_h-y2, original_w-x1, original_h-y1
    else:
        raise ValueError(f"unsupported ROI rotation: {rotation}")
    return (max(0.0,min(float(original_w),ox1)),max(0.0,min(float(original_h),oy1)),max(0.0,min(float(original_w),ox2)),max(0.0,min(float(original_h),oy2)),conf)


def _map_roi_boxes(prediction, roi: dict, rotation: int):
    rx1, ry1, rx2, ry2 = roi["bounds"]
    resized_h, resized_w = roi["shape"]
    sx = max(1.0, float(rx2-rx1)) / max(1.0, float(resized_w))
    sy = max(1.0, float(ry2-ry1)) / max(1.0, float(resized_h))
    mapped=[]
    for raw in _raw_prediction_boxes(prediction):
        x1,y1,x2,y2,conf=_inverse_rotate_box(raw,rotation,resized_w,resized_h)
        mapped.append((float(rx1)+x1*sx,float(ry1)+y1*sy,float(rx1)+x2*sx,float(ry1)+y2*sy,conf))
    return mapped


def _rotate_image(image, rotation: int):
    import cv2
    rotation = int(rotation) % 360
    if rotation == 0:return image
    if rotation == 90:return cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    if rotation == 270:return cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:return cv2.rotate(image,cv2.ROTATE_180)
    raise ValueError(f"unsupported ROI rotation: {rotation}")


def _normalized_zone_to_pixels(zone, source_w, source_h):
    return (float(zone[0])*source_w,float(zone[1])*source_h,float(zone[2])*source_w,float(zone[3])*source_h)


def _hard_reject(box, source_w:int, source_h:int, zones, max_box_height_ratio:float, overlap_threshold:float):
    if not zones:return False
    box_h=max(0.0,box[3]-box[1])/max(1.0,float(source_h))
    if box_h>max_box_height_ratio:return False
    box_area=max(1.0,_area(box));cx=(box[0]+box[2])*0.5;cy=(box[1]+box[3])*0.5
    for zone in zones:
        if len(zone)<4:continue
        z=_normalized_zone_to_pixels(zone,source_w,source_h)
        if z[0]<=cx<=z[2] and z[1]<=cy<=z[3]:return True
        if _intersection(box,z)/box_area>=overlap_threshold:return True
    return False


def _filter_hard_masks(camera_id, boxes, source_w, source_h, hard_cfg):
    camera_zones=dict(hard_cfg.get("cameras") or {}).get(camera_id,[])
    if not camera_zones:return boxes,0
    max_h=float(hard_cfg.get("max_box_height_ratio",0.30))
    overlap=float(hard_cfg.get("overlap_threshold",0.15))
    kept=[];rejected=0
    for box in boxes:
        if _hard_reject(box,source_w,source_h,camera_zones,max_h,overlap):rejected+=1
        else:kept.append(box)
    return kept,rejected


def _stats(values):
    values=list(values)
    if not values:return {"count":0,"p50":0.0,"p95":0.0,"max":0.0}
    ordered=sorted(values)
    def pct(p):return ordered[min(len(ordered)-1,int((len(ordered)-1)*p))]
    return {"count":len(ordered),"p50":pct(.50),"p95":pct(.95),"max":ordered[-1]}


def _detector_process_main(input_queue, output_queue, config:dict, model_path:str):
    import faulthandler
    faulthandler.enable(all_threads=True)
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
        try:
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
        except Exception:pass
        kwargs=_predict_kwargs(config);roi_kwargs=_roi_predict_kwargs(config,kwargs)
        device=str(kwargs["device"])
        if device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        model=YOLO(model_path)
        imgsz=kwargs["imgsz"];warm_h,warm_w=(int(imgsz[0]),int(imgsz[1])) if isinstance(imgsz,tuple) else (int(imgsz),int(imgsz))
        model.predict(source=[np.zeros((warm_h,warm_w,3),dtype=np.uint8)],**kwargs)
        output_queue.put(("ready",{"device":device,"model":model_path}))
    except BaseException as exc:
        try:output_queue.put(("startup_error",f"{type(exc).__name__}: {exc}"))
        except Exception:pass
        return

    duplicate_iou=float(config.get("duplicate_iou",0.58));fusion_containment=float(config.get("fusion_containment",0.84));fusion_center_distance=float(config.get("fusion_center_distance",0.40))
    hard_cfg=dict(config.get("hard_exclusion") or {})
    roi_cfg=dict(config.get("roi_second_pass") or {});roi_enabled=bool(roi_cfg.get("enabled",False));global_trigger_max=max(0,int(roi_cfg.get("trigger_max_full_roi_persons",1)))

    while True:
        try:payload=input_queue.get(timeout=0.25)
        except queue.Empty:continue
        except (EOFError,OSError):return
        if payload is None:return
        batch_id,entries=payload;started=time.perf_counter()
        try:full_predictions=model.predict(source=[entry["full_image"] for entry in entries],**kwargs)
        except BaseException as exc:
            try:output_queue.put(("batch_error",{"batch_id":batch_id,"error":f"{type(exc).__name__}: {exc}"}))
            except Exception:pass
            continue

        boxes_by_camera={};hard_rejects=0
        for entry,prediction in zip(entries,full_predictions):
            cid=entry["camera_id"]
            boxes=_map_full_boxes(prediction,entry["source_w"],entry["source_h"],entry["full_shape"])
            boxes,rejected=_filter_hard_masks(cid,boxes,entry["source_w"],entry["source_h"],hard_cfg)
            boxes_by_camera[cid]=boxes;hard_rejects+=rejected

        roi_jobs=[]
        if roi_enabled:
            for entry in entries:
                roi=entry.get("roi")
                if not roi:continue
                full_inside=sum(1 for box in boxes_by_camera.get(entry["camera_id"],()) if _box_center_inside(box,roi["bounds"]))
                mode=str(roi.get("mode","augment")).lower();trigger_max=int(roi.get("trigger_max_full_roi_persons",global_trigger_max))
                should_run=bool(roi.get("always_run",False)) or mode=="verify" or full_inside<=trigger_max
                if not should_run:continue
                for rotation in roi.get("rotations",[0]):
                    rotation=int(rotation)%360;roi_jobs.append((entry,roi,rotation,_rotate_image(roi["image"],rotation)))

        roi_wall_ms=0.0
        if roi_jobs:
            roi_started=time.perf_counter()
            try:
                roi_predictions=model.predict(source=[job[3] for job in roi_jobs],**roi_kwargs);grouped={}
                for (entry,roi,rotation,_image),prediction in zip(roi_jobs,roi_predictions):
                    accept_conf=float(roi.get("accept_conf",roi_cfg.get("accept_conf",roi_kwargs["conf"])))
                    mapped=[box for box in _map_roi_boxes(prediction,roi,rotation) if box[4]>=accept_conf]
                    mapped,rejected=_filter_hard_masks(entry["camera_id"],mapped,entry["source_w"],entry["source_h"],hard_cfg);hard_rejects+=rejected
                    grouped.setdefault(entry["camera_id"],[]).extend(mapped)
                for entry in entries:
                    roi=entry.get("roi")
                    if not roi:continue
                    cid=entry["camera_id"];mode=str(roi.get("mode","augment")).lower();roi_boxes=grouped.get(cid,[])
                    if mode=="verify":
                        outside=[box for box in boxes_by_camera.get(cid,[]) if not _box_center_inside(box,roi["bounds"])]
                        boxes_by_camera[cid]=outside+roi_boxes
                    else:boxes_by_camera.setdefault(cid,[]).extend(roi_boxes)
            except BaseException as exc:
                try:output_queue.put(("roi_error",{"batch_id":batch_id,"error":f"{type(exc).__name__}: {exc}"}))
                except Exception:pass
            roi_wall_ms=(time.perf_counter()-roi_started)*1000.0

        produced=time.monotonic();result_entries=[];total_boxes=0
        for entry in entries:
            boxes=_deduplicate_boxes(boxes_by_camera.get(entry["camera_id"],[]),duplicate_iou,fusion_containment,fusion_center_distance)
            total_boxes+=len(boxes);result_entries.append((entry["camera_id"],entry["frame_id"],entry["captured_mono"],produced,boxes))
        wall_ms=(time.perf_counter()-started)*1000.0
        try:
            output_queue.put(("result",{"batch_id":batch_id,"wall_ms":wall_ms,"roi_wall_ms":roi_wall_ms,"roi_inputs":len({job[0]["camera_id"] for job in roi_jobs}),"roi_variants":len(roi_jobs),"hard_rejects":hard_rejects,"inputs":len(entries),"detections":total_boxes,"entries":result_entries}))
        except (BrokenPipeError,EOFError,OSError):return


class YoloDetectorWorker:
    """Latest-only detector bridge with one in-flight CUDA batch and freshness gates."""
    def __init__(self,frame_stores,config:dict,project_root:Path):
        self.frame_stores=dict(frame_stores);self.config=dict(config);self.project_root=Path(project_root);self.camera_ids=sorted(self.frame_stores)
        self.batch_size=max(1,min(len(self.camera_ids) or 1,int(self.config.get("batch_size",3))));self.poll_interval=max(0.002,float(self.config.get("poll_interval_ms",5.0))/1000.0)
        self.max_submit_age_ms=max(0.0,float(self.config.get("max_submit_age_ms",350.0)));self.max_result_age_ms=max(0.0,float(self.config.get("max_result_age_ms",650.0)))
        raw_imgsz=_normalize_imgsz(self.config.get("imgsz",[448,704]));self.input_h,self.input_w=(int(raw_imgsz[0]),int(raw_imgsz[1])) if isinstance(raw_imgsz,tuple) else (int(raw_imgsz),int(raw_imgsz))
        self.roi_cfg=dict(self.config.get("roi_second_pass") or {});raw_roi=_normalize_imgsz(self.roi_cfg.get("imgsz",[512,640]),(512,640));self.roi_h,self.roi_w=(int(raw_roi[0]),int(raw_roi[1])) if isinstance(raw_roi,tuple) else (int(raw_roi),int(raw_roi))
        self.roi_cameras=dict(self.roi_cfg.get("cameras") or {});self.roi_every_n=max(1,int(self.roi_cfg.get("every_n",2)));self._roi_seen={cid:0 for cid in self.camera_ids}
        model_value=str(self.config.get("model","models/yolo26m.pt"));model_path=Path(model_value).expanduser();self.model_path=model_path if model_path.is_absolute() else self.project_root/model_path
        self.start_delay_sec=max(0.0,float(self.config.get("start_delay_sec",2.0)));self.results=LatestDetectionStore();self._stop=threading.Event();self._thread=None;self._process=None;self._ctx=mp.get_context("spawn");self._input_queue=None;self._output_queue=None
        self._last_versions={cid:0 for cid in self.camera_ids};self._cursor=0;self._batch_id=0;self._inflight_batch_id=None;self._lock=threading.Lock();self._started_mono=time.monotonic();self._ready=False
        self._submitted=0;self._batches=0;self._inputs=0;self._detections=0;self._roi_inputs=0;self._roi_variants=0;self._hard_rejects=0;self._stale_submit_drops=0;self._stale_result_drops=0;self._queue_full_drops=0;self._skipped_latest=0
        self._last_batch_ms=0.0;self._last_roi_ms=0.0;self._last_prepare_ms=0.0;self._last_submit_age_ms=0.0;self._last_finish_age_ms=0.0;self._last_error=""
        self._batch_ms=deque(maxlen=300);self._finish_age_ms=deque(maxlen=300);self._submit_age_ms=deque(maxlen=300)
        self._per_camera_inputs={cid:0 for cid in self.camera_ids};self._per_camera_last_frame_id={cid:0 for cid in self.camera_ids};self._per_camera_last_detection_mono={cid:0.0 for cid in self.camera_ids};self._per_camera_finish_age_ms={cid:None for cid in self.camera_ids}

    def start(self):
        if self._thread and self._thread.is_alive():return
        self._stop.clear();self._thread=threading.Thread(target=self._run_bridge,name="core-v1-yolo-bridge",daemon=False);self._thread.start()
    def stop(self):
        self._stop.set()
        if self._input_queue is not None:
            try:self._input_queue.put_nowait(None)
            except Exception:pass
    def join(self,timeout=10):
        deadline=time.monotonic()+timeout
        if self._thread:self._thread.join(max(0.0,deadline-time.monotonic()))
        process=self._process
        if process is not None and process.is_alive():
            process.join(max(0.0,deadline-time.monotonic()))
            if process.is_alive():process.terminate();process.join(2.0)
        return (not self._thread or not self._thread.is_alive()) and (process is None or not process.is_alive())
    def _spawn_process(self):
        if not self.model_path.exists():raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self._input_queue=self._ctx.Queue(maxsize=1);self._output_queue=self._ctx.Queue(maxsize=16);self._process=self._ctx.Process(target=_detector_process_main,name="core-v1-yolo-cuda",args=(self._input_queue,self._output_queue,self.config,str(self.model_path)),daemon=False);self._process.start();log.info("CORE_V1_YOLO_PROCESS_STARTED pid=%s start_method=spawn",self._process.pid)

    def _select_latest_batch(self):
        if not self.camera_ids:return []
        selected=[];n=len(self.camera_ids);scanned=0;now=time.monotonic()
        while scanned<n and len(selected)<self.batch_size:
            cid=self.camera_ids[(self._cursor+scanned)%n];frame,version=self.frame_stores[cid].get()
            if frame is not None and version>self._last_versions[cid]:
                if self._last_versions[cid]>0 and version>self._last_versions[cid]+1:self._skipped_latest+=version-self._last_versions[cid]-1
                age_ms=max(0.0,(now-float(frame.captured_monotonic))*1000.0)
                if self.max_submit_age_ms>0 and age_ms>self.max_submit_age_ms:
                    self._last_versions[cid]=version;self._stale_submit_drops+=1
                else:selected.append((cid,frame,version,age_ms))
            scanned+=1
        self._cursor=(self._cursor+max(1,scanned))%n
        return selected

    def _prepare_roi(self,cid,frame):
        camera_cfg=dict(self.roi_cameras.get(cid) or {})
        if not camera_cfg or not bool(self.roi_cfg.get("enabled",False)):return None
        self._roi_seen[cid]+=1;every_n=max(1,int(camera_cfg.get("every_n",self.roi_every_n)))
        if self._roi_seen[cid]%every_n:return None
        import cv2
        values=camera_cfg.get("box") or camera_cfg.get("roi")
        if not isinstance(values,(list,tuple)) or len(values)<4:return None
        nx1,ny1,nx2,ny2=[max(0.0,min(1.0,float(v))) for v in values[:4]]
        if nx2<=nx1 or ny2<=ny1:return None
        x1=max(0,min(frame.width-1,int(round(nx1*frame.width))));y1=max(0,min(frame.height-1,int(round(ny1*frame.height))));x2=max(x1+1,min(frame.width,int(round(nx2*frame.width))));y2=max(y1+1,min(frame.height,int(round(ny2*frame.height))))
        crop=frame.image[y1:y2,x1:x2]
        if crop.size==0:return None
        resized=cv2.resize(crop,(self.roi_w,self.roi_h),interpolation=cv2.INTER_LINEAR);rotations=camera_cfg.get("rotations",[0]);rotations=rotations if isinstance(rotations,(list,tuple)) else [rotations];rotations=[int(v)%360 for v in rotations if int(v)%360 in {0,90,180,270}] or [0]
        return {"image":resized,"bounds":(x1,y1,x2,y2),"shape":(self.roi_h,self.roi_w),"mode":str(camera_cfg.get("mode","augment")).lower(),"always_run":bool(camera_cfg.get("always_run",False)),"rotations":rotations,"accept_conf":float(camera_cfg.get("accept_conf",self.roi_cfg.get("accept_conf",self.roi_cfg.get("conf",0.045)))),"trigger_max_full_roi_persons":int(camera_cfg.get("trigger_max_full_roi_persons",self.roi_cfg.get("trigger_max_full_roi_persons",1)))}

    def _prepare_payload(self,selected):
        import cv2
        started=time.perf_counter();entries=[]
        for cid,frame,_version,age_ms in selected:
            full=cv2.resize(frame.image,(self.input_w,self.input_h),interpolation=cv2.INTER_LINEAR);entries.append({"camera_id":cid,"frame_id":int(frame.frame_id),"captured_mono":float(frame.captured_monotonic),"source_w":int(frame.width),"source_h":int(frame.height),"full_shape":(self.input_h,self.input_w),"full_image":full,"roi":self._prepare_roi(cid,frame)});self._last_submit_age_ms=age_ms;self._submit_age_ms.append(age_ms)
        with self._lock:self._last_prepare_ms=(time.perf_counter()-started)*1000.0
        return entries

    def _submit_if_idle(self):
        if self._inflight_batch_id is not None:return
        selected=self._select_latest_batch()
        if not selected:return
        entries=self._prepare_payload(selected);self._batch_id+=1;batch_id=self._batch_id
        try:self._input_queue.put_nowait((batch_id,entries))
        except queue.Full:
            self._queue_full_drops+=len(entries);return
        for cid,_frame,version,_age in selected:self._last_versions[cid]=version
        self._inflight_batch_id=batch_id
        with self._lock:self._submitted+=len(entries)

    def _drain_outputs(self):
        if self._output_queue is None:return
        while True:
            try:kind,payload=self._output_queue.get_nowait()
            except queue.Empty:return
            except (EOFError,OSError):return
            if kind=="ready":
                with self._lock:self._ready=True;self._last_error=""
                log.info("CORE_V1_YOLO_READY process_pid=%s device=%s model=%s",self._process.pid if self._process else None,payload.get("device"),payload.get("model"));continue
            if kind in {"startup_error","batch_error","roi_error"}:
                error=payload if isinstance(payload,str) else payload.get("error",str(payload))
                if kind!="roi_error" and isinstance(payload,dict) and payload.get("batch_id")==self._inflight_batch_id:self._inflight_batch_id=None
                with self._lock:self._last_error=error
                log.error("CORE_V1_YOLO_%s %s",kind.upper(),error);continue
            if kind!="result":continue
            if payload.get("batch_id")==self._inflight_batch_id:self._inflight_batch_id=None
            now=time.monotonic()
            for cid,frame_id,captured_mono,produced_mono,raw_boxes in payload["entries"]:
                finish_age_ms=max(0.0,(now-float(captured_mono))*1000.0);self._last_finish_age_ms=finish_age_ms;self._finish_age_ms.append(finish_age_ms);self._per_camera_finish_age_ms[str(cid)]=finish_age_ms
                if self.max_result_age_ms>0 and finish_age_ms>self.max_result_age_ms:
                    self._stale_result_drops+=1;continue
                boxes=tuple(PersonBox(*map(float,box)) for box in raw_boxes);self.results.put(DetectionResult(camera_id=str(cid),frame_id=int(frame_id),frame_captured_monotonic=float(captured_mono),produced_monotonic=float(produced_mono),boxes=boxes))
                with self._lock:self._per_camera_inputs[str(cid)]+=1;self._per_camera_last_frame_id[str(cid)]=int(frame_id);self._per_camera_last_detection_mono[str(cid)]=float(produced_mono)
            with self._lock:
                self._batches+=1;self._inputs+=int(payload["inputs"]);self._detections+=int(payload["detections"]);self._roi_inputs+=int(payload.get("roi_inputs",0));self._roi_variants+=int(payload.get("roi_variants",0));self._hard_rejects+=int(payload.get("hard_rejects",0));self._last_batch_ms=float(payload["wall_ms"]);self._last_roi_ms=float(payload.get("roi_wall_ms",0.0));self._batch_ms.append(self._last_batch_ms);self._last_error=""

    def _run_bridge(self):
        if self.start_delay_sec and self._stop.wait(self.start_delay_sec):return
        try:self._spawn_process()
        except Exception as exc:
            with self._lock:self._last_error=f"{type(exc).__name__}: {exc}"
            log.exception("CORE_V1_YOLO_PROCESS_START_FAILED");return
        while not self._stop.is_set():
            self._drain_outputs();process=self._process
            if process is not None and not process.is_alive():
                exitcode=process.exitcode
                with self._lock:
                    self._ready=False
                    if not self._last_error:self._last_error=f"detector process exited unexpectedly with exitcode={exitcode}"
                log.error("CORE_V1_YOLO_PROCESS_EXITED exitcode=%s camera_core_continues=true",exitcode);break
            if self._ready:self._submit_if_idle()
            self._stop.wait(self.poll_interval)
        self._drain_outputs()

    def metrics(self):
        now=time.monotonic();process=self._process
        with self._lock:
            elapsed=max(0.001,now-self._started_mono);cameras={}
            for cid in self.camera_ids:
                last=self._per_camera_last_detection_mono[cid];cameras[cid]={"inputs":self._per_camera_inputs[cid],"input_rate":self._per_camera_inputs[cid]/elapsed,"last_frame_id":self._per_camera_last_frame_id[cid],"observation_age_ms":((now-last)*1000.0) if last else None,"last_finish_age_ms":self._per_camera_finish_age_ms[cid]}
            resource=process_metrics(process.pid if process and process.is_alive() else None)
            return {"ready":self._ready,"process_alive":bool(process and process.is_alive()),"process_pid":process.pid if process else None,"process_exitcode":process.exitcode if process and not process.is_alive() else None,"start_method":"spawn","model":str(self.model_path),"device":str(self.config.get("device","cuda:0")),"quantize":self.config.get("quantize",32),"batch_size":self.batch_size,"inflight_batch_id":self._inflight_batch_id,"submitted_camera_inputs":self._submitted,"batches":self._batches,"batch_rate":self._batches/elapsed,"camera_inputs":self._inputs,"camera_input_rate":self._inputs/elapsed,"detections":self._detections,"hard_rejects":self._hard_rejects,"roi_inputs":self._roi_inputs,"roi_variants":self._roi_variants,"roi_rate":self._roi_inputs/elapsed,"stale_submit_drops":self._stale_submit_drops,"stale_result_drops":self._stale_result_drops,"queue_full_drops":self._queue_full_drops,"skipped_latest":self._skipped_latest,"last_prepare_ms":self._last_prepare_ms,"last_batch_ms":self._last_batch_ms,"batch_ms":_stats(self._batch_ms),"last_roi_ms":self._last_roi_ms,"last_submit_age_ms":self._last_submit_age_ms,"submit_age_ms":_stats(self._submit_age_ms),"last_finish_age_ms":self._last_finish_age_ms,"finish_age_ms":_stats(self._finish_age_ms),"resources":resource,"last_error":self._last_error,"cameras":cameras}
