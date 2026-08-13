from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import queue
import threading
import time

import cv2
import numpy as np

from .crop_selector import PersonCropSelector
from .global_identity import GlobalIdentityManager

log = logging.getLogger(__name__)


@dataclass
class _LocalTrack:
    local_id: int
    box: object
    last_seen: float
    hits: int = 1
    last_embed_at: float = 0.0
    best_quality: float = 0.0
    global_id: str | None = None
    last_similarity: float | None = None


@dataclass(frozen=True)
class _ReIDJob:
    camera_id: str
    local_id: int
    frame_id: int
    observed_at: float
    quality: float
    crop: object


def _iou(a, b):
    x1=max(float(a.x1),float(b.x1));y1=max(float(a.y1),float(b.y1));x2=min(float(a.x2),float(b.x2));y2=min(float(a.y2),float(b.y2))
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,float(a.x2)-float(a.x1))*max(0.0,float(a.y2)-float(a.y1));bb=max(0.0,float(b.x2)-float(b.x1))*max(0.0,float(b.y2)-float(b.y1))
    union=aa+bb-inter
    return inter/union if union>0 else 0.0


def _center_distance(a,b):
    acx=(float(a.x1)+float(a.x2))*0.5;acy=(float(a.y1)+float(a.y2))*0.5;bcx=(float(b.x1)+float(b.x2))*0.5;bcy=(float(b.y1)+float(b.y2))*0.5
    aw=max(1.0,float(a.x2)-float(a.x1));ah=max(1.0,float(a.y2)-float(a.y1));bw=max(1.0,float(b.x2)-float(b.x1));bh=max(1.0,float(b.y2)-float(b.y1))
    return math.hypot(acx-bcx,acy-bcy)/max(20.0,aw,ah,bw,bh)


class _OSNetExtractor:
    def __init__(self, config: dict):
        self.config=dict(config);self.device=str(self.config.get("device","cpu"));self.model_name=str(self.config.get("model","osnet_x0_25"));self.model_path=str(self.config.get("model_path","") or "");self.height=max(64,int(self.config.get("input_height",256)));self.width=max(32,int(self.config.get("input_width",128)))
        import torch
        import torchreid
        self.torch=torch
        use_cuda=self.device.startswith("cuda") and torch.cuda.is_available()
        if self.device.startswith("cuda") and not use_cuda:raise RuntimeError("ReID CUDA requested but unavailable")
        model=torchreid.models.build_model(name=self.model_name,num_classes=1000,loss="softmax",pretrained=not bool(self.model_path),use_gpu=use_cuda)
        if self.model_path:
            torchreid.utils.load_pretrained_weights(model,self.model_path)
        model.eval();model.to(self.device);self.model=model
        self.mean=np.asarray([0.485,0.456,0.406],dtype=np.float32).reshape(3,1,1);self.std=np.asarray([0.229,0.224,0.225],dtype=np.float32).reshape(3,1,1)

    def _tensor(self,crop):
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(self.width,self.height),interpolation=cv2.INTER_LINEAR);array=rgb.astype(np.float32).transpose(2,0,1)/255.0;array=(array-self.mean)/self.std
        return self.torch.from_numpy(array)

    def extract(self,crops):
        batch=self.torch.stack([self._tensor(c) for c in crops],dim=0).to(self.device,non_blocking=False)
        with self.torch.inference_mode():features=self.model(batch)
        if isinstance(features,(tuple,list)):features=features[0]
        features=features.float();features=features/(features.norm(dim=1,keepdim=True)+1e-12)
        return features.detach().cpu().numpy()


class ReIDCoordinator:
    """Non-blocking, event-driven ReID attached beside the camera hot path."""
    def __init__(self, frame_stores, detections, config: dict | None = None):
        self.frame_stores=dict(frame_stores);self.detections=detections;self.config=dict(config or {});self.enabled=bool(self.config.get("enabled",False));self.selector=PersonCropSelector(self.config.get("crop"));self.identities=GlobalIdentityManager(self.config.get("identity"))
        self.poll_sec=max(0.01,float(self.config.get("poll_interval_ms",35))/1000.0);self.track_timeout=max(0.5,float(self.config.get("local_track_timeout_sec",2.0)));self.match_iou=max(0.0,float(self.config.get("local_match_iou",0.12)));self.match_center=max(0.1,float(self.config.get("local_match_center",0.72)));self.min_hits=max(1,int(self.config.get("min_track_hits",2)));self.embed_cooldown=max(0.2,float(self.config.get("embed_cooldown_sec",1.2)));self.refresh_sec=max(self.embed_cooldown,float(self.config.get("refresh_sec",4.0)));self.quality_improvement=max(0.0,float(self.config.get("quality_improvement",0.08)));self.batch_size=max(1,int(self.config.get("batch_size",4)));self.batch_wait=max(0.0,float(self.config.get("batch_wait_ms",20))/1000.0);self.max_job_age=max(0.1,float(self.config.get("max_job_age_ms",700))/1000.0)
        self._lock=threading.Lock();self._stop=threading.Event();self._observer=None;self._infer=None;self._job_queue=queue.Queue(maxsize=max(1,int(self.config.get("queue_size",8))));self._latest_jobs={};self._last_result_frame={};self._tracks={cid:{} for cid in self.frame_stores};self._next_local={cid:1 for cid in self.frame_stores};self._extractor=None;self._ready=False;self._last_error="";self._started=time.monotonic();self._submitted=0;self._embedded=0;self._replaced_jobs=0;self._stale_jobs=0;self._crop_rejects=0;self._frame_misses=0;self._batches=0;self._last_batch_ms=0.0

    def start(self):
        if not self.enabled:return
        self._stop.clear();self._observer=threading.Thread(target=self._observe_loop,name="core-v1-reid-observer",daemon=False);self._infer=threading.Thread(target=self._infer_loop,name="core-v1-reid-infer",daemon=False);self._observer.start();self._infer.start()

    def stop(self):
        self._stop.set()
        try:self._job_queue.put_nowait(None)
        except Exception:pass

    def join(self,timeout=5):
        deadline=time.monotonic()+timeout
        for thread in (self._observer,self._infer):
            if thread:thread.join(max(0.0,deadline-time.monotonic()))

    def _associate(self,camera_id,boxes,observed_at):
        tracks=self._tracks[camera_id]
        for tid in list(tracks):
            if observed_at-tracks[tid].last_seen>self.track_timeout:del tracks[tid]
        pairs=[]
        for tid,track in tracks.items():
            for di,box in enumerate(boxes):
                iou=_iou(track.box,box);dist=_center_distance(track.box,box)
                if iou>=self.match_iou or dist<=self.match_center:pairs.append(((1.0-iou)+0.25*dist,tid,di))
        pairs.sort();used_t=set();used_d=set();assigned=[]
        for _score,tid,di in pairs:
            if tid in used_t or di in used_d:continue
            track=tracks[tid];track.box=boxes[di];track.last_seen=observed_at;track.hits+=1;used_t.add(tid);used_d.add(di);assigned.append((track,boxes[di]))
        for di,box in enumerate(boxes):
            if di in used_d:continue
            tid=self._next_local[camera_id];self._next_local[camera_id]+=1;track=_LocalTrack(tid,box,observed_at);tracks[tid]=track;assigned.append((track,box))
        return assigned

    def _enqueue_latest(self,job:_ReIDJob):
        key=(job.camera_id,job.local_id)
        with self._lock:
            if key in self._latest_jobs:self._replaced_jobs+=1
            self._latest_jobs[key]=job
        try:self._job_queue.put_nowait(key);self._submitted+=1
        except queue.Full:
            # Key remains in the coalescing map. A newer wake-up for this track
            # will process the latest crop; no FIFO of stale image crops forms.
            self._replaced_jobs+=1

    def _observe_loop(self):
        while not self._stop.is_set():
            snapshot=self.detections.snapshot() if self.detections is not None else {}
            for camera_id,result in snapshot.items():
                if int(result.frame_id)<=int(self._last_result_frame.get(camera_id,-1)):continue
                self._last_result_frame[camera_id]=int(result.frame_id);store=self.frame_stores.get(camera_id);frame=store.get_frame(result.frame_id) if store and hasattr(store,"get_frame") else None
                if frame is None:self._frame_misses+=1;continue
                observed=float(result.frame_captured_monotonic);assigned=self._associate(camera_id,list(result.boxes),observed);now=time.monotonic()
                for track,box in assigned:
                    if track.hits<self.min_hits:continue
                    decision=self.selector.evaluate(frame,box)
                    if not decision.accepted:self._crop_rejects+=1;continue
                    due=now-track.last_embed_at>=self.refresh_sec;better=decision.score>=track.best_quality+self.quality_improvement
                    first=track.last_embed_at<=0
                    if not first and not due and not better:continue
                    if not first and now-track.last_embed_at<self.embed_cooldown:continue
                    track.last_embed_at=now;track.best_quality=max(track.best_quality,decision.score);self._enqueue_latest(_ReIDJob(camera_id,track.local_id,int(result.frame_id),observed,decision.score,decision.crop))
            self._stop.wait(self.poll_sec)

    def _pop_job(self,key):
        if key is None:return None
        with self._lock:return self._latest_jobs.pop(key,None)

    def _infer_loop(self):
        try:
            self._extractor=_OSNetExtractor(self.config);self._ready=True;log.info("CORE_V1_REID_READY model=%s device=%s",self.config.get("model","osnet_x0_25"),self.config.get("device","cpu"))
        except Exception as exc:
            self._last_error=f"{type(exc).__name__}: {exc}";log.exception("CORE_V1_REID_DISABLED error=%s",self._last_error);return
        while not self._stop.is_set():
            try:key=self._job_queue.get(timeout=.25)
            except queue.Empty:continue
            if key is None:break
            jobs=[];job=self._pop_job(key)
            if job is not None:jobs.append(job)
            deadline=time.monotonic()+self.batch_wait
            while len(jobs)<self.batch_size and time.monotonic()<deadline:
                try:key2=self._job_queue.get(timeout=max(0.0,deadline-time.monotonic()))
                except queue.Empty:break
                if key2 is None:self._stop.set();break
                job2=self._pop_job(key2)
                if job2 is not None:jobs.append(job2)
            now=time.monotonic();jobs=[j for j in jobs if now-j.observed_at<=self.max_job_age or not (self._stale_jobs:=self._stale_jobs+1)]
            if not jobs:continue
            started=time.perf_counter()
            try:features=self._extractor.extract([j.crop for j in jobs])
            except Exception as exc:self._last_error=f"{type(exc).__name__}: {exc}";log.exception("CORE_V1_REID_BATCH_FAILED");continue
            self._last_batch_ms=(time.perf_counter()-started)*1000.0;self._batches+=1
            for job,feature in zip(jobs,features):
                gid,similarity,_reason=self.identities.assign(camera_id=job.camera_id,local_track_id=job.local_id,embedding=feature,observed_at=job.observed_at);track=self._tracks.get(job.camera_id,{}).get(job.local_id)
                if track is not None:track.global_id=gid;track.last_similarity=float(similarity)
                self._embedded+=1

    def labels(self,camera_id:str):
        with self._lock:
            tracks=list(self._tracks.get(str(camera_id),{}).values())
        result=[]
        now=time.monotonic()
        for track in tracks:
            if track.global_id and now-track.last_seen<=self.track_timeout:result.append({"local_id":track.local_id,"global_id":track.global_id,"box":track.box,"similarity":track.last_similarity})
        return result

    def snapshot(self):
        cameras={}
        with self._lock:
            for cid,tracks in self._tracks.items():
                cameras[cid]=[{"local_id":t.local_id,"global_id":t.global_id,"last_seen":t.last_seen,"hits":t.hits,"similarity":t.last_similarity} for t in tracks.values()]
        return {"cameras":cameras,"global":self.identities.snapshot()}

    def metrics(self):
        elapsed=max(.001,time.monotonic()-self._started)
        return {"enabled":self.enabled,"ready":self._ready,"model":self.config.get("model","osnet_x0_25"),"device":self.config.get("device","cpu"),"submitted":self._submitted,"embedded":self._embedded,"embed_rate":self._embedded/elapsed,"batches":self._batches,"last_batch_ms":self._last_batch_ms,"replaced_jobs":self._replaced_jobs,"stale_jobs":self._stale_jobs,"crop_rejects":self._crop_rejects,"frame_misses":self._frame_misses,"queue_depth":self._job_queue.qsize(),"last_error":self._last_error,"identity":self.identities.metrics()}
