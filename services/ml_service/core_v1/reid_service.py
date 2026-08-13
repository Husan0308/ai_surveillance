from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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


def _norm(vector):
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _cosine(a, b) -> float:
    return float(np.dot(_norm(a), _norm(b)))


@dataclass
class _FeatureSample:
    embedding: np.ndarray
    quality: float
    observed_at: float


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
    last_reason: str | None = None
    samples: deque = field(default_factory=deque)
    descriptor: np.ndarray | None = None


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
        self.config=dict(config);self.device=str(self.config.get("device","cpu"));self.model_name=str(self.config.get("model","osnet_x0_5"));self.model_path=str(self.config.get("model_path","") or "");self.height=max(64,int(self.config.get("input_height",256)));self.width=max(32,int(self.config.get("input_width",128)))
        import torch, torchreid
        self.torch=torch;use_cuda=self.device.startswith("cuda") and torch.cuda.is_available()
        if self.device.startswith("cuda") and not use_cuda:raise RuntimeError("ReID CUDA requested but unavailable")
        try:torch.set_num_threads(max(1,int(self.config.get("torch_cpu_threads",1))))
        except Exception:pass
        model=torchreid.models.build_model(name=self.model_name,num_classes=1000,loss="softmax",pretrained=not bool(self.model_path),use_gpu=use_cuda)
        if self.model_path:torchreid.utils.load_pretrained_weights(model,self.model_path)
        model.eval();model.to(self.device);self.model=model
        self.mean=np.asarray([0.485,0.456,0.406],dtype=np.float32).reshape(3,1,1);self.std=np.asarray([0.229,0.224,0.225],dtype=np.float32).reshape(3,1,1)

    def _tensor(self,crop):
        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(self.width,self.height),interpolation=cv2.INTER_LINEAR);arr=rgb.astype(np.float32).transpose(2,0,1)/255.0;arr=(arr-self.mean)/self.std
        return self.torch.from_numpy(arr)

    def extract(self,crops):
        batch=self.torch.stack([self._tensor(c) for c in crops],dim=0).to(self.device,non_blocking=False)
        with self.torch.inference_mode():features=self.model(batch)
        if isinstance(features,(tuple,list)):features=features[0]
        features=features.float();features=features/(features.norm(dim=1,keepdim=True)+1e-12)
        return features.detach().cpu().numpy()


class ReIDCoordinator:
    """Tracklet ReID v2: asynchronous embeddings + multi-frame pair matching."""

    def __init__(self, frame_stores, detections, config: dict | None = None):
        self.frame_stores=dict(frame_stores);self.detections=detections;self.config=dict(config or {});self.enabled=bool(self.config.get("enabled",False));self.selector=PersonCropSelector(self.config.get("crop"));self.identities=GlobalIdentityManager(self.config.get("identity"))
        self.poll_sec=max(.01,float(self.config.get("poll_interval_ms",35))/1000.0);self.track_timeout=max(.5,float(self.config.get("local_track_timeout_sec",2.5)));self.match_iou=max(0.0,float(self.config.get("local_match_iou",.12)));self.match_center=max(.1,float(self.config.get("local_match_center",.72)));self.min_hits=max(1,int(self.config.get("min_track_hits",2)))
        self.embed_cooldown=max(.2,float(self.config.get("embed_cooldown_sec",.75)));self.refresh_sec=max(self.embed_cooldown,float(self.config.get("refresh_sec",3.0)));self.quality_improvement=max(0.0,float(self.config.get("quality_improvement",.05)));self.batch_size=max(1,int(self.config.get("batch_size",4)));self.batch_wait=max(0.0,float(self.config.get("batch_wait_ms",20))/1000.0);self.max_job_age=max(.1,float(self.config.get("max_job_age_ms",700))/1000.0)
        tracklet=dict(self.config.get("tracklet") or {});self.sample_capacity=max(3,int(tracklet.get("max_samples",6)));self.min_descriptor_samples=max(2,int(tracklet.get("min_samples",3)));self.topk=max(2,int(tracklet.get("topk",4)));self.duplicate_feature_cos=max(.90,min(.9999,float(tracklet.get("duplicate_feature_cos",.985))));self.duplicate_quality_gain=max(0.0,float(tracklet.get("duplicate_quality_gain",.04)))
        self.pair_cfg=dict(self.config.get("pair_matching") or {});self.same_room_pairs=[tuple(map(str,p[:2])) for p in self.pair_cfg.get("pairs",[]) if isinstance(p,(list,tuple)) and len(p)>=2];self.default_candidate=float(self.pair_cfg.get("candidate_threshold",.68));self.default_strong=float(self.pair_cfg.get("strong_threshold",.76));self.default_margin=max(0.0,float(self.pair_cfg.get("margin",.035)));self.confirm_hits=max(1,int(self.pair_cfg.get("confirm_hits",2)));self.strong_confirm_hits=max(1,int(self.pair_cfg.get("strong_confirm_hits",1)));self.evidence_ttl=max(.2,float(self.pair_cfg.get("evidence_ttl_sec",2.0)));self.pair_overrides=dict(self.pair_cfg.get("overrides") or {})
        self._lock=threading.RLock();self._stop=threading.Event();self._observer=None;self._infer=None;self._job_queue=queue.Queue(maxsize=max(1,int(self.config.get("queue_size",8))));self._latest_jobs={};self._last_result_frame={};self._tracks={cid:{} for cid in self.frame_stores};self._next_local={cid:1 for cid in self.frame_stores};self._evidence={};self._pair_scores=deque(maxlen=512);self._extractor=None;self._ready=False;self._last_error="";self._started=time.monotonic();self._submitted=0;self._embedded=0;self._batches=0;self._last_batch_ms=0.0;self._replaced_jobs=0;self._stale_jobs=0;self._crop_rejects=0;self._frame_misses=0;self._released_tracks=0;self._descriptor_updates=0;self._duplicate_features=0;self._pair_attempts=0;self._pair_confirms=0;self._pair_merges=0;self._pair_rejects=0

    def start(self):
        if not self.enabled:return
        self._stop.clear();self._observer=threading.Thread(target=self._observe_loop,name="core-v1-reid-observer",daemon=False);self._infer=threading.Thread(target=self._infer_loop,name="core-v1-reid-infer",daemon=False);self._observer.start();self._infer.start()
    def stop(self):
        self._stop.set()
        try:self._job_queue.put_nowait(None)
        except Exception:pass
    def join(self,timeout=6):
        deadline=time.monotonic()+timeout
        for thread in (self._observer,self._infer):
            if thread:thread.join(max(0.0,deadline-time.monotonic()))

    def _associate(self,camera_id,boxes,observed_at):
        tracks=self._tracks[camera_id]
        for tid in list(tracks):
            if observed_at-tracks[tid].last_seen>self.track_timeout:self.identities.release_track(camera_id,tid);del tracks[tid];self._released_tracks+=1
        pairs=[]
        for tid,track in tracks.items():
            for di,box in enumerate(boxes):
                iou=_iou(track.box,box);dist=_center_distance(track.box,box)
                if iou>=self.match_iou or dist<=self.match_center:pairs.append(((1.0-iou)+.25*dist,tid,di))
        pairs.sort();used_t=set();used_d=set();assigned=[]
        for _score,tid,di in pairs:
            if tid in used_t or di in used_d:continue
            track=tracks[tid];track.box=boxes[di];track.last_seen=observed_at;track.hits+=1;used_t.add(tid);used_d.add(di);assigned.append((track,boxes[di]))
        for di,box in enumerate(boxes):
            if di in used_d:continue
            tid=self._next_local[camera_id];self._next_local[camera_id]+=1;track=_LocalTrack(tid,box,observed_at,samples=deque(maxlen=self.sample_capacity));tracks[tid]=track;assigned.append((track,box))
        return assigned

    def _enqueue_latest(self,job):
        key=(job.camera_id,job.local_id)
        with self._lock:
            if key in self._latest_jobs:self._replaced_jobs+=1
            self._latest_jobs[key]=job
        try:self._job_queue.put_nowait(key);self._submitted+=1
        except queue.Full:self._replaced_jobs+=1

    def _observe_loop(self):
        while not self._stop.is_set():
            snapshot=self.detections.snapshot() if self.detections is not None else {}
            for camera_id,result in snapshot.items():
                if int(result.frame_id)<=int(self._last_result_frame.get(camera_id,-1)):continue
                self._last_result_frame[camera_id]=int(result.frame_id);store=self.frame_stores.get(camera_id);frame=store.get_frame(result.frame_id) if store and hasattr(store,"get_frame") else None
                if frame is None:self._frame_misses+=1;continue
                observed=float(result.frame_captured_monotonic);assigned=self._associate(camera_id,list(result.boxes),observed);now=time.monotonic()
                for track,box in assigned:
                    if track.global_id:self.identities.touch_track(camera_id,track.local_id,observed)
                    if track.hits<self.min_hits:continue
                    decision=self.selector.evaluate(frame,box)
                    if not decision.accepted:self._crop_rejects+=1;continue
                    first=len(track.samples)<self.min_descriptor_samples;due=now-track.last_embed_at>=self.refresh_sec;better=decision.score>=track.best_quality+self.quality_improvement
                    if not first and not due and not better:continue
                    if track.last_embed_at>0 and now-track.last_embed_at<self.embed_cooldown:continue
                    track.last_embed_at=now;track.best_quality=max(track.best_quality,decision.score);self._enqueue_latest(_ReIDJob(camera_id,track.local_id,int(result.frame_id),observed,decision.score,decision.crop))
            self._stop.wait(self.poll_sec)

    def _pop_job(self,key):
        with self._lock:return self._latest_jobs.pop(key,None) if key is not None else None

    def _update_descriptor(self,track,embedding,quality,observed_at):
        vector=_norm(embedding)
        if track.samples:
            nearest=max(_cosine(vector,s.embedding) for s in track.samples);best_q=max(s.quality for s in track.samples)
            if nearest>=self.duplicate_feature_cos and quality<best_q+self.duplicate_quality_gain:self._duplicate_features+=1;return False
        track.samples.append(_FeatureSample(vector,float(quality),float(observed_at)))
        ranked=sorted(track.samples,key=lambda s:s.quality,reverse=True)[:min(self.topk,len(track.samples))];weights=np.asarray([max(.05,s.quality) for s in ranked],dtype=np.float32);weights=weights/weights.sum();track.descriptor=_norm(sum(float(w)*s.embedding for w,s in zip(weights,ranked)));self._descriptor_updates+=1;return True

    def _pair_params(self,left,right):
        cfg={}
        for key in (f"{left}:{right}",f"{right}:{left}"):
            if key in self.pair_overrides:cfg=dict(self.pair_overrides[key] or {});break
        return float(cfg.get("candidate_threshold",self.default_candidate)),float(cfg.get("strong_threshold",self.default_strong)),float(cfg.get("margin",self.default_margin))

    def _mature(self,camera_id):
        now=time.monotonic();return [t for t in self._tracks.get(camera_id,{}).values() if t.descriptor is not None and len(t.samples)>=self.min_descriptor_samples and now-t.last_seen<=self.track_timeout]

    def _assignment(self,left_tracks,right_tracks):
        scores=np.asarray([[_cosine(a.descriptor,b.descriptor) for b in right_tracks] for a in left_tracks],dtype=np.float32)
        assignments=[]
        try:
            import lap
            size=max(scores.shape);cost=np.ones((size,size),dtype=np.float64)*2.0;cost[:scores.shape[0],:scores.shape[1]]=1.0-scores;_total,x,_y=lap.lapjv(cost,extend_cost=False)
            for i,j in enumerate(x[:scores.shape[0]]):
                if 0<=j<scores.shape[1]:assignments.append((i,int(j),float(scores[i,j])))
        except Exception:
            candidates=sorted(((-float(scores[i,j]),i,j) for i in range(scores.shape[0]) for j in range(scores.shape[1])));used_i=set();used_j=set()
            for neg,i,j in candidates:
                if i in used_i or j in used_j:continue
                used_i.add(i);used_j.add(j);assignments.append((i,j,-neg))
        return assignments,scores

    def _evaluate_pairs(self):
        now=time.monotonic()
        for left_cam,right_cam in self.same_room_pairs:
            left=self._mature(left_cam);right=self._mature(right_cam)
            if not left or not right:continue
            assignments,scores=self._assignment(left,right);candidate,strong,margin=self._pair_params(left_cam,right_cam)
            for li,ri,score in assignments:
                a=left[li];b=right[ri];self._pair_attempts+=1
                row=sorted((float(v) for v in scores[li,:]),reverse=True);col=sorted((float(v) for v in scores[:,ri]),reverse=True);row2=row[1] if len(row)>1 else -1.0;col2=col[1] if len(col)>1 else -1.0;actual_margin=min(score-row2,score-col2);self._pair_scores.append({"pair":f"{left_cam}:{right_cam}","left":a.local_id,"right":b.local_id,"score":round(score,5),"margin":round(actual_margin,5),"ts":now})
                key=(left_cam,a.local_id,right_cam,b.local_id);previous=self._evidence.get(key,{"hits":0,"last":0.0})
                if score<candidate or actual_margin<margin:
                    self._pair_rejects+=1
                    if now-previous["last"]>self.evidence_ttl:self._evidence.pop(key,None)
                    continue
                hits=1 if now-previous["last"]>self.evidence_ttl else previous["hits"]+1;self._evidence[key]={"hits":hits,"last":now,"score":score};required=self.strong_confirm_hits if score>=strong else self.confirm_hits
                if hits<required:continue
                self._pair_confirms+=1;gid,reason=self.identities.merge_tracks(left_cam,a.local_id,right_cam,b.local_id,score,now)
                if gid:
                    a.global_id=gid;b.global_id=gid;a.last_similarity=score;b.last_similarity=score;a.last_reason="pair_"+reason;b.last_reason="pair_"+reason
                    if reason=="merged":self._pair_merges+=1
                self._evidence.pop(key,None)

    def _infer_loop(self):
        try:self._extractor=_OSNetExtractor(self.config);self._ready=True;log.info("CORE_V1_REID_V2_READY model=%s device=%s",self.config.get("model","osnet_x0_5"),self.config.get("device","cpu"))
        except Exception as exc:self._last_error=f"{type(exc).__name__}: {exc}";log.exception("CORE_V1_REID_DISABLED error=%s",self._last_error);return
        while not self._stop.is_set():
            try:key=self._job_queue.get(timeout=.25)
            except queue.Empty:self._evaluate_pairs();continue
            if key is None:break
            jobs=[];first=self._pop_job(key)
            if first is not None:jobs.append(first)
            deadline=time.monotonic()+self.batch_wait
            while len(jobs)<self.batch_size and time.monotonic()<deadline:
                try:key2=self._job_queue.get(timeout=max(0.0,deadline-time.monotonic()))
                except queue.Empty:break
                if key2 is None:self._stop.set();break
                job2=self._pop_job(key2)
                if job2 is not None:jobs.append(job2)
            now=time.monotonic();fresh=[]
            for job in jobs:
                if now-job.observed_at>self.max_job_age:self._stale_jobs+=1
                else:fresh.append(job)
            if not fresh:continue
            started=time.perf_counter()
            try:features=self._extractor.extract([j.crop for j in fresh])
            except Exception as exc:self._last_error=f"{type(exc).__name__}: {exc}";log.exception("CORE_V1_REID_BATCH_FAILED");continue
            self._last_batch_ms=(time.perf_counter()-started)*1000.0;self._batches+=1
            for job,feature in zip(fresh,features):
                track=self._tracks.get(job.camera_id,{}).get(job.local_id)
                if track is None:continue
                changed=self._update_descriptor(track,feature,job.quality,job.observed_at);self._embedded+=1
                if not changed or track.descriptor is None or len(track.samples)<self.min_descriptor_samples:continue
                gid,reason=self.identities.ensure_track(job.camera_id,job.local_id,track.descriptor,job.observed_at);track.global_id=gid;track.last_reason=reason;track.last_similarity=self.identities.identity_similarity(gid,track.descriptor)
            self._evaluate_pairs()

    def labels(self,camera_id):
        now=time.monotonic();return [{"local_id":t.local_id,"global_id":t.global_id,"box":t.box,"similarity":t.last_similarity,"reason":t.last_reason} for t in list(self._tracks.get(str(camera_id),{}).values()) if t.global_id and now-t.last_seen<=self.track_timeout]

    def snapshot(self):
        cameras={cid:[{"local_id":t.local_id,"global_id":t.global_id,"last_seen":t.last_seen,"hits":t.hits,"similarity":t.last_similarity,"reason":t.last_reason,"tracklet_samples":len(t.samples),"descriptor_ready":t.descriptor is not None and len(t.samples)>=self.min_descriptor_samples} for t in tracks.values()] for cid,tracks in self._tracks.items()}
        return {"algorithm":"tracklet-reid-v2","cameras":cameras,"global":self.identities.snapshot(),"recent_pair_scores":list(self._pair_scores)[-40:]}

    def metrics(self):
        elapsed=max(.001,time.monotonic()-self._started);return {"enabled":self.enabled,"ready":self._ready,"algorithm":"tracklet-reid-v2","model":self.config.get("model","osnet_x0_5"),"device":self.config.get("device","cpu"),"submitted":self._submitted,"embedded":self._embedded,"embed_rate":self._embedded/elapsed,"batches":self._batches,"last_batch_ms":self._last_batch_ms,"descriptor_updates":self._descriptor_updates,"duplicate_features":self._duplicate_features,"pair_attempts":self._pair_attempts,"pair_confirms":self._pair_confirms,"pair_merges":self._pair_merges,"pair_rejects":self._pair_rejects,"replaced_jobs":self._replaced_jobs,"stale_jobs":self._stale_jobs,"crop_rejects":self._crop_rejects,"frame_misses":self._frame_misses,"released_tracks":self._released_tracks,"queue_depth":self._job_queue.qsize(),"last_error":self._last_error,"identity":self.identities.metrics()}
