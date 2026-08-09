"""Isolated maintained Ultralytics ByteTrack adapter for P2 evaluation."""
from types import SimpleNamespace
import time
import numpy as np
from .schemas import CameraTrackResult,TrackedPerson,TrackState

class _Results:
    def __init__(self,xyxy,conf=None,cls=None):
        self.xyxy=np.asarray(xyxy,np.float32).reshape(-1,4)
        self.conf=np.asarray(conf if conf is not None else np.ones(len(self.xyxy)),np.float32)
        self.cls=np.asarray(cls if cls is not None else np.zeros(len(self.xyxy)),np.float32)
    @property
    def xywh(self):
        if not len(self.xyxy):return np.empty((0,4),np.float32)
        out=self.xyxy.copy();out[:,:2]=(self.xyxy[:,:2]+self.xyxy[:,2:])/2;out[:,2:]=self.xyxy[:,2:]-self.xyxy[:,:2]
        return out
    def __len__(self):return len(self.xyxy)
    def __getitem__(self,item):return _Results(self.xyxy[item],self.conf[item],self.cls[item])

class OfficialByteTrackAdapter:
    """Person-only, per-camera adapter; it never invokes detector inference."""
    def __init__(self,camera_id,config=None):
        from ultralytics.trackers.byte_tracker import BYTETracker
        cfg=config or {};fps=max(1,float(cfg.get("effective_ai_fps",10)))
        lost_ms=float(cfg.get("max_lost_time_ms",cfg.get("lost_memory_seconds",1.5)*1000))
        args=SimpleNamespace(track_high_thresh=float(cfg.get("track_high_thresh",.22)),
            track_low_thresh=float(cfg.get("track_low_thresh",.05)),new_track_thresh=float(cfg.get("new_track_thresh",.28)),
            track_buffer=max(1,int(round(lost_ms*fps/1000))),match_thresh=float(cfg.get("official_match_thresh",.8)),
            fuse_score=bool(cfg.get("fuse_score",True)))
        self.camera_id=camera_id;self.tracker=BYTETracker(args);self.created={};self.last_seen={}
    def update(self,result):
        rows=self.tracker.update(_Results([d.bbox_xyxy for d in result.detections],[d.confidence for d in result.detections]))
        now=result.receive_timestamp;tracks=[]
        for row in rows:
            local_id=int(row[4]);track_id=f"{self.camera_id}:TRACK-{local_id:05d}";self.created.setdefault(local_id,now);self.last_seen[local_id]=now
            bbox=tuple(float(v) for v in row[:4]);confidence=float(row[5])
            tracks.append(TrackedPerson(track_id,TrackState.CONFIRMED,bbox,confidence,0,0,0,(0.,0.),self.camera_id,local_id,
                self.created[local_id],now,now-self.created[local_id],0.,bbox,0,True))
        return CameraTrackResult(result.camera_id,result.frame_id,result.capture_timestamp,result.receive_timestamp,tuple(tracks))

def nvdcf_capability():
    from pathlib import Path
    root=Path("/opt/nvidia/deepstream/deepstream-7.1")
    library=root/"lib/libnvds_nvmultiobjecttracker.so"
    config=root/"samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
    return {"runtime_available":library.exists(),"config_available":config.exists(),
        "comparable_external_detection_adapter":False,
        "reason":"NvDCF consumes NvDsBatchMeta on GstBuffers; the current Python detector emits structured CPU metadata, so a fair adapter requires a future DeepStream metadata-injection harness."}
