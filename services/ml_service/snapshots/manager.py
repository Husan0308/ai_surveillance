"""Bounded quality-replacement snapshots for runtime unknown identities."""
from __future__ import annotations
import json,threading,time,queue,uuid
from pathlib import Path
import cv2

class UnknownSnapshotManager:
    def __init__(self,root,max_identities=500,retention_days=7,min_improvement=.10):
        self.root=Path(root).expanduser().resolve();self.root.mkdir(parents=True,exist_ok=True);self.max_identities=max(1,int(max_identities));self.retention=float(retention_days)*86400;self.improvement=float(min_improvement);self._lock=threading.RLock();self.index={};self._queue=queue.Queue(32);self._stop=threading.Event();self._worker=threading.Thread(target=self._run,name="unknown-snapshots",daemon=False);self._load();self.cleanup();self._worker.start()
    def _load(self):
        path=self.root/"index.json"
        try:self.index=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError,ValueError):self.index={}
    def _save(self):
        temp=self.root/"index.tmp";temp.write_text(json.dumps(self.index,sort_keys=True),encoding="utf-8");temp.replace(self.root/"index.json")
    @staticmethod
    def quality(image,bbox):
        if image is None or not getattr(image,"size",0):return 0.0
        h,w=image.shape[:2];x1,y1,x2,y2=[max(0,int(v)) for v in bbox];x2=min(w,x2);y2=min(h,y2)
        if x2<=x1 or y2<=y1:return 0.0
        crop=image[y1:y2,x1:x2];area=min(1.0,(crop.shape[0]*crop.shape[1])/max(w*h*.25,1));sharp=min(1.0,float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())/200);return .55*area+.45*sharp
    @staticmethod
    def _is_unknown(global_id):return str(global_id).startswith(("UNK ","UNK-"))
    def submit(self,global_id,camera_id,frame_id,timestamp,image,bbox,kind="body"):
        if not self._is_unknown(global_id):return False
        h,w=image.shape[:2];x1,y1,x2,y2=[max(0,int(v)) for v in bbox];x2=min(w,x2);y2=min(h,y2);crop=image[y1:y2,x1:x2]
        if not crop.size:return False
        try:self._queue.put_nowait((global_id,camera_id,frame_id,timestamp,crop.copy(),(0,0,crop.shape[1],crop.shape[0]),kind));return True
        except queue.Full:return False
    def _run(self):
        while not self._stop.is_set():
            try:item=self._queue.get(timeout=.25)
            except queue.Empty:continue
            try:self.consider(*item)
            finally:self._queue.task_done()
    def close(self,timeout=3):self._stop.set();self._worker.join(timeout);self._save()
    def metrics(self):return {"identities":len(self.index),"queue_depth":self._queue.qsize(),"disk_bytes":self.disk_usage()}
    def consider(self,global_id,camera_id,frame_id,timestamp,image,bbox,kind="body",quality=None):
        if not self._is_unknown(global_id) or kind not in ("face","body"):return None
        h,w=image.shape[:2];x1,y1,x2,y2=[max(0,int(v)) for v in bbox];x2=min(w,x2);y2=min(h,y2);crop=image[y1:y2,x1:x2]
        if not crop.size:return None
        score=float(self.quality(image,bbox) if quality is None else quality);key=str(global_id);now=float(timestamp or time.time())
        with self._lock:
            current=self.index.get(key,{}).get(kind)
            if current and score<float(current.get("quality",0))+self.improvement:return None
            filename=f"{key}_{kind}.jpg";path=(self.root/filename).resolve()
            if path.parent!=self.root:raise ValueError("unsafe snapshot path")
            temporary=self.root/f".{filename}.{uuid.uuid4().hex}.tmp.jpg"
            if not cv2.imwrite(str(temporary),crop,[cv2.IMWRITE_JPEG_QUALITY,85]):return None
            temporary.replace(path)
            record={"global_id":key,"camera_id":str(camera_id),"frame_id":int(frame_id),"timestamp":now,"quality":score,"bbox":list(map(float,bbox)),"type":kind,"storage_path":str(path)}
            self.index.setdefault(key,{})[kind]=record;self._save();self.cleanup(now);return record
    def remap(self,old_id,canonical_id):
        if old_id==canonical_id:return False
        with self._lock:
            old=self.index.pop(str(old_id),{})
            target=self.index.setdefault(str(canonical_id),{})
            for kind,record in old.items():
                current=target.get(kind)
                if current is None or float(record.get("quality",0))>float(current.get("quality",0)):
                    old_path=Path(record.get("storage_path",""));new_path=(self.root/f"{canonical_id}_{kind}.jpg").resolve()
                    if new_path.parent!=self.root:continue
                    if old_path.exists():old_path.replace(new_path)
                    record={**record,"global_id":str(canonical_id),"storage_path":str(new_path)};target[kind]=record
                else:
                    Path(record.get("storage_path","")).unlink(missing_ok=True)
            self._save();return bool(old)

    def cleanup(self,now=None):
        now=float(now or time.time())
        with self._lock:
            for key,value in list(self.index.items()):
                newest=max((float(item.get("timestamp",0)) for item in value.values()),default=0)
                if now-newest>self.retention:
                    for item in value.values():Path(item.get("storage_path","")).unlink(missing_ok=True)
                    self.index.pop(key,None)
            if len(self.index)>self.max_identities:
                ordered=sorted(self.index,key=lambda key:max(float(v.get("timestamp",0)) for v in self.index[key].values()))
                for key in ordered[:len(self.index)-self.max_identities]:
                    for item in self.index[key].values():Path(item.get("storage_path","")).unlink(missing_ok=True)
                    self.index.pop(key,None)
            self._save();return len(self.index)
    def disk_usage(self):
        total=0
        with self._lock:
            for path in self.root.glob("*.jpg"):
                try:total+=path.stat().st_size
                except FileNotFoundError:continue
        return total
