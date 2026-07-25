import time
import numpy as np

def iou(a, b):
    ax1,ay1,ax2,ay2 = a[:4]; bx1,by1,bx2,by2 = b[:4]
    ix1,iy1 = max(ax1,bx1),max(ay1,by1); ix2,iy2 = min(ax2,bx2),min(ay2,by2)
    iw,ih = max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter = iw*ih
    ua = max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ub = max(0.0,bx2-bx1)*max(0.0,by2-by1)
    union = ua+ub-inter
    return inter/union if union>0 else 0.0

def iou_matrix(boxes_a, boxes_b):
    if len(boxes_a)==0 or len(boxes_b)==0:
        return np.zeros((len(boxes_a),len(boxes_b)),dtype=np.float32)
    a=np.array([b[:4] for b in boxes_a],dtype=np.float32)
    b=np.array([x[:4] for x in boxes_b],dtype=np.float32)
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    iw=np.maximum(0.0,x2-x1); ih=np.maximum(0.0,y2-y1); inter=iw*ih
    ua=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); ub=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    union=ua[:,None]+ub[None,:]-inter
    return np.where(union>0,inter/union,0.0).astype(np.float32)

def cosine(a,b):
    if a is None or b is None: return 0.0
    a=np.asarray(a,dtype=np.float32); b=np.asarray(b,dtype=np.float32)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

class Track:
    def __init__(self,track_id,box,embedding=None):
        self.id=track_id; self.box=list(box)
        self.conf=float(box[4]) if len(box)>4 else 0.0
        self.hits=1; self.missing=0
        self.first_seen=time.time(); self.last_seen=time.time()
        self.cx=(self.box[0]+self.box[2])/2.0; self.cy=(self.box[1]+self.box[3])/2.0
        self.vx=0.0; self.vy=0.0
        self.known=False; self.person_id=None; self.name=f"Unknown-{self.id}"
        self.face_conf=0.0; self.unknown_saved=False
        self.last_recognized_event_ts=0.0; self.last_unknown_event_ts=0.0
        self.reid_emb=embedding
    @property
    def age(self): return time.time()-self.first_seen
    def predict(self):
        w=self.box[2]-self.box[0]; h=self.box[3]-self.box[1]
        pcx=self.cx+self.vx; pcy=self.cy+self.vy
        return [pcx-w/2,pcy-h/2,pcx+w/2,pcy+h/2,self.conf]
    def update(self,box,embedding=None):
        ox,oy=self.cx,self.cy; self.box=list(box)
        self.conf=float(box[4]) if len(box)>4 else self.conf
        self.cx=(self.box[0]+self.box[2])/2.0; self.cy=(self.box[1]+self.box[3])/2.0
        self.vx=0.7*self.vx+0.3*(self.cx-ox); self.vy=0.7*self.vy+0.3*(self.cy-oy)
        self.hits+=1; self.missing=0; self.last_seen=time.time()
        if embedding is not None: self.reid_emb=embedding

class ByteTracker:
    def __init__(self,track_buffer=30,match_thresh=0.25,high_thresh=0.35,
                 low_thresh=0.1,new_track_thresh=0.3,reid_weight=0.0):
        self.next_id=1; self.tracks=[]
        self.track_buffer=int(track_buffer); self.match_thresh=float(match_thresh)
        self.high_thresh=float(high_thresh); self.low_thresh=float(low_thresh)
        self.new_track_thresh=float(new_track_thresh); self.reid_weight=float(reid_weight)
    def _iou(self,a,b):
        ax1,ay1,ax2,ay2=a[0],a[1],a[2],a[3]; bx1,by1,bx2,by2=b[0],b[1],b[2],b[3]
        ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
        iw,ih=max(0.0,ix2-ix1),max(0.0,iy2-iy1); inter=iw*ih
        ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1); ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
        union=ua+ub-inter
        return inter/union if union>0 else 0.0
    def update(self,detections,det_embeddings=None):
        for t in self.tracks: t.predict()
        active=[t for t in self.tracks if t.missing<=self.track_buffer]
        if not detections:
            for t in self.tracks: t.missing+=1
            self.tracks=[t for t in self.tracks if t.missing<=self.track_buffer*2]
            return [t for t in self.tracks if t.missing<=self.track_buffer]
        high_dets,low_dets=[],[]
        for d in detections:
            c=d[4] if len(d)>4 else 0.0
            if c>=self.high_thresh: high_dets.append(d)
            elif c>=self.low_thresh: low_dets.append(d)
        mti,mdi=set(),set()
        if active and high_dets:
            pairs=[]
            for ti,tr in enumerate(active):
                for di,det in enumerate(high_dets):
                    s=self._iou(tr.box,det)
                    if s>=self.match_thresh: pairs.append((s,ti,di))
            pairs.sort(key=lambda x:x[0],reverse=True)
            for s,ti,di in pairs:
                if ti in mti or di in mdi: continue
                active[ti].update(high_dets[di]); mti.add(ti); mdi.add(di)
        for ti,tr in enumerate(active):
            if ti not in mti: tr.missing+=1
        rem=[active[ti] for ti in range(len(active)) if ti not in mti]
        if rem and low_dets:
            p2=[]
            for ti,tr in enumerate(rem):
                for di,det in enumerate(low_dets):
                    s=self._iou(tr.box,det)
                    if s>=self.match_thresh: p2.append((s,ti,di))
            p2.sort(key=lambda x:x[0],reverse=True); ml=set()
            for s,ti,di in p2:
                if ti in ml: continue
                rem[ti].update(low_dets[di]); ml.add(ti)
            for ti,tr in enumerate(rem):
                if ti not in ml: tr.missing+=1
        for di,det in enumerate(high_dets):
            if di in mdi: continue
            c=det[4] if len(det)>4 else 0.0
            if c<self.new_track_thresh: continue
            emb=det_embeddings[di] if det_embeddings and di<len(det_embeddings) else None
            tr=Track(self.next_id,det,emb); self.next_id+=1; self.tracks.append(tr)
        self.tracks=[t for t in self.tracks if t.missing<=self.track_buffer*2]
        return [t for t in self.tracks if t.missing<=self.track_buffer]
