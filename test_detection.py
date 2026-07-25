#!/usr/bin/env python3
"""ByteTracker test — import dan mustaqil, kod ichida."""
import sys, os, cv2, numpy as np, yaml, time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Logger stub
class _L:
    def info(self,*a,**k): pass
    def error(self,*a,**k): print(f"[ERR] {a}",flush=True)
    def warning(self,*a,**k): pass
    def debug(self,*a,**k): pass
def get_logger(n): return _L()
import types
m = types.ModuleType("backend.core.logger")
m.get_logger = get_logger
sys.modules["backend.core.logger"] = m

# Config
config = {}
for f in ["config/project.yaml","project.yaml"]:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        with open(p) as fh: config = yaml.safe_load(fh) or {}
        print(f"✅ Config: {f}"); break

print("="*60)
print("BYTETRACKER TEST (kod ichida)")
print("="*60)

from backend.ai.pose_engine import PoseEngine
pose = PoseEngine(config)
print(f"Pose: available={pose.available}, conf={pose.conf}")

# ===== Track klassi (tracker.py dan nusxa) =====
class Track:
    def __init__(self, track_id, box, embedding=None):
        self.id = track_id
        self.box = list(box)
        self.conf = float(box[4]) if len(box) > 4 else 0.0
        self.hits = 1
        self.missing = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.cx = (self.box[0] + self.box[2]) / 2.0
        self.cy = (self.box[1] + self.box[3]) / 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.known = False
        self.person_id = None
        self.name = f"Unknown-{self.id}"
        self.face_conf = 0.0
        self.unknown_saved = False
        self.last_recognized_event_ts = 0.0
        self.last_unknown_event_ts = 0.0
        self.reid_emb = embedding

    @property
    def age(self):
        return time.time() - self.first_seen

    def predict(self):
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        pcx = self.cx + self.vx
        pcy = self.cy + self.vy
        return [pcx - w/2, pcy - h/2, pcx + w/2, pcy + h/2, self.conf]

    def update(self, box, embedding=None):
        old_cx, old_cy = self.cx, self.cy
        self.box = list(box)
        self.conf = float(box[4]) if len(box) > 4 else self.conf
        self.cx = (self.box[0] + self.box[2]) / 2.0
        self.cy = (self.box[1] + self.box[3]) / 2.0
        self.vx = 0.7 * self.vx + 0.3 * (self.cx - old_cx)
        self.vy = 0.7 * self.vy + 0.3 * (self.cy - old_cy)
        self.hits += 1
        self.missing = 0
        self.last_seen = time.time()
        if embedding is not None:
            self.reid_emb = embedding

# ===== FIXED ByteTracker (ichida) =====
class FixedByteTracker:
    def __init__(self, track_buffer=30, match_thresh=0.25, high_thresh=0.35,
                 low_thresh=0.1, new_track_thresh=0.3, reid_weight=0.0):
        self.next_id = 1
        self.tracks = []
        self.track_buffer = int(track_buffer)
        self.match_thresh = float(match_thresh)
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(new_track_thresh)

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a[0],a[1],a[2],a[3]
        bx1,by1,bx2,by2 = b[0],b[1],b[2],b[3]
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
        inter = iw*ih
        ua = max(0,ax2-ax1)*max(0,ay2-ay1)
        ub = max(0,bx2-bx1)*max(0,by2-by1)
        union = ua+ub-inter
        return inter/union if union>0 else 0

    def update(self, detections, det_embeddings=None):
        for t in self.tracks:
            t.predict()
        active = [t for t in self.tracks if t.missing <= self.track_buffer]

        if not detections:
            for t in self.tracks: t.missing += 1
            self.tracks = [t for t in self.tracks if t.missing <= self.track_buffer*2]
            return [t for t in self.tracks if t.missing <= self.track_buffer]

        high_dets, low_dets = [], []
        for d in detections:
            c = d[4] if len(d)>4 else 0
            if c >= self.high_thresh: high_dets.append(d)
            elif c >= self.low_thresh: low_dets.append(d)

        matched_ti, matched_di = set(), set()
        if active and high_dets:
            pairs = []
            for ti,tr in enumerate(active):
                for di,det in enumerate(high_dets):
                    s = self._iou(tr.box, det)
                    if s >= self.match_thresh:
                        pairs.append((s,ti,di))
            pairs.sort(key=lambda x:x[0], reverse=True)
            for s,ti,di in pairs:
                if ti in matched_ti or di in matched_di: continue
                active[ti].update(high_dets[di])
                matched_ti.add(ti); matched_di.add(di)

        for ti,tr in enumerate(active):
            if ti not in matched_ti: tr.missing += 1

        remaining = [active[ti] for ti in range(len(active)) if ti not in matched_ti]
        if remaining and low_dets:
            pairs2 = []
            for ti,tr in enumerate(remaining):
                for di,det in enumerate(low_dets):
                    s = self._iou(tr.box, det)
                    if s >= self.match_thresh:
                        pairs2.append((s,ti,di))
            pairs2.sort(key=lambda x:x[0], reverse=True)
            ml = set()
            for s,ti,di in pairs2:
                if ti in ml: continue
                remaining[ti].update(low_dets[di]); ml.add(ti)
            for ti,tr in enumerate(remaining):
                if ti not in ml: tr.missing += 1

        for di,det in enumerate(high_dets):
            if di in matched_di: continue
            c = det[4] if len(det)>4 else 0
            if c < self.new_track_thresh: continue
            emb = det_embeddings[di] if det_embeddings and di<len(det_embeddings) else None
            tr = Track(self.next_id, det, emb)
            self.next_id += 1
            self.tracks.append(tr)

        self.tracks = [t for t in self.tracks if t.missing <= self.track_buffer*2]
        return [t for t in self.tracks if t.missing <= self.track_buffer]

tracker = FixedByteTracker(high_thresh=0.35, new_track_thresh=0.3)
print(f"FixedByteTracker: high={tracker.high_thresh}, new={tracker.new_track_thresh}")

# Video
source = sys.argv[1] if len(sys.argv)>1 else "/dev/video0"
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"❌ Cannot open {source}"); sys.exit(1)
print(f"✅ {int(cap.get(3))}x{int(cap.get(4))}")

print("\nRunning 10 frames...")
print("-"*65)
print(f"{'Fr':>3} | {'Box':>3} | {'Confs':>22} | {'Trk':>3} | IDs")
print("-"*65)

for i in range(10):
    ret, frame = cap.read()
    if not ret: break
    boxes, kpts = pose.detect(frame)
    confs = [round(b[4],3) if len(b)>4 else -1 for b in boxes]
    tracks = tracker.update(boxes)
    ids = [t.id for t in tracks]
    print(f"{i:3d} | {len(boxes):3d} | {str(confs):>22} | {len(tracks):3d} | {ids}")

print("-"*65)
print(f"\n📊 Final: {len(tracker.tracks)} tracks, next_id={tracker.next_id}")
if tracker.tracks:
    print("🟢 BYTETRACKER ISHLAYAPTI!")
else:
    print("🔴 BYTETRACKER ISHLAMAYAPTI")
cap.release()