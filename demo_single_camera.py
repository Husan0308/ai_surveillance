#!/usr/bin/env python3
"""
BITTA KAMERA — TO'LIQ AI DEMO
👤 Detection  🎯 Tracking  😊 Face  🔥 Heatmap (OpenCV)
"""
import sys, os, cv2, numpy as np, yaml, time, types, importlib.util, sqlite3

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Logger stub
class _L:
    def info(self,*a,**k): pass
    def error(self,*a,**k): print(f"[ERR] {a}",flush=True)
    def warning(self,*a,**k): pass
    def debug(self,*a,**k): pass
m = types.ModuleType("backend.core.logger")
m.get_logger = lambda n: _L()
m.setup_logging = lambda *a,**k: None
sys.modules["backend.core.logger"] = m

# Config
config = {}
for f in ["config/project.yaml","project.yaml"]:
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        with open(p) as fh: config = yaml.safe_load(fh) or {}
        print(f"✅ Config: {f}"); break

print("="*70)
print("  BITTA KAMERA — TO'LIQ AI DEMO")
print("="*70)

# 1. Pose Engine
from backend.ai.pose_engine import PoseEngine
pose = PoseEngine(config)
print(f"[Pose]    available={pose.available} imgsz={pose.imgsz}")
if not pose.available: print("❌ Pose yo'q!"); sys.exit(1)

# 2. Tracker
from backend.ai.tracker import ByteTracker
tracker = ByteTracker(track_buffer=30, match_thresh=0.25,
                      high_thresh=0.35, low_thresh=0.1, new_track_thresh=0.3)
print(f"[Tracker] high={tracker.high_thresh} new={tracker.new_track_thresh}")

# 3. Face Engine
face = None; face_on = True
try:
    spec = importlib.util.spec_from_file_location(
        "face_direct", os.path.join(ROOT, "backend/ai/face_engine.py"))
    fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
    face = fm.FaceEngine(config)
    print(f"[Face]    available={face.available}")
    if face.available:
        db_path = os.path.join(ROOT, config.get("database",{}).get("sqlite_path","data/surveillance.db"))
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path); conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT person_id,name,embedding FROM face_embeddings WHERE embedding IS NOT NULL").fetchall():
                face.add_to_gallery(row["person_id"], row["name"], np.frombuffer(row["embedding"], dtype=np.float32))
            conn.close()
            print(f"[Face]    ✅ Gallery: {len(face.gallery)}")
except Exception as e:
    print(f"[Face]    ⚠ {e}")

# 4. Heatmap config
hm_cfg = config.get("heatmap", {})
GW = int(hm_cfg.get("grid_w", 64))
GH = int(hm_cfg.get("grid_h", 36))
OPACITY = float(hm_cfg.get("opacity", 0.6))
BLUR = bool(hm_cfg.get("blur", True))
print(f"[Heatmap] grid={GW}x{GH} opacity={OPACITY} blur={BLUR}")

# Heatmap grid (OpenCV — QImage YO'Q)
heat_grid = np.zeros((GH, GW), dtype=np.float32)
DECAY = 0.95  # har frame da eskirish

# 5. Kamera
source = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
print(f"[Camera]  {source}")
cap = cv2.VideoCapture(source)
if not cap.isOpened(): print("❌ Ochilmadi!"); sys.exit(1)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Camera]  ✅ {W}x{H}")

COLORS = [(0,255,0),(255,100,0),(0,165,255),(255,255,0),
          (255,0,255),(0,255,255),(128,255,0),(255,128,0)]
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,6),(5,11),(6,12),
            (5,7),(6,8),(7,9),(8,10),(0,1),(0,2),(1,3),(2,4)]

def iou(a, b):
    ax1,ay1,ax2,ay2=a[:4]; bx1,by1,bx2,by2=b[:4]
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    ua=max(0,ax2-ax1)*max(0,ay2-ay1); ub=max(0,bx2-bx1)*max(0,by2-by1)
    union=ua+ub-inter
    return inter/union if union>0 else 0

print("\n"+"="*70)
print("  BOSHLANDI!  q=chiqish  h=heatmap  f=face  s=screenshot")
print("="*70)

show_heat = True
fc = 0
track_names = {}
face_interval = 5

while True:
    ret, frame = cap.read()
    if not ret or frame is None: time.sleep(0.05); continue
    fc += 1; t0 = time.time()

    # Stage 1: Detection
    boxes, kpts = pose.detect(frame)
    # Stage 2: Tracking
    tracks = tracker.update(boxes)

    # Stage 3: Face
    if face_on and face and face.available and fc % face_interval == 0:
        faces = face.detect(frame, need_embedding=True)
        for tr in tracks:
            best_f, best_i = None, 0.3
            for f in faces:
                s = iou(tr.box, f["bbox"])
                if s > best_i: best_i=s; best_f=f
            if best_f and best_f["embedding"] is not None:
                pid, name, score = face.recognize(best_f["embedding"])
                track_names[tr.id] = (name if pid else "Unknown", score)

    # Stage 4: Heatmap update (OpenCV grid)
    heat_grid *= DECAY  # eskirish
    ankle_count = 0
    for tr in tracks:
        x1,y1,x2,y2 = tr.box[:4]
        ax, ay = (x1+x2)/2.0, y2  # default ankle
        if kpts:
            for kpt in kpts:
                if kpt is None or len(kpt)<17: continue
                ap = pose.ankle_point(kpt)
                if ap and x1<=ap[0]<=x2 and y1<=ap[1]<=y2:
                    ax, ay = ap; break
        # Grid koordinata
        gx = int(ax / W * GW)
        gy = int(ay / H * GH)
        gx = max(0, min(GW-1, gx))
        gy = max(0, min(GH-1, gy))
        heat_grid[gy, gx] = min(heat_grid[gy, gx] + 1.0, 5.0)
        ankle_count += 1

    ms = (time.time()-t0)*1000

    # === Vizualizatsiya ===
    disp = frame.copy()

    # Heatmap overlay (to'liq OpenCV)
    if show_heat and ankle_count > 0:
        try:
            # Normalize
            hmax = heat_grid.max()
            if hmax > 0:
                norm = heat_grid / hmax
            else:
                norm = heat_grid

            # Blur
            if BLUR:
                norm = cv2.GaussianBlur(norm, (15, 15), 0)

            # Resize to frame size
            h_resized = cv2.resize(norm, (W, H), interpolation=cv2.INTER_LINEAR)

            # Colormap (JET = issiq ranglar)
            h_color = cv2.applyColorMap((h_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Faqat heatmap bor joylarni blend qilish
            mask = (h_resized > 0.05).astype(np.float32)
            alpha = mask * OPACITY
            alpha3 = np.stack([alpha]*3, axis=-1)

            disp = (disp.astype(np.float32) * (1 - alpha3) +
                    h_color.astype(np.float32) * alpha3).astype(np.uint8)
        except Exception as e:
            if fc % 100 == 0: print(f"  ⚠ heat render: {e}")

    # Track box + skeleton + label
    for i, tr in enumerate(tracks):
        c = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = [int(v) for v in tr.box[:4]]
        cv2.rectangle(disp, (x1,y1), (x2,y2), c, 2)
        ni = track_names.get(tr.id)
        label = f"ID:{tr.id} {ni[0]} {ni[1]:.2f}" if ni else f"ID:{tr.id}"
        cv2.rectangle(disp, (x1,y1-25), (x1+len(label)*9,y1), c, -1)
        cv2.putText(disp, label, (x1+2,y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        if kpts:
            for kpt in kpts:
                if kpt is None or len(kpt)<17: continue
                kcx = np.mean([kpt[5][0],kpt[6][0]]) if kpt[5][0]>0 and kpt[6][0]>0 else 0
                if not (x1-30<=kcx<=x2+30): continue
                for a,b in SKELETON:
                    pa,pb = kpt[a],kpt[b]
                    if pa[0]>0 and pa[1]>0 and pb[0]>0 and pb[1]>0:
                        cv2.line(disp,(int(pa[0]),int(pa[1])),(int(pb[0]),int(pb[1])),c,2)
                for kp in kpt:
                    if kp[0]>0 and kp[1]>0:
                        cv2.circle(disp,(int(kp[0]),int(kp[1])),3,c,-1)

    # HUD
    hud = f"Trk:{len(tracks)} Box:{len(boxes)} Ank:{ankle_count} {ms:.0f}ms H:{'ON' if show_heat else 'OFF'} F:{'ON' if face_on else 'OFF'}"
    cv2.rectangle(disp,(0,0),(W,40),(0,0,0),-1)
    cv2.putText(disp,hud,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    if fc % 30 == 0:
        gmax = heat_grid.max()
        print(f"  Fr {fc:4d} | Box:{len(boxes):2d} | Trk:{len(tracks):2d} | "
              f"Ank:{ankle_count:2d} | GridMax:{gmax:.2f} | {ms:.0f}ms")

    cv2.imshow("AI Demo (q=quit h=heat f=face s=shot)", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('h'): show_heat = not show_heat; print(f"  🔥 Heat: {'ON' if show_heat else 'OFF'}")
    if k == ord('f'): face_on = not face_on; print(f"  😊 Face: {'ON' if face_on else 'OFF'}")
    if k == ord('s'):
        fn = f"/tmp/demo_{fc}.jpg"; cv2.imwrite(fn, disp); print(f"  📸 {fn}")

cap.release(); cv2.destroyAllWindows()
print(f"\n📊 {fc} frames | tracks:{len(tracker.tracks)} | names:{len(track_names)}")
print("🟢 HAMMASI ISHLADI!" if tracker.tracks else "🔴 ISHLAMAYAPTI")
