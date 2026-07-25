#!/usr/bin/env python3
"""TO'LIQ PIPELINE: Detection + Tracking + Heatmap (bitta kamera)"""
import sys, os, cv2, numpy as np, yaml, time, types, importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Logger stub (circular import buzish)
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
print("  TO'LIQ PIPELINE TEST")
print("="*70)

# 1. Pose Engine (haqiqiy backend)
from backend.ai.pose_engine import PoseEngine
pose = PoseEngine(config)
print(f"[Pose]    available={pose.available} imgsz={pose.imgsz} conf={pose.conf}")
if not pose.available:
    print("❌ Pose yo'q!"); sys.exit(1)

# 2. Tracker (haqiqiy backend, tuzatilgan)
from backend.ai.tracker import ByteTracker
tracker = ByteTracker(track_buffer=30, match_thresh=0.25,
                      high_thresh=0.35, low_thresh=0.1, new_track_thresh=0.3)
print(f"[Tracker] high={tracker.high_thresh} new={tracker.new_track_thresh}")

# 3. Heatmap Engine (importlib bilan)
spec = importlib.util.spec_from_file_location(
    "hm_direct", os.path.join(ROOT, "backend/features/heatmap.py"))
hm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm_mod)
heatmap = hm_mod.HeatmapEngine(config, "TEST-CAM")
heatmap.set_on(True)
print(f"[Heatmap] grid={heatmap.gw}x{heatmap.gh} on={heatmap.on if hasattr(heatmap,'on') else '?'}")

# 4. Kamera
source = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
print(f"[Camera]  {source}")
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print("❌ Kamera ochilmadi!"); sys.exit(1)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Camera]  ✅ {W}x{H}")

COLORS = [(0,255,0),(255,100,0),(0,0,255),(255,255,0),
          (255,0,255),(0,255,255),(128,255,0),(255,128,0)]

# COCO skeleton juftliklari
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,6),(5,11),(6,12),
            (5,7),(6,8),(7,9),(8,10),(0,1),(0,2),(1,3),(2,4)]

print("\n"+"="*70)
print("  BOSHLANDI!  q=chiqish  h=heatmap on/off")
print("="*70)

show_heat = True
fc = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.05); continue
    fc += 1
    t0 = time.time()

    # === Stage 1: Detection (Pose) ===
    boxes, kpts = pose.detect(frame)

    # === Stage 2: Tracking ===
    tracks = tracker.update(boxes)

    # === Stage 3: Heatmap persons (ankle nuqtalari) ===
    hpersons = []
    for tr in tracks:
        x1,y1,x2,y2 = tr.box[:4]
        ankle = ((x1+x2)/2.0, y2)  # default: bbox pastki markazi
        # Aniq ankle: track bilan mos kelgan keypoint dan
        if kpts:
            for kpt in kpts:
                if kpt is None or len(kpt) < 17: continue
                ap = pose.ankle_point(kpt)
                if ap and x1 <= ap[0] <= x2 and y1 <= ap[1] <= y2:
                    ankle = ap; break
        class _P: pass
        p = _P(); p.track_id = tr.id; p.box = list(tr.box[:4]); p.ankle = ankle
        hpersons.append(p)

    heatmap.update(hpersons, W, H, online=True)
    ms = (time.time()-t0)*1000

    # === Vizualizatsiya ===
    disp = frame.copy()

    # Heatmap overlay
    if show_heat and hpersons:
        try:
            hi = heatmap.get_image()
            if hi is not None and not hi.isNull():
                ptr = hi.bits(); ptr.setsize(hi.byteCount())
                arr = np.array(ptr).reshape(hi.height(), hi.width(), 4)
                hbgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                hres = cv2.resize(hbgr, (W, H), interpolation=cv2.INTER_LINEAR)
                mask = cv2.cvtColor(hres, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                m3 = cv2.merge([mask]*3)
                ov = cv2.addWeighted(disp, 1.0, hres, 0.55, 0)
                disp = np.where(m3 > 0, ov, disp)
        except Exception as e:
            if fc % 100 == 0: print(f"  ⚠ heat render: {e}")

    # Track box + skeleton + label
    for i, tr in enumerate(tracks):
        c = COLORS[i % len(COLORS)]
        x1,y1,x2,y2 = [int(v) for v in tr.box[:4]]
        cv2.rectangle(disp, (x1,y1), (x2,y2), c, 2)
        cv2.putText(disp, f"ID:{tr.id} {tr.conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        # Skeleton
        if kpts:
            for kpt in kpts:
                if kpt is None or len(kpt) < 17: continue
                # faqat shu track ga tegishli kpt
                kcx = np.mean([kpt[5][0], kpt[6][0]]) if kpt[5][0]>0 and kpt[6][0]>0 else 0
                if not (x1-30 <= kcx <= x2+30): continue
                for a,b in SKELETON:
                    pa, pb = kpt[a], kpt[b]
                    if pa[0]>0 and pa[1]>0 and pb[0]>0 and pb[1]>0:
                        cv2.line(disp, (int(pa[0]),int(pa[1])),
                                 (int(pb[0]),int(pb[1])), c, 2)
                for kp in kpt:
                    if kp[0]>0 and kp[1]>0:
                        cv2.circle(disp, (int(kp[0]),int(kp[1])), 3, c, -1)

    # HUD
    hud = f"Tracks:{len(tracks)}  Boxes:{len(boxes)}  {ms:.0f}ms  Heat:{'ON' if show_heat else 'OFF'}"
    cv2.rectangle(disp, (0,0), (W, 45), (0,0,0), -1)
    cv2.putText(disp, hud, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if fc % 30 == 0:
        print(f"  Fr {fc:4d} | Box:{len(boxes):2d} | Trk:{len(tracks):2d} | "
              f"Heat:{len(hpersons):2d} | {ms:.0f}ms")

    cv2.imshow("Pipeline Test (q=quit, h=heatmap)", disp)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k == ord('h'):
        show_heat = not show_heat
        heatmap.set_on(show_heat)
        print(f"  🔥 Heatmap: {'ON' if show_heat else 'OFF'}")

cap.release(); cv2.destroyAllWindows()
print(f"\n📊 {fc} frames, final tracks: {len(tracker.tracks)}")
print("🟢 DETECTION+TRACKING ISHLAYAPTI!" if tracker.tracks else "🔴 ISHLAMAYAPTI")
print("="*70)
