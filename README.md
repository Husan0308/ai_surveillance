# AI Surveillance — Sentinel Camera V2

Current production branch: `rebuild/gpu-v2-clean`.

The live camera path is intentionally kept simple and GPU-native:

```text
6 RTSP cameras
    ↓
DeepStream / NVDEC
    ↓
YOLO26m sparse person detection (704x384, micro-batch 2)
    ↓
per-camera NvDCF local tracking
    ↓
OSD / 2x3 Sentinel wall
```

Cross-camera identity is a separate bounded side path so a slow model cannot stall
camera ingest, detection, NvDCF or display:

```text
NvDCF track metadata + existing sparse detector frame mailbox
    ↓
5–10 temporally diverse person crops
    ↓
quality / overlap / truncation / duplicate gates
    ↓
diversity-aware top-3 evidence
    ↓
CPU ReID embedding + gallery candidate ranking
    ↓
room/time hard constraints + runner-up margin
    ↓
TENTATIVE Global ID
    ↓
Qwen-VL OLD-vs-NEW visual verification (async, optional)
    ↓
fresh evidence votes
    ↓
CONFIRMED Global ID / LOST / SUSPECT / rollback
```

## Camera topology

`config/cameras.yaml` is authoritative:

- `Devs`: CAM-01 + CAM-04
- `Entrance`: CAM-02 + CAM-05
- `Main Rooms`: CAM-03 + CAM-06

Two cameras in the same room may observe the same Global ID simultaneously. Two
active tracks in the same camera may not share one Global ID, and one Global ID may
not be active in two different rooms at the same time.

## ReID safety rules

- Local NvDCF IDs remain authoritative inside each camera.
- A local ID break does not mean the Global ID is lost.
- Short chair/desk occlusions keep the previous Global ID during the bounded display
  hold; a newly created local track is matched back against the same Global gallery.
- Tentative evidence is quarantined and cannot poison a confirmed gallery.
- Appearance matching uses multi-shot evidence and a runner-up margin, not one crop.
- Qwen returns `SAME`, `DIFFERENT`, or `UNCERTAIN`; it is independent evidence and
  cannot bypass room/time hard constraints.
- False merges are treated as more dangerous than temporary false splits.
- Confirmed assignments can enter `SUSPECT` and be rolled back after repeated
  contradictory evidence.

## ReID backend

`config/reid.yaml` defaults to `backend: auto` and keeps ReID on CPU so the GTX GPU
remains available to the live detector/tracker path.

- Preferred when installed: OSNet-AIN x1.0 / MSMT17 through `torchreid`.
- Safe fallback: Intel/OpenVINO `person-reidentification-retail-0288` through OpenCV
  DNN, also CPU-only.

Model files are stored under `.runtime/camera_v2/models/reid/` and are gitignored.

Prepare and warm the selected model once:

```bash
python scripts/setup_camera_v2_reid.py
```

## Qwen verifier

Qwen is optional at runtime and never blocks video. Point it at an OpenAI-compatible
Qwen-VL server:

```bash
export QWEN_REID_URL=http://127.0.0.1:PORT/v1
export QWEN_REID_MODEL=qwen3-vl
```

The verifier receives one compact JPEG contact sheet containing up to three OLD
Global-ID crops on the first row and up to three NEW-track crops on the second row.
Timeouts, invalid JSON or unavailable serving degrade to `UNCERTAIN` rather than
blocking or forcing an identity decision.

## Preflight

After pulling code or changing native ReID metadata code:

```bash
rm -f .runtime/camera_v2/libcamera_v2_meta.so
python scripts/setup_camera_v2_reid.py
python scripts/preflight_camera_v2_reid.py
```

The ReID preflight checks camera/room topology, synthetic Global-ID behavior,
same-camera occlusion reconnect, hard room conflict rules, crop quality, ctypes/C
ABI sizes and compiles the native DeepStream metadata bridge on the target machine.

## Run Sentinel

```bash
bash scripts/run_sentinel_vms.sh
```

The launcher runs both ReID and UI preflights before opening Sentinel.

## Important calibration note

Values in `config/reid.yaml` are conservative starting thresholds. Similarity scores
are model- and camera-domain dependent, so final deployment thresholds must be
calibrated from labeled positive pairs and hard-negative pairs captured by these
actual six cameras. Do not copy thresholds from another ReID model or site and treat
them as universal.
