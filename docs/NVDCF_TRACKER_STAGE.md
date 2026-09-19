# NvDCF per-camera tracker stage

Branch: `rebuild/deepstream91-nvdcf-v1-20260919`

Starting point: frozen YOLO26m detector stage after the 660-second soak and
six-camera detector isolation/recovery gates passed.

## Scope

This stage adds only per-camera multi-object tracking:

```text
6 RTSP
 -> NVDEC/NVMM
 -> nvstreammux(batch=6)
 -> nvinfer(YOLO26m raw OTM, FP16, interval=0)
 -> DeepStream NMS(conf=0.25, IoU=0.45)
 -> nvtracker(NvDCF_stable_person, 960x544, GPU, batch processing)
 -> tiler
 -> nvdsosd(person ID=<object_id>)
 -> NVENC/MKV + live ffplay
```

Not enabled in this stage:

- cross-camera ReID;
- face recognition;
- global identity merge;
- pose;
- heatmap.

The low-level tracker is
`/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so`
with the pinned DeepStream image's
`config_tracker_NvDCF_stable_person.yml`. The perf profile is deliberately used instead
of the accuracy profile so this milestone does not pull the Re-ID model into the
tracker stage. Tracker input resolution is 960x544; both values are multiples of
32.



## DeepStream 9 tracker property API

DeepStream 9 uses batch processing exclusively in Gst-nvtracker. The old
`enable-batch-process` and `enable-past-frame` GStreamer properties are not
set by this stage. Runtime preflight inspects the installed `nvtracker` plugin
and verifies the expected DeepStream 9 properties before opening cameras.



### GPU-enabled plugin preflight

The `gst-inspect-1.0 nvtracker` preflight runs in the same pinned DeepStream
image with GPU 0 exposed and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`. Without NVIDIA runtime
driver injection, DeepStream plugins can fail to load because `libcuda.so.1`
is unavailable, which makes `gst-inspect` incorrectly report that
`nvtracker` does not exist.

## Evidence

The tracker probe runs after `nvtracker`. It writes
`.runtime/<run>/tracks.jsonl` with source ID, frame, PTS, tracker object ID,
detector confidence, NvDCF tracker confidence, and tracked bounding box.

`CAM-XX TRACK` telemetry reports:

- tracked frames;
- output person objects;
- untracked objects;
- duplicate IDs within one frame;
- unique IDs seen on that source.

The checker also preserves the detector gates: source FPS/queue/error counters,
YOLO parser health, NMS overlap evidence, recording validation, RSS, and
process-specific GPU memory.

## 60-second gate

Run:

```bash
python3 -m unittest tests.test_yolo26m_tracker_gate -v

python3 scripts/validate_yolo26m_tracker.py \
  --duration 60 \
  --out .runtime/yolo26m-nvdcf-60s-v1

python3 scripts/yolo26m_tracker/check_gate.py \
  .runtime/yolo26m-nvdcf-60s-v1
```

The first checker run is expected to remain BLOCKED only for the separate visual
review if automated evidence is clean.

Visually verify:

- all six tiles advance;
- one continuously visible person keeps the same ID;
- nearby people retain separate IDs;
- no obvious rapid ID flicker/switching;
- tracked boxes stay aligned.

Then:

```bash
python3 scripts/yolo26m_tracker/mark_visual_review.py \
  .runtime/yolo26m-nvdcf-60s-v1 \
  --pass-review \
  --notes "6 cameras checked; continuous people keep stable IDs; nearby people remain separate."

python3 scripts/yolo26m_tracker/check_gate.py \
  .runtime/yolo26m-nvdcf-60s-v1
```

Do not begin ReID or any later stage until this tracker gate passes.


## ID-switch tuning after first visual gate

The first clean 60-second NvDCF run was not accepted because human visual
review observed the same continuously visible person receiving a new tracker ID.
This is a real tracker-stage failure even though object metadata had no duplicate
IDs within individual frames.

The active low-level config is now
`config/deepstream/config_tracker_NvDCF_stable_person.yml`. ReID remains
explicitly disabled (`reidType: 0`). Compared with the initial balanced
profile, this retry uses cascaded association, longer shadow retention, HOG plus
ColorNames visual features, and a larger DCF feature image. The intent is to
reduce fragmentation before adding any cross-camera/ReID model.

Key active settings:

```yaml
minTrackerConfidence: 0.15
maxShadowTrackingAge: 90
associationMatcherType: 1
minMatchingScore4SizeSimilarity: 0.5
minMatchingScore4Iou: 0.05
minMatchingScore4VisualSimilarity: 0.6
matchingScoreWeight4VisualSimilarity: 0.7
useColorNames: 1
useHog: 1
featureImgSizeLevel: 3
reidType: 0
```

This retry must again pass the 60-second automated gate and a separate visual
review. In particular, a continuously visible person must not receive a new ID
without a genuine disappearance/re-entry.
