# YOLO26m person detection on DeepStream 9.1

Six-camera transport baseline: **PASS**. The starting checkpoint is
`5db3745` on `rebuild/deepstream91-cam01-cam02-cam03-cam04-cam05-cam06-v1-20260918`.
This work is on `rebuild/deepstream91-yolo26m-person-v1-20260919`.
Current AI gate: YOLO26m DeepStream person detection — **BLOCKED** after the first 60-second run.
Tracker/ReID/face/pose/heatmap: **NOT STARTED**.

## Preserved transport and new processing

The existing native runner and launcher have an optional detection build. Their
camera-only defaults remain available. Source IDs 0–5, RTSP TCP/100 ms,
NVDEC/NVMM, queues, native reconnect/isolation code, mux settings, 3-row × 2-column
tiler, NVENC/MKV and UDP/ffplay transport are preserved. The detection launcher
requires preview availability and refuses a latency other than 100 ms.

```text
6 real RTSP -> nvurisrcbin/NVDEC -> NVMM -> existing queues
 -> nvstreammux(batch=6)
 -> nvinfer(YOLO26m, FP16, interval=0)
 -> nvmultistreamtiler(3 rows, 2 columns, 1920x1620)
 -> nvvideoconvert(RGBA/NVMM) -> nvdsosd(GPU)
 -> existing nvvideoconvert(NV12/NVMM) -> NVENC
 -> existing MKV + UDP MPEG-TS -> automatically opened ffplay
```

The two added pad probes access buffer timestamps and DeepStream metadata only.
They do not map pixels, run Python inference, or use OpenCV. Python handles model
export, process supervision and text telemetry; deployed inference is entirely
`nvinfer`/TensorRT. The display shows `person <confidence>` without tracker IDs.

## Exact model/export

The local COCO detection weights are `yolo26m.pt`, copied from
`/home/apsidal/ai_surveillance_old_good/yolo26m.pt`. The export validates detect
task, model scale `m`, 80 COCO labels, person index 0 and the one-to-one head.
The model is not retrained or changed to a one-class network.

```bash
.venv/bin/python scripts/yolo26m_person/export_model.py \
  --weights /home/apsidal/ai_surveillance_old_good/yolo26m.pt \
  --out .runtime/models/yolo26m
```

Recorded exporter versions: Ultralytics 8.4.124, PyTorch 2.13.0+cu126,
ONNX 1.22.0. The exact export call uses `format=onnx`, `imgsz=640`, `batch=6`,
`dynamic=True`, `simplify=True`, `opset=17`, `nms=False`, `end2end=True`,
`agnostic_nms=True`, `max_det=300`, `half=False`, `device=cpu`.
After export, the spatial input dimensions are fixed to 640; batch stays dynamic.
The exported model has input `(batch,3,640,640)` and output `(batch,300,6)`.
The graph is checked and contains no `NonMaxSuppression` operator.

The installed exporter implements `agnostic_nms=True` in the one-to-one head as
one maximum-scoring class per candidate before top-k selection. This does not
introduce IoU suppression. Live model evaluation does not use Ultralytics.

## Parser/config and geometry

[Parser source](../scripts/yolo26m_person/parser.cpp) builds against the actual
DeepStream 9.1 `nvdsinfer_custom_impl.h` ABI. Per-frame input must be FLOAT
`(300,6)` with 1,800 elements. Rows are network pixel coordinates
`[x1,y1,x2,y2,confidence,class_id]`. The parser rejects malformed tensors;
rejects nonfinite values, unsafe/nonintegral class IDs, invalid confidence and
zero/negative boxes; clamps coordinates to 640×640; and returns only class 0.
The runtime fails if the parser reports malformed/rejected data. Other valid
COCO classes are intentionally filtered without being treated as errors.
All object fields are zero-initialized, including DeepStream 9.1 rotation angle.
No NMS runs in the parser or DeepStream clustering.

[Primary GIE config](../config/deepstream/config_infer_primary_yolo26m.txt)
uses `process-mode=1`, `network-type=0`, `gie-unique-id=1`, `gpu-id=0`,
`batch-size=6`, `interval=0`, `network-mode=2`, `cluster-mode=4`,
`maintain-aspect-ratio=1`, `symmetric-padding=1`, RGB and scale 1/255.
The threshold is `pre-cluster-threshold=0.25`; top-k is 300.

Geometry is inverted exactly once: the parser returns network coordinates;
DeepStream's installed `gstnvinfer_meta_utils.cpp` subtracts its padding offsets
and divides by its scale ratios to restore mux coordinates (2560×1440).
The tiler then maps those metadata rectangles to the correct output tile.
No hand-tuned scale/offset multipliers are applied.

## Build on the deployment GPU

```bash
python3 scripts/yolo26m_person/build.py
```

The pinned DeepStream samples image has SDK/TensorRT headers but lacks CUDA
runtime development headers. The build helper extracts NVIDIA's matching
`cuda-cudart-dev-13-2` and `cuda-crt-13-2` 13.2.51-1 packages under ignored
`.runtime/build-deps/cuda13.2`; it does not install or replace host/runtime
packages. It compiles the parser `.so` and runs a native synthetic tensor unit
test inside the pinned image. Synthetic tensors are used only for unit testing,
never as camera frames or runtime detection evidence.

`trtexec` in that same container builds with `--fp16`, minimum batch 1,
optimum/maximum batch 6, input 3×640×640, and 2,048 MiB workspace. The dynamic
batch range supports the frozen mux's partial batches and source isolation.
The FP16 engine retains FLOAT input/output bindings for preprocessing/parsing.
The model's internal eligible layers use FP16; TensorRT may retain FP32 where
required. No engine from an older GPU/runtime is reused.

Artifacts live under `.runtime/models/yolo26m/`: ONNX, labels, parser, engine,
export manifest, engine manifest and build logs. The runtime checks artifact
SHA256s and deployment GPU UUID/driver/image provenance before opening cameras.
The engine is mounted read-only during the camera gate to avoid silent rebuilds.
All model binaries and runtime artifacts are ignored by Git.

## Short gate and evidence

```bash
python3 -m unittest tests.test_cam_six_stability tests.test_yolo26m_gate -v
python3 scripts/validate_yolo26m_person.py \
  --duration 60 --out .runtime/yolo26m-person-short
python3 scripts/yolo26m_person/check_gate.py .runtime/yolo26m-person-short
```

`CAM-XX STATS` is source FPS/packet/queue/decode telemetry. `CAM-XX DETECT` counts
inferred frames and person detections (summed across frames, not unique people),
batches containing that source, errors and last detection Unix timestamp.
`YOLO STATS` reports total batch calls, inferred frames, persons, parser status
and element residence latency. That latency includes preprocessing, scheduling,
inference and parsing; it is not a pure GPU kernel timing measurement.

`detections.jsonl` samples object metadata once per 20 source frames.
`gpu.csv` is device-wide utilization/VRAM; native process CPU is percent of one
logical core and excludes ffplay. `preview.json` records preview process health.
`gate.json` checks all six sources, >60 s runtime, errors/loss/PTS, queue growth,
inference progression, person-only metadata, parser health, near-identical box
pairs, GPU/memory evidence and finalized recording. A telemetry PASS also
requires separate visual review of real person boxes and the live preview.

The camera-only reference measured about 6×20 FPS, 1,487 MiB VRAM,
22.84% of one CPU core, and 25.98% NVDEC. Measured results follow below. The detector gate did not pass; no longer AI soak
or additional AI feature was started.

## First live gate result — 2026-09-19: BLOCKED

The process ran 60.924 seconds, loaded the local engine through `nvinfer`, and
exited normally. All six cameras decoded in NVDEC/NVMM. The native parser tests,
nine existing camera evidence tests and five new detector evidence tests passed.
The native detector executable compiled successfully inside the pinned container.

| Camera | Source FPS mean (five-second range after startup) | Inferred frames | Person detections summed over frames |
| --- | --- | --- | --- |
| CAM-01 | 20.000 (19.797–20.200) | 1,204 | 1,893 |
| CAM-02 | 20.020 (19.997–20.197) | 1,203 | 2,455 |
| CAM-03 | 20.000 (19.797–20.203) | 1,204 | 909 |
| CAM-04 | 19.980 (19.797–20.200) | 1,211 | 2,489 |
| CAM-05 | 20.000 (19.797–20.200) | 1,211 | 1,399 |
| CAM-06 | 20.000 (19.800–20.200) | 1,212 | 2,633 |

Total: 7,245 inferred frames in 1,360 delivered batches and 11,778 person
metadata detections. Native element residence latency averaged 17.682 ms,
maximum 81.685 ms including startup. The tiled recording has 1,359 packets
and 60.343 seconds of video. Source packet loss/late counters, PTS reversals,
runtime warnings/errors, parser errors and parser rejected-row counts were zero.
Sampled source queue occupancy remained zero.

GPU utilization averaged 41.7% (33–51%). Device VRAM was constant at 1,895 MiB.
NVDEC averaged 27.1%, NVENC 9.3%. Native CPU averaged 36.14% of one logical core;
RSS was 1,160.000 to 1,161.734 MiB after startup. This is compared with the
previous camera-only baseline, not a simultaneous controlled benchmark; GPU
memory includes other desktop processes.

**Blocking evidence:** CAM-04/source 3, frame 400, PTS 20,323,479,367 ns
contained two class-0 boxes for the same visible person. Their confidences were
0.437744 and 0.552734. Mux-space `(left,top,width,height)` values were
`(1797,228.5,237,173.5)` and `(1802,229,232,171.5)`, IoU **0.967619**.
The checker found this pair among 597 sampled metadata rows; this sampling does
not establish that other frames were duplicate-free. The one-to-one/agnostic
export and absence of NMS were verified; these observations do not justify
claiming duplicate-free detector output. The user's fail-closed requirement
therefore leaves this gate BLOCKED. No deduplication/NMS, threshold change,
transport adjustment or further camera run was used to conceal the failure.

The live ffplay window visibly displayed real person boxes with plausible
coordinates/confidences and correct tile placement. A continuously decoded
50-second recorded frame was also visually inspected. Initial input-side seeking
(`ffmpeg -ss ... -i ...`) produced green block artifacts in extracted stills;
decoding from the start produced clean frames. This was a seek-related diagnostic
artifact, not evidence of a corrupt live overlay. The frozen encoder/recording
settings were left unchanged. To inspect the recording, play from the start or
seek after opening the input (decode up to the requested time):

```bash
ffmpeg -v error -i .runtime/yolo26m-person-short/CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv \
  -ss 50 -frames:v 1 -q:v 4 .runtime/yolo26m-person-short/review.jpg
```

Engine build: 323.766 seconds, 42.7045 MiB, RTX 3060, driver 595.91.07,
TensorRT 10.16.1.11, CUDA runtime 13.2, FP16, profile min/opt/max batch 1/6/6.

- Weights SHA256: `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`
- ONNX SHA256: `2bd2b6fb247b34adae93b25e8504c7f020089ca4f83a44be3473e428e29b39a1`
- Engine SHA256: `ce217ebcec3e222b4974c5e5ddd42158af72f959752ceb6318ccd2812983e3c1`

Warnings were limited to the image's existing optional plugin-scanner dependencies
and TensorRT's weakly-typed-network deprecation notice during FP16 build.
The deliberate malformed-tensor unit test prints an expected `PARSER_ERROR`;
the live parser reported zero errors. No OOM, CPU inference fallback, or runtime
GStreamer errors occurred.

Local, Git-ignored evidence:

- [Gate measurements](../.runtime/yolo26m-person-short/gate.json)
- [Separate visual review / duplicate coordinates](../.runtime/yolo26m-person-short/visual_review.json)
- [Actual live ffplay capture](../.runtime/yolo26m-person-short/preview-live.jpg)
- [Clean continuously decoded frame](../.runtime/yolo26m-person-short/overlay-linear-50s.jpg)
- [Tiled recording](../.runtime/yolo26m-person-short/CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv)
- [Per-object metadata](../.runtime/yolo26m-person-short/detections.jsonl)
- [Runtime log](../.runtime/yolo26m-person-short/pipeline.log)
- [Engine provenance](../.runtime/models/yolo26m/engine.json)
- [Export provenance](../.runtime/models/yolo26m/export.json)

The validator/preview have stopped and transient camera credentials were removed.
The frozen checkpoint branch still points at `5db3745`; implementation changes
remain uncommitted on the new detection branch. RTSP recovery code was preserved
but isolation tests were not repeated with inference after this blocked gate.


## Raw one-to-many + DeepStream NMS retry gate

The next detector retry uses the YOLO26 one-to-many raw output
`(batch,84,8400)` and leaves suppression to Gst-nvinfer. The active config is
`config/deepstream/config_infer_primary_yolo26m_raw_otm.txt` with
`cluster-mode=2`, `nms-iou-threshold=0.50`, and
`pre-cluster-threshold=0.25`. The parser emits only person proposals and does
not perform its own NMS.

Before cameras open, runtime validation fails closed unless that exact DeepStream
NMS configuration and the raw one-to-many engine/parser are selected.

The final metadata checker evaluates every saved person box pair from the same
camera/frame. It records pairs above IoU 0.50, pairs at or above 0.90 and 0.95,
the maximum person IoU, and the worst surviving pair. Any final pair above IoU
0.70 blocks the gate because DeepStream NMS should have rejected the
lower-confidence proposal.

Evidence is written to `overlap_evidence.json` beside `gate.json`. This rule
is intended to prevent the previous CAM-04 double-box failure from being
accepted.



### Duplicate-box threshold adjustment after long-run visual review

A later 660-second raw one-to-many run still showed visually duplicated person
boxes while the maximum surviving same-frame IoU was exactly 0.70. The previous
0.70 NMS setting therefore proved too permissive for this office scene. The
active retry threshold is now 0.50, while confidence remains 0.25. This change
targets overlapping duplicate proposals rather than hiding them by raising the
confidence threshold. Acceptance still requires visual review so legitimate
nearby people are not accidentally collapsed by overly aggressive suppression.

## References

- [Ultralytics NMS-free detection](https://docs.ultralytics.com/guides/end2end-detection)
- [NVIDIA DeepStream 9.1 Gst-nvinfer](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvinfer.html)

Runtime behavior and API details were also checked against installed exporter
source and the headers/source in the pinned deployment image.
