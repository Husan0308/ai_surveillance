# Entrance Room — Gate 3 YOLO26m vs PeopleNet Transformer Offline A/B

**Verdict: `GATE3_PARTIAL`**

Both detectors executed through the isolated DeepStream appsrc → batch-2 legacy
`nvstreammux` → PGIE → NvDCF path on the same 300 recorded frames per camera.
The frame and source-PTS crosswalks matched, with no frames dropped before mux,
PGIE, or NvDCF. This is an execution and short-sample comparison, **not an
accuracy benchmark**: no verified person-box or identity labels exist. I do not
select a winning detector. PeopleNet's short run also reached 99% device SM,
so no longer or live GPU test was attempted.

| Gate | Result |
|---|---|
| Verified rectified input hashes and camera order | PASS |
| YOLO26m batch-2 inference + NvDCF | PASS |
| PeopleNet Transformer batch-2 inference + NvDCF | PASS |
| Same sampled source-frame crosswalk, per camera | PASS |
| Detector accuracy / labeled quality metrics | NOT VALIDATED |
| Tracker identity quality / ID switches | NOT VALIDATED |
| Same-TensorRT performance comparison | PARTIAL (frozen engines require different runtimes) |
| Production latency impact | UNMEASURED |
| Production services and assets unchanged | PASS |

## Scope and input evidence

- Worktree: `/home/apsidal/ai_surveillance-yolo-gate1`
- Branch: `experiment/entrance-yolo26m-offline-gate1`
- Base commit: `11b510d75e8c2ce08c15cb43e1726e49a503016d`
- Date: 2026-10-19
- Existing AMC-rectified clips were reused without rectifying them again. Each
  is 1920×1080, MPEG-4 Part 2, 30/1 FPS, 5,400 frames, 180 seconds:

| DeepStream source | Camera | Input | SHA-256 |
|---|---|---|---|
| 0 | CAM-02 | `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4` | `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160` |
| 1 | CAM-05 | `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4` | `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813` |

The comparison used input indices 0–299 (the first 10 source seconds) from each
camera. Both models received the exact same sequence. OpenCV's
`CAP_PROP_POS_MSEC` per-file position timestamps, carried as source-sidecar
values, matched between runs from 0 through 9.966666666 s; the crosswalk also
matched source ID, frame index and these timestamp values at mux, PGIE, and
tracker. They are not raw packet-level PTS extracted independently from the MP4
container.
This proves repeatability of the supplied-file timeline, **not synchronization
of camera exposure times**. Appsrc advertises 20 FPS while the file is 30 FPS;
frames were fed consecutively with a 50 ms pacing delay. Host feeding took
22.121 s (YOLO) and 22.211 s (PeopleNet), or about 13.6 processed frames/s per
camera. No result here is presented as 30-FPS or real-time throughput.

## Runtime and detector contracts

The same host RTX 3060, driver 595.99.02, appsrc code, 1920×1080 rectified
geometry, legacy mux, batch size 2, and stock NvDCF sample configuration were
used. The NvDCF YAML in each image has the same SHA-256:
`383002ab8628203a6921616adf46c371e261b9a76509943e4b61657f9102e86c`.
`USE_NEW_NVSTREAMMUX=no` was process-scoped. Neither MQTT, RTSP, Global ID nor
map projection was used.

| Detector | Runtime image and TensorRT | Engine / parser / labels SHA-256 | Inference settings retained |
|---|---|---|---|
| YOLO26m | DeepStream 9.1 image `nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f`; TensorRT 10.16.1.11 | Engine `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16`; parser `d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19`; labels `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` | batch 2, interval 0, 640×640, class 0 person, confidence threshold 0.25, NMS IoU 0.45, frozen raw YOLO parser. Engine-only config SHA-256 `7fc838917b87e28426ef8ddd34e69688a165e1544fe712b9f791a287a366e7c4`. |
| PeopleNet Transformer | DeepStream 9.1 image ID `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`; TensorRT 10.16.0.72 | Engine `99451ab53e3c4b8f2bc50dbd4e33cb15bfcd28ba592e2fd766adfc08770cb3a8`; parser `0e01920c57fa6762887112d5c0e0e4239e10a094cd33ed4fd580add91d08e015`; labels `5618c52db083cfdf40b81fd57d71fda73240738cefcdee7f97add8e0715737e4` | batch 2, interval 0, 544×960, class 1 person only (`filter-out-class-ids=0;2;3`), confidence threshold 0.30, top-k 20, TAO DDETR parser. Hashed engine-only config SHA-256 `88f5b514984498b4e098285a80d36d9144fdcaeb3c5fb4e2d784d09e1971b409`. |

The frozen source-config hashes are YOLO `fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113` and PeopleNet Transformer `5f3f4e177e4f1e33743bc537bfcb3e331cab87cc3912982e1732f193e4214e79`. Verified ONNX source hashes (not mounted into the engine-only runs) are YOLO `d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33` and PeopleNet `974d6a77b4fde5ad83b18bbabc5e3771220e6ee22224e98cdc0cbf733da06038`.

**Runtime confound:** PeopleNet's frozen engine did not deserialize in the YOLO
image. The same-runtime attempt logged that its plan was incompatible with
TensorRT 10.16.1.11 (see
`/tmp/entrance-yolo26m-gate3-20261019T1248Z/pnt-smoke/deepstream.stderr.log`,
lines 31–40), and no frames were sent in that failed attempt. I did not rebuild
the engine. The successful PeopleNet arm used its already-compatible
TensorRT 10.16.0.72 image. Therefore detector-specific inference settings and
runtime versions both differ; throughput/latency are observed results, not a
strict same-TensorRT benchmark. The engine-only PeopleNet test config removed
only the `onnx-file` line to forbid nvinfer's rebuild fallback; the source
config and model assets were not changed.

Both successful runs logged TensorRT engine deserialization and successful
PGIE initialization. The actual PGIE and NvDCF metadata—not just successful
pipeline startup—were captured. The custom parser outputs used the expected
person class IDs. The appsrc C++ probe code was compiled in each selected
DeepStream image and ran in the tested pipeline.

## Measured detections and local tracking

Every value below covers 300 frames per camera in the bounded sample. Counts
are model outputs, not counts of ground-truth people.

| Camera | Detector | Person detections | NvDCF person observations | Unique camera-local IDs | Confidence p50 / p95 |
|---|---|---:|---:|---:|---:|
| CAM-02 | YOLO26m | 191 | 180 | 5 | 0.4321 / 0.8730 |
| CAM-02 | PeopleNet Transformer | 903 | 894 | 3 | 0.9625 / 0.9753 |
| CAM-05 | YOLO26m | 537 | 527 | 4 | 0.6875 / 0.8887 |
| CAM-05 | PeopleNet Transformer | 901 | 894 | 3 | 0.9637 / 0.9813 |

All 300 frames per source reached each of mux, PGIE, and NvDCF for both models;
both runs report zero queue drops before mux. NvDCF IDs are local to each
camera. More detections, more tracker observations, or fewer distinct IDs do
not by themselves establish higher recall, better tracking, less fragmentation,
or fewer false positives. No ID switches, IDF1, HOTA, miss rate, precision,
recall, or F1 are claimed.

Proposal-level comparison (not ground truth): greedy one-to-one box matching
on each identical source frame found 191/191 YOLO boxes in PeopleNet proposals
at IoU ≥0.50 for CAM-02 (162 at ≥0.75), and 537/537 for CAM-05 (536 at ≥0.75).
PeopleNet emitted additional unmatched boxes. Intra-model overlapping-box
candidates at IoU ≥0.50 were 0/0 for YOLO (CAM-02/CAM-05) and 3/1 for
PeopleNet; they are **not adjudicated duplicates**. These comparisons describe
proposal overlap only and cannot identify which model is correct.

One qualitative spot-check at CAM-02 frame 120 shows three visible seated
people, three PeopleNet proposals and no YOLO proposal in the saved side-by-side
image. This is a useful warning about a possible YOLO miss, not a labeled
benchmark or an accuracy conclusion. The corresponding review image is
`/tmp/entrance-yolo26m-gate3-20261019T1248Z/review/cam-02-frame-0120.jpg`.

## Performance and resource observations

Timing probes measure host wall time between appsrc-correlated pad crossings.
The PGIE-pad duration is not an isolated CUDA kernel time; host-input-to-stage
also includes host pipe, appsrc, mux scheduling and batching. GPU telemetry was
sampled and includes device-wide activity from other processes. No Nsight
Systems/CUDA-kernel trace was captured; the profiling scope is the bounded
pipeline/run telemetry listed here.

| Measure | YOLO26m | PeopleNet Transformer |
|---|---:|---:|
| PGIE output FPS per camera | 13.598 | 13.648 |
| NvDCF output FPS per camera | 14.005 | 13.971 |
| PGIE pad wall-time p50 / p95 | 10.544 / 14.780 ms | 61.476 / 158.223 ms |
| NvDCF pad wall-time p50 / p95 | 1.085 / 2.631 ms | 3.418 / 66.977 ms |
| Host input → PGIE p50 / p95, CAM-02 | 45.967 / 63.859 ms | 98.319 / 366.763 ms |
| Host input → PGIE p50 / p95, CAM-05 | 31.457 / 46.590 ms | 84.371 / 351.410 ms |
| Appsrc initialization-to-ready | 4.396 s | 5.800 s |
| Total run wall time (300-frame arm) | 27.175 s | 29.204 s |
| Sampled device SM average / peak | 8.4% / 17% (29 samples) | 57.0% / 99% (31 samples) |
| App process SM average / peak | 5.9% / 11% (23 samples) | 76.7% / 97% (23 samples) |
| App process framebuffer average / peak | 520 / 560 MiB | 534 / 580 MiB |
| Container CPU average / peak | 63.9% / 87.48% | 64.1% / 105.00% |
| Container memory peak | 708.1 MiB | 829.8 MiB |

Before each GPU run the three-sample device baseline was 0–1% SM and
2,283–2,289 MiB / 12,288 MiB used; the visible GPU processes were the existing
STT Python process and RustDesk. The bounded PeopleNet run nevertheless reached
99% device SM (97% app-process sample), so no 180-second or live run was
started. Afterward the test container had exited, device SM was 0%, memory was
2,281 MiB, and only the pre-existing STT and RustDesk GPU processes remained.
The AMC backend remained healthy. No active six-camera AI Surveillance workload
or production latency baseline was available, so production latency impact is
**unmeasured**, not “no impact.”

The sampled utilization averages include initialization/drain and device-wide
activity and are sparse; they should not be treated as a full sustained
benchmark. The large observed PeopleNet utilization and pad-tail latency are a
resource warning, but the different TensorRT runtime prevents a clean
detector-only performance attribution.

## Accuracy review package

No verified bounding-box or person-identity ground truth was found for this
clip. The only annotation-like file found was the prior
`identity-annotation-template.csv`; it does not contain verified detector
boxes or identities. Therefore precision/recall/F1 and ID-switch metrics are
unavailable.

Created a private balanced review package from 15 frames per camera
(indices 0, 20, …, 280). It contains 30 full-resolution annotated proposal
images with both models' boxes/confidences, two contact sheets, and
`proposed_annotations.json` with empty `human_verified_annotations` and
`PENDING_HUMAN_ANNOTATION` status. YOLO proposals are green; PeopleNet proposals
are blue. These are model proposals, not labels. Directory mode is 0700 and
files are 0600; the package is not committed.

- `/tmp/entrance-yolo26m-gate3-20261019T1248Z/review/cam-02-contact-sheet.jpg`
- `/tmp/entrance-yolo26m-gate3-20261019T1248Z/review/cam-05-contact-sheet.jpg`
- `/tmp/entrance-yolo26m-gate3-20261019T1248Z/review/proposed_annotations.json`

## Harness fixes and validation

The experimental branch now has a hash-pinned PeopleNet engine-only staging
helper for the compatible frozen runtime, an opt-in offline comparison mode,
lossless bounded frame feeding, event JSONL output, per-stage host timing probes,
and a 600-frame (30 seconds at nominal feeder pacing) cap. The PeopleNet engine,
parser, labels and source config hashes are checked before staging; ONNX is
excluded from the engine-only mount to prevent an implicit plan rebuild. The
normal default PeopleNet command remains on its existing image/config/model
path. The default live pipeline behavior was not replaced.

Regression command and result:

```text
/home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m pytest -q \
  tests/test_entrance_yolo26m_gate1.py \
  tests/test_entrance_live_tracking_gate.py \
  tests/test_entrance_yolo26m_gate3.py
27 passed in 0.15s
```

Python syntax checks and `git diff --check` passed. The DeepStream C++ app
compiled and executed in both tested pinned images. No production services,
detector assets, NvDCF config, camera sources, original videos, calibration,
MQTT, production database, or RTSP streams were changed or used.

## Evidence and reproduction

Owner-private run evidence is under
`/tmp/entrance-yolo26m-gate3-20261019T1248Z/`:

- `yolo-300-sample/mux-smoke.json`, `mux-smoke-events.jsonl`,
  `deepstream.stdout.log`, `deepstream.stderr.log`, and the staged config/assets.
- `pnt-300-sample/mux-smoke.json`, `mux-smoke-events.jsonl`,
  `deepstream.stdout.log`, `deepstream.stderr.log`, and the staged config/assets.
- `yolo-300-gpu-dmon.log`, `yolo-300-gpu-pmon.log`,
  `yolo-300-docker-stats.csv`.
- `pnt-300-gpu-dmon.log`, `pnt-300-gpu-pmon.log`,
  `pnt-300-docker-stats.csv`.
- Prior PNT same-runtime mismatch: `pnt-smoke/deepstream.stderr.log`.

Equivalent bounded commands (recorded MP4 only):

```bash
python -B -m services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4 \
  --cam05-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4 \
  --input-geometry rectified --frames 300 --inference --detector yolo26m \
  --yolo-artifacts /home/apsidal/ai_surveillance-yolo-gate1/.runtime/sixcam-approved-bbox-20261006-r12-profile/models \
  --output /tmp/entrance-yolo26m-gate3-20261019T1248Z/yolo-300-sample

python -B -m services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4 \
  --cam05-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4 \
  --input-geometry rectified --frames 300 --inference --detector peoplenet --pnt-engine-only \
  --output /tmp/entrance-yolo26m-gate3-20261019T1248Z/pnt-300-sample
```

The actual runs were bounded by the test harness and used the pinned, already
available DeepStream images with `--network none`, offline files, read-only
application/model/config mounts, and no RTSP, MQTT or calibration action.

## Decision and next justified step

Both pipeline-execution gates pass, but detector accuracy and identity quality
are not validated, performance comparisons are runtime-confounded, and the
PeopleNet arm approached full GPU utilization. **No detector is declared the
winner.** The next useful step is to have a reviewer label the 30 private
sample images (with additional balanced intervals only if needed), then compute
precision/recall/F1 using a documented one-to-one IoU matching rule. Any further
GPU comparison should wait for a safe resource envelope and, for performance
attribution, compatible TensorRT engine/runtime versions. Gate 3 stops here;
no Gate 4 or deployment is authorized.
