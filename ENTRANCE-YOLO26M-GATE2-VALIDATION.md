# Entrance Room YOLO26m — Gate 2 Offline GPU Smoke

**Final verdict: GATE2_PASS.** The compatibility correction was validated in one bounded offline run on 2026-10-19. The initial incompatible-image attempt and its evidence remain below as historical context; the final results in this section supersede that earlier blocked status.

## Final Gate 2 result

| Gate | Final result | Evidence |
|---|---|---|
| `ENGINE_DESERIALIZATION` | **PASS** | DeepStream logs report `deserialized trt engine from :/models/yolo26m_raw_otm_b6_fp16.engine`, `Use deserialized engine model`, and completed engine warmup. The engine-only config and mount contain no ONNX source, preventing rebuild fallback. |
| `BATCH2_INFERENCE` | **PASS** | nvinfer initialized with `batch-size=2`; the 8 frames from each source traversed PGIE and produced object metadata. The engine profile reports min batch 1 / opt 6 / max 6. |
| `CUSTOM_YOLO_PARSER` | **PASS** | The generated config loads the frozen `libyolo26rawperson.so` and `NvDsInferParseYolo26RawPerson`; PGIE emitted class 0 `person` objects (47 total). Parser dependencies resolve in the selected image. |
| `CAM02_PGIE` | **PASS** | Source 0 / CAM-02: 8 PGIE frames, 23 person detections, complete input-frame indices 0–7. |
| `CAM05_PGIE` | **PASS** | Source 1 / CAM-05: 8 PGIE frames, 24 person detections, complete input-frame indices 0–7. |
| `NVDCF` | **PASS** | Tracker initialized and emitted 8 frames per source with local tracker metadata; 17 CAM-02 and 18 CAM-05 track observations. IDs remain local, not global. |
| `FRAME_METADATA_MATCH` | **PASS** | MUX, PGIE and tracker each emitted all 8 ordered frames per camera; source IDs and source-PTS sidecar values matched indices 0–7. No mux drops or invalid crosswalks. |
| `GPU_SAFETY` | **PASS** | Three pre-run samples: 0% device SM, 2,291 MiB / 12,288 MiB. During the ~3-second run, sampled device SM peaked at 1%; device framebuffer peaked at 2,701 MiB and returned to 2,289 MiB after shutdown. This was not a sustained-performance test. |
| `PRODUCTION_UNCHANGED` | **PASS** | No RTSP/MQTT, network was disabled for the test container, no production files/services were changed, AMC backend remained healthy, and the temporary container exited. No six-camera analytics workload was present for a production latency comparison. |
| `PEOPLE_DETECTION_QUALITY` | **NOT_VALIDATED** | The smoke verifies execution and metadata flow only; detection accuracy was not evaluated. |
| `LIVE_RTSP` | **NOT_RUN** | Explicitly out of scope. |
| `GATE3_AB` | **NOT_RUN** | Explicitly out of scope. |

### Runtime mismatch and image selection

The frozen engine build manifest is `/home/apsidal/ai_surveillance-gate3j/.runtime/sixcam-approved-bbox-20261006-r12-profile/build-manifest.json` (SHA-256 `d5374fac5f0e4db5708f084d94812aaa9d9040e21cede9607604b6dace9c4ced`). It records the exact image `nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f`.

| Local DeepStream image | Image digest / ID | TensorRT runtime | Decision |
|---|---|---|---|
| `9.1-triton-multiarch` | `fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b` | 10.16.0.72 | Rejected for this frozen plan; this is the prior incompatible runtime and remains the default PeopleNet image. |
| `9.1-samples-multiarch` | `4f80b374e4a5086552825fe0f5bdd015c8cfd3dbe430cdde5ce9572e80e01583` | 10.16.1.11 | Version-compatible, but not the image named in the engine build manifest. |
| Engine-manifest image | `10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f` | 10.16.1.11 (`libnvinfer10` 10.16.1.11-1+cuda13.2) | Selected only for experimental YOLO26m; exact manifest match and successfully deserialized the engine. |

The frozen engine SHA-256 is `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16`. Its source/build manifest records the exact selected image; the engine file was not replaced or rebuilt. The prior failure on image `fd31…` reported TensorRT 10.16.0.72 rejecting the plan. The root cause was therefore a plan/runtime version mismatch, not a demonstrated batch-2 limitation.

The selected image was verified as DeepStream 9.1.0, TensorRT Python 10.16.1.11, GStreamer app/video 1.24.2, and contained loadable `nvinfer`, `nvtracker`, and `nvstreammux`. `ldd` found no unresolved parser dependencies. The appsrc C++ program linked against this image's DeepStream/GStreamer development libraries. Host NVIDIA driver was 595.99.02.

### Staging and fail-closed changes

- Only the experimental YOLO command now selects the engine-manifest image by digest. The default PeopleNet path remains on its existing `9.1-triton-multiarch` image and PeopleNet model/config mount.
- The candidate config hash is `7fc838917b87e28426ef8ddd34e69688a165e1544fe712b9f791a287a366e7c4`. Relative to frozen config hash `fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113`, its only changes are `batch-size=6` → `batch-size=2` and removal of `onnx-file`.
- The private read-only container model mount contains only the verified engine, parser, and labels (no ONNX). This prevents nvinfer from attempting the automatic ONNX rebuild fallback if plan deserialization fails.
Frozen model asset hashes (source and staged copy where mounted):

| Asset | SHA-256 |
|---|---|
| `yolo26m_raw_otm.onnx` (validated on host; deliberately not mounted) | `d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33` |
| `yolo26m_raw_otm_b6_fp16.engine` | `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16` |
| `libyolo26rawperson.so` | `d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19` |
| `labels.txt` | `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` |
- A first run with the matching image proved deserialization and inference but exposed two harness defects: live-style leaky queues dropped frame index 5 in both sources, and JSON reporting referenced the old staged-config filename. The offline-only path now uses bounded, backpressured lossless input; live mode keeps its previous bounded/leaky behavior. The result writer uses the engine-only filename, and new validation requires exact ordered frame/PTS crosswalks at MUX, PGIE, and NvDCF.
- Regression tests also verify exact image selection, unchanged PeopleNet selection, no-ONNX fail-closed staging, hash-preserving model staging, rejection of altered staged engines, and rejection of missing/reordered frame crosswalks.

### Inputs and final run

The existing already-rectified pair was reused without rectifying again:

| Input | Mapping | SHA-256 | ffprobe |
|---|---|---|---|
| `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4` | Source 0 / CAM-02 | `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160` | MPEG-4 Part 2, 1920×1080, 30/1 fps, 5,400 frames, 180 s |
| `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4` | Source 1 / CAM-05 | `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813` | MPEG-4 Part 2, 1920×1080, 30/1 fps, 5,400 frames, 180 s |

Final command (recorded inputs only; first eight frames per camera):

```bash
timeout --signal=INT --kill-after=10s 90s \
  /home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m \
  services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4 \
  --cam05-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4 \
  --input-geometry rectified --frames 8 --inference --detector yolo26m \
  --yolo-artifacts /home/apsidal/ai_surveillance-yolo-gate1/.runtime/sixcam-approved-bbox-20261006-r12-profile/models \
  --output /tmp/entrance-yolo26m-gate2-retest-5AZMGf/gpu-smoke-lossless
```

Run interval: `2026-10-19T12:18:46Z`–`12:18:49Z`; exit code 0. The generated `mux-smoke.json` records 8 sent/muxed/PGIE/tracker frames per camera, zero queue drops, 47 class-0 `person` detector observations (23 CAM-02, 24 CAM-05) and 35 local tracker observations (17 CAM-02, 18 CAM-05). No calibration, MQTT, MV3DT/global-ID processing, RTSP, or network access occurred.

The wrapper invoked the pinned container with `--pull=never`, `--network none`, `--gpus all`, process-scoped `USE_NEW_NVSTREAMMUX=no`, and read-only app/model/config mounts. It compiled and ran the modified `deepstream_appsrc.cpp` in the same selected image.

Private evidence directory: `/tmp/entrance-yolo26m-gate2-retest-5AZMGf/` (mode 0700). Key artifacts are `gpu-smoke-lossless/mux-smoke.json`, `gpu-smoke-lossless/deepstream.stdout.log`, `gpu-smoke-lossless/deepstream.stderr.log`, `gpu-smoke-lossless/pgie-yolo26m-b2-engine-only.txt`, `gpu-runtime-preflight.log`, `deepstream-9.1-image-inventory.log`, `rerun-gpu-dmon.log`, and `rerun-gpu-pmon.log`. The staged plan/parser/labels hashes match their frozen source hashes; the source engine hash remained unchanged.

Focused regression suite: **19 passed, 0 failed** using `/home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m pytest -q tests/test_entrance_yolo26m_gate1.py tests/test_entrance_live_tracking_gate.py`. `py_compile`, `git diff --check`, and C++ compile/link against the selected image also passed.

**Scope:** Gate 2 proves isolated engine deserialization, batch-2 PGIE execution, custom-parser metadata, per-camera NvDCF flow, and exact frame attribution. It does not validate detector accuracy, long-run performance, production deployment support, Gate 3 A/B quality, or live RTSP.

## Initial Gate 2 attempt (historical, superseded)

Gate 1 remains passed. This Gate 2 attempt did not execute YOLO inference: the pinned DeepStream 9.1 TensorRT runtime rejected the frozen engine before the pipeline became ready. No Gate 3 A/B test or live RTSP test was started.

## Acceptance gates

| Gate | Result | Evidence |
|---|---|---|
| `ENGINE_DESERIALIZATION` | **FAIL** | TensorRT rejected the frozen plan as incompatible with the runtime; see the exact error below. |
| `BATCH2_INFERENCE` | **FAIL** | nvinfer did not initialize and no inference frames were produced. Batch-2 engine/profile compatibility therefore remains untested. |
| `CUSTOM_YOLO_PARSER` | **NOT_RUN** | The staged config referenced the frozen parser, but no PGIE execution reached parser output. |
| `CAM02_PGIE` | **NOT_RUN** | The application did not signal `READY`; the harness sends no frames before readiness. |
| `CAM05_PGIE` | **NOT_RUN** | Same shared PGIE initialization failure; no frames were sent. |
| `NVDCF` | **NOT_RUN** | The tracker library initialized, but it received no frames and emitted no tracker metadata. |
| `FRAME_METADATA_MATCH` | **NOT_RUN** | No appsrc frames reached nvstreammux; no source/frame/PTS crosswalk was produced. |
| `GPU_SAFETY` | **PASS** | Baseline was 0% GPU SM across three samples, 2,293/12,288 MiB used. During the bounded failed initialization, device peak was 71% SM / 49% memory-controller; the test process peaked at 60% SM / 41% memory-controller and 374 MiB framebuffer. After shutdown, utilization and memory returned to baseline. |
| `PRODUCTION_UNCHANGED` | **PASS** | No RTSP, production model/config/service, project or database changes. No test container/process remained. AMC backend remained healthy. |
| `PEOPLE_DETECTION_QUALITY` | **NOT_VALIDATED** | No detections were produced. |
| `LIVE_RTSP` | **NOT_RUN** | Offline-only authorization. |
| `GATE3_AB` | **NOT_RUN** | Explicitly out of scope. |

## Preflight and assets

- Worktree: `/home/apsidal/ai_surveillance-yolo-gate1`
- Branch and commit: `experiment/entrance-yolo26m-offline-gate1` at `1d29d8d40fe569b7e48484c3fb2f101387419bae`; this is still the remote branch tip.
- The focused Gate 1 suite was rerun with `/home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m pytest -q tests/test_entrance_yolo26m_gate1.py tests/test_entrance_live_tracking_gate.py`: **15 passed**.
- The source frozen PGIE config hash remained `fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113`.
- Batch-2 staged config: `/tmp/entrance-yolo26m-gate2-preflight-20261019T115631Z/pgie-yolo26m-b2.txt`, SHA-256 `e0f3d0c6f7e5acc54b00ea764e4d368ef78bf464700c22582b4aa9a246111756`. Its only difference from the frozen config was `batch-size=6` → `batch-size=2`.
- Four source assets were copied (not moved) into the worktree's ignored, owner-only `.runtime` model directory. Source and copied hashes matched both before and after the smoke:

| Asset | SHA-256 |
|---|---|
| `yolo26m_raw_otm.onnx` | `d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33` |
| `yolo26m_raw_otm_b6_fp16.engine` | `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16` |
| `libyolo26rawperson.so` | `d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19` |
| `labels.txt` | `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a` |

The labels file has 80 entries and `person` is class index 0, as recorded in Gate 1. The asset directory is mounted read-only inside the container.

## Input media

The preserved offline MV3DT run report identifies the pair as project `20261018_080650_1744`, with `cam_00 = CAM-02` and `cam_01 = CAM-05`. It records that the project rectified CAM-02 video had 5,404 frames and CAM-05 had 5,400 frames; for the prior equal-duration MV3DT run, a separate stream-copy of CAM-02 was trimmed by four tail frames. Gate 2 used that existing rectified pair and did not rectify it again.

| Input | Camera | SHA-256 | Codec | Geometry | Rate / frames / duration |
|---|---|---|---|---|---|
| `.../dataset/videos/cam_00.mp4` | CAM-02 | `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160` | MPEG-4 Part 2 (`mp4v`) | 1920×1080, yuv420p | 30/1 FPS, 5,400 frames, 180.000 s |
| `.../dataset/videos/cam_01.mp4` | CAM-05 | `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813` | MPEG-4 Part 2 (`mp4v`) | 1920×1080, yuv420p | 30/1 FPS, 5,400 frames, 180.000 s |

Both ffprobe reports were checked. The clips are already in AMC-rectified image coordinates; no second rectification was applied. The previous report documents full-duration offline processing and decoding. This 8-frame smoke did not claim cross-camera exposure synchronization or calibration accuracy.

## Run and failure evidence

The pinned image was already present locally:

`nvcr.io/nvidia/deepstream@sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`
Image ID: `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`.

The one bounded command was equivalent to:

```bash
timeout --signal=INT --kill-after=10s 90s \
  /home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m \
  services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4 \
  --cam05-video /home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4 \
  --input-geometry rectified --frames 8 --inference --detector yolo26m \
  --yolo-artifacts /home/apsidal/ai_surveillance-yolo-gate1/.runtime/sixcam-approved-bbox-20261006-r12-profile/models \
  --output /tmp/entrance-yolo26m-gate2-run-64nPcJ
```

The wrapper used the pinned image with `--pull=never`, `--network none`, and read-only source, config, and model mounts. It ran from 2026-10-19 11:57:50Z to 11:58:27Z (about 37 seconds). The harness waited its bounded 30 seconds for `READY`, then stopped and returned exit code 1. Since input sending starts only after `READY`, **zero of the requested 8 frames per camera were sent**.

`deepstream.stderr.log` gives the root cause:

```text
ERROR: [TRT]: IRuntime::deserializeCudaEngine: Error Code 6: API Usage Error
(The engine plan file is not compatible with this version of TensorRT,
expecting library version 10.16.0.72 got [non-printable/malformed version bytes],
please rebuild.)
ERROR: ... Deserialize engine failed from file: /models/yolo26m_raw_otm_b6_fp16.engine
... deserialize engine ... failed
... deserialize backend context ... failed, try rebuild
... Trying to create engine from model files
```

Thus the failure is an explicit TensorRT plan/runtime version mismatch; it is **not evidence that batch-2 itself is unsupported**. The engine's serialized version bytes are not legible in the captured diagnostic, so the originating TensorRT version cannot be named from this log. nvinfer automatically began its ONNX rebuild fallback. The test did not authorize or wait for a replacement engine: initialization was stopped at the readiness timeout, model assets were mounted read-only, and the engine hash remained unchanged in both source and staging locations. No `READY`, `MUX`, `PGIE`, `TRACK`, or `DONE` records were produced. The GStreamer scanner also emitted warnings for optional unrelated audio/media plugins; nvinfer's TensorRT incompatibility was the terminal blocker.

Evidence directory (owner-private under `/tmp`): `/tmp/entrance-yolo26m-gate2-run-64nPcJ/`

- `deepstream.stderr.log`, `deepstream.stdout.log`
- `runner.stderr.log`, `runner.stdout.log`
- `gpu-dmon.log`, `gpu-pmon.log`
- `pgie-yolo26m-b2.txt`

The DeepStream process/container exited; `docker ps` showed no test container, and GPU utilization returned to 0%. The only visible GPU processes after the run were pre-existing RustDesk and STT processes.

## Production and repository status

Before the run, no AI Surveillance API/ML listener was present on ports 8100/8101, and no six-camera DeepStream production process/container appeared in the process or GPU inventory. Therefore a production analytics latency baseline was unavailable; no claim is made about production latency impact. The isolated test used recorded files only and no MQTT/global ID, map projection, live camera, or production database.

The AMC backend container remained healthy; the pre/post container list was unchanged. The original r12 assets and copied hashes match. The tracked worktree remains at commit `1d29d8d40fe569b7e48484c3fb2f101387419bae`; no code fix was made, so no commit or push was performed. Only the requested ignored runtime asset copies and this report were added in the isolated worktree.

## Next safe action

Do not rerun this engine or permit nvinfer's rebuild fallback. First obtain an engine explicitly built/serialized for the TensorRT runtime inside the pinned DeepStream 9.1 image (and a batch profile that supports batch 2), or an approved compatible runtime matching the frozen engine. Any candidate engine must be a new isolated artifact with its own hash. After that is authorized and validated, repeat only the bounded Gate 2 smoke. This report does not authorize Gate 3 or live RTSP.
