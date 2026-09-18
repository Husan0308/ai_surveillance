# CAM-01 camera-only validation

Scope: only the existing enabled `CAM-01` from `config/cameras.yaml`. Credentials
come from the existing shared loader and `.env` precedence. CAM-02 through CAM-06
are never opened. This is a dedicated diagnostic executable, not another
production service. No API/frontend or AI components are imported or started.

## Platform and endpoint

The exact NVIDIA image digest is in `config/deepstream-platform.json`:
DeepStream 9.1.0, CUDA runtime 13.2, TensorRT package 10.16.1.11. Host driver
595.91.07 and RTX 3060 12288 MiB were validated after reboot.

CAM-01 endpoint: `rtsp://192.168.1.210:554/Streaming/Channels/101`.
Password is deliberately omitted. Raw TCP/DESCRIBE/ffprobe verified Digest
authentication, H.264, 2560×1440, 20/1 FPS. No URL or credential changes were made.
The original per-camera latency remains 20 ms in the saved configuration; this
separate stability profile uses 300 ms. It does not inherit old frame-dropping
or inference settings.

## Run the camera-only diagnostic

From the repository root, with the existing Python YAML/dotenv dependencies,
Docker GPU access and the pinned image already available:

```bash
# Verify the explicit RTSP -> H.264 depay/parser -> NVDEC -> NVMM path first:
python3 scripts/validate_cam01.py --mode decode --duration 20 --out .runtime/cam01-decode

# Real video output and uninterrupted 11-minute soak:
python3 scripts/validate_cam01.py --mode record --duration 660 --out .runtime/cam01-stability

# Check recorded frames, runtime health and the measured steady-state interval:
python3 scripts/cam01_validation/check_stability.py .runtime/cam01-stability
```

Choose a fresh `--out` directory to retain previous evidence. Reusing an output
directory replaces its recording and telemetry. The launcher holds the existing
camera GPU-owner lock and uses one fixed container name, preventing two instances
of this validator. It compiles the native diagnostic inside the pinned container;
no host SDK/compiler compatibility changes or new host packages are needed.

The runtime graph is:

```text
CAM-01 RTSP over TCP
  -> nvurisrcbin [rtspsrc -> H.264 depay/parser -> nvv4l2decoder]
  -> NVMM (CUDA device memory)
  -> bounded, non-leaky queue
  -> nvstreammux (batch-size=1)
  -> nvvideoconvert (NVMM/NV12)
  -> nvv4l2h264enc (NVENC, diagnostic output only)
  -> h264parse -> matroskamux -> CAM-01.mkv
```

No `nvinfer`, tracker, software decoder, OpenCV/NumPy, pixel mapping, or Python
frame callback is in this graph. Native probes count buffer headers/timestamps
only. Python launches Docker, reads text logs and samples nvidia-smi every five
seconds. Writing compressed encoded output to disk is intentional; decoded video
surfaces stay in GPU memory. TensorRT exists in the NVIDIA image but no inference
engine/model is created.

Source settings: TCP; 300 ms jitter latency; drop-on-latency=false; all frames
decoded; four extra decoder surfaces; default RTP/RTCP timestamp handling;
10-second TCP timeout; native source reconnect timeout 5 seconds; unlimited native
attempts. Queues are bounded at 12 buffers, non-leaky. Hard source errors/EOS have
a delayed same-process restart fallback; a 90-second absence of advancing output
fails the diagnostic cleanly. Authentication rejection fails visibly. Shared
pipeline errors terminate with nonzero status.

Mux settings: batch-size=1, width=2560, height=1440, live-source=true,
batched-push-timeout=50000 microseconds, sync-inputs=false, attach-sys-ts=true,
nvbuf-memory-type=2 (CUDA device), cache-buffer=false, drop-pipeline-eos=true.
Intentional recording completion sends EOS directly to the output encoder so the
live source's EOS/reconnect policy does not suppress file finalization.

## Real video evidence and display limitation

The preferred EGL window was tested. This desktop reports `llvmpipe`,
`Accelerated: no`, and zero XRandR providers. Container EGL emitted DRI3 and
repeated X11 shared-memory errors. That run was stopped and excluded from
acceptance. Only the output branch was switched to NVENC recording; the RTSP,
NVDEC and one-source DeepStream mux path stayed the same.

The recorded real video is `.runtime/cam01-stability/CAM-01.mkv`. Open it in a
video player after the test. Preview frames and machine-readable telemetry are
saved in the same directory. Offline ffmpeg extraction of preview images is
only diagnostic; it is not part of the live processing path. The smoke-test
preview visibly showed an office, the camera's `dev-room-2` label and timestamp.

`--mode display` remains available for a working accelerated X11 session. It
uses an authenticated Xauthority mount and a CAM-01 titled window; it does not
weaken X-server access with `xhost +`. Display mode is not validated on this
software-rendered desktop. The current validated output is the recording.

## Reconnect test

In one terminal:

```bash
python3 scripts/validate_cam01.py --mode record --duration 100 --out .runtime/cam01-reconnect
```

While that instance is running, in another terminal:

```bash
python3 scripts/cam01_validation/test_reconnect.py --out .runtime/cam01-reconnect
```

The test waits for 200 output frames, verifies the dedicated validator container
and its private bridge network, disconnects only that container for 12 seconds,
and restores the network in `finally`. It never changes camera/NVR configuration
or host-wide firewall/network state. It requires resumed frame progression and
the same container ID/host PID. A separate uninterrupted soak follows; recovery
gaps are never counted toward continuous stability.

Observed reconnect evidence is `.runtime/cam01-reconnect/reconnect.json`:
network interruption produced zero FPS and a 9.055 s sampled frame age; native
source reconnection resumed in the same PID, returning to 20.2 FPS. The last
pre-outage to first post-outage frame gap was 17.852 s. The automated check
confirmed at least 100 new frames at normal rate within 12.022 s after network
restoration. The process then ran through 100 s and exited cleanly. Five source
warnings and transient NVIDIA allocator `Bad file descriptor` cleanup diagnostics
occurred during native decoder teardown; they stopped after recovery. They are
retained in the reconnect log, not hidden or counted as a clean-soak pass.

The pinned image also emits startup plugin-scanner warnings for unused optional
audio/media, Intel MSDK, UCX, Triton and Rivermax dependencies. Required camera
plugins load successfully. These startup scanner warnings are distinct from the
pipeline bus warning/error counters and remain visible in the saved logs.

## Acceptance evidence

`pipeline.log` contains native frame/PTS/age counters, queue occupancy, process
CPU/RSS, RTP loss/late counters, errors/warnings and a clean completion marker.
`gpu.csv` contains device-wide GPU/NVDEC/NVENC utilization and used VRAM. CPU is
percent of one logical CPU core; GPU VRAM includes all processes on that GPU.
Filesink `rendered` counts sink buffers, not video frames; the `output` field
counts encoded H.264 access units, checked against recorded packet count.

The offline checker requires at least 600 seconds between telemetry samples
after the first 30 seconds of warmup. It rejects frozen counters/PTS, missing
samples, source errors/reconnects during the clean soak, packet loss/late packets,
frame drops, sustained FPS outside 18–22, frame age over 1 s, gaps over 1 s,
queue backlog, decode/output divergence and missing finalized recorded video.
It compares early/late steady-state RSS and CPU and limits VRAM spread; these
10-minute observations cannot establish absence of leaks over days of operation.

The final numerical result is written to
`.runtime/cam01-stability/stability.json`. A telemetry PASS alone does not replace
the separate RTSP/authentication, real-frame visual and reconnect evidence.
The camera pass stops at CAM-01; it never advances to another camera or AI stage.

## Final measured result — 2026-09-18: PASS

The validator exited with status 0 and `hardware_decoder=1 fatal=0`. The evidence
checker returned PASS with no failures. Total measured runtime was 660.032 s,
including 629.999 s of measured steady operation after warmup. The finalized
H.264 recording is 659.456 s long with 13,189 packets/frames, matching the native
output counter exactly.

| Measurement | Observed result |
| --- | --- |
| Steady FPS | Mean 20.002; five-second samples 19.602–20.203 |
| GPU compute utilization | 0–1% |
| NVDEC utilization | Mean 4.22%; 2–9% |
| NVENC utilization | Mean 4.39%; 0–17% |
| Device-wide used VRAM | Constant 321 MiB after warmup |
| Native process CPU | Mean 4.74%, peak 5.58% of one logical core |
| Native process RSS | 275.426 to 275.547 MiB; early/late median growth 0.066 MiB |
| Sampled queue occupancy | 0 buffers throughout steady interval |
| Maximum observed output gap | 371.401 ms |
| Runtime bus warnings/errors, reconnects | 0 / 0 / 0 |
| RTP lost/late, sink drops, duplicate/backward PTS | All 0 |

The early preview (`start.jpg`, recording offset 30 s) visibly shows camera
`dev-room-2` at 17:41:54. The late preview (`end.jpg`, offset 630 s) shows the same
camera at 17:51:54 with changed real scene content. Both were visually inspected.
No synthetic source or inference elements were used. The separate interruption
and recovery test also passed, as detailed above.

Local evidence:

- [Full real CAM-01 recording](../.runtime/cam01-stability/CAM-01.mkv)
- [Early frame](../.runtime/cam01-stability/start.jpg) and [late frame](../.runtime/cam01-stability/end.jpg)
- [Stability measurements](../.runtime/cam01-stability/stability.json)
- [Pipeline log](../.runtime/cam01-stability/pipeline.log) and [GPU samples](../.runtime/cam01-stability/gpu.csv)
- [Reconnect measurements](../.runtime/cam01-reconnect/reconnect.json)

The six offline acceptance-checker tests passed. The validator stopped after this
CAM-01 pass; other cameras and AI stages were not started. Runtime evidence is
local and ignored by Git, so these evidence links require this workspace.

## Changes

- `scripts/validate_cam01.py`: CAM-01 selection, pinned container/build, lock,
  protected transient credentials, log redaction and GPU sampling.
- `scripts/cam01_validation/main.cpp`: native diagnostic graph and telemetry.
- `scripts/cam01_validation/test_reconnect.py`: scoped interruption/recovery test.
- `scripts/cam01_validation/check_stability.py`: offline evidence gate.
- `tests/test_cam01_stability.py`: rejection tests for misleading/insufficient evidence.
- This document and the current README status/commands.

Credentials are read into a mode-0600 temporary file under the ignored runtime
output directory, mounted only into the validator, and removed on normal exit or
handled termination. Do not commit runtime artifacts. No saved camera URL,
password, existing service implementation or AI feature is modified.
