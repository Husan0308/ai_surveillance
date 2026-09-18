# DeepStream 9.1 migration audit — 2026-09-18

PASS 0 audit complete. Platform migration BLOCKED pending driver upgrade/reboot.
The user elected to perform that locally after this audit. Camera implementation,
runtime acceptance and stability testing have not started. No AI stage is enabled.

## Audit scope

Inventoried all 547 tracked files, including 53 CI workflows, and scanned source,
configs, launchers, tests and docs for stack dependencies, GStreamer plugins, old
TensorRT APIs, CPU copies and dropping policies. Inspected camera/config, ML, API,
frontend entrypoints and native bridge build logic. Generated `.runtime`,
`artifacts`, `.venv` and `.venv-trt86` remain local historical data/environments.
No repository AGENTS.md was found. Two pre-existing README edits are preserved.

Evidence commands: `nvidia-smi`, `nvcc --version`, `deepstream-app --version-all`,
`gst-inspect-1.0`, `dpkg-query`, `apt-cache policy`, APT install simulation and
Docker inventory/manifest/package inspection. No host packages were installed.

## Installed and target stack

| Component | Observed host | Selected container / required host |
| --- | --- | --- |
| OS | Ubuntu 24.04.4 x86_64 | Ubuntu 24.04 |
| GPU | RTX 3060, 12288 MiB | Same GPU |
| Driver | 580.178.04 | NVIDIA baseline 595.58.03; Ubuntu candidate 595.91.07 |
| DeepStream | 7.1.0 | 9.1.0, confirmed in downloaded image |
| CUDA | DeepStream reports runtime 12.6; nvcc not on PATH | 13.2; image cuda-cudart package 13.2.51-1 |
| TensorRT | DeepStream reports 10.6; APT also has 10.16.1.11 runtime and 11.2.1.2 development packages | 10.16.1.11-1+cuda13.2, confirmed in image |
| Container toolkit | 1.19.1-1; NVIDIA Docker runtime registered | Reuse existing toolkit |

The nvidia-smi CUDA 13.0 field is driver capability, not installed toolkit version.
Host TensorRT packages are mixed and unsuitable as one coherent development
stack. The old Python 3.10 TRT86 environment additionally pins TensorRT
8.6.1.post1, CUDA runtime 12.1.105, cuBLAS 12.1.3.1 and cuDNN 8.9.1.23 for Pascal.

NVIDIA's [9.1 requirements](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_Installation.html)
list CUDA 13.2, TensorRT 10.16.0.72 and driver 595.58.03. The selected official image
ships the newer TensorRT 10.16.1.11 patch; preserve that coherent image rather than
replacing packages. Local skills still mention CUDA 13.1/TRT 10.14/driver590 and
were superseded by the current official docs and actual image contents.

The [documented samples image](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_docker_containers.html)
is `nvcr.io/nvidia/deepstream:9.1-samples-multiarch`. Its amd64 digest is pinned
in `config/deepstream-platform.json` and fully downloaded. Triton is unnecessary.
Package inspection ran without GPU access or host mounts; this is not GPU runtime
validation. GStreamer scanning in that inspection emitted missing libcuda and
optional audio/codec-library warnings. Missing libcuda is expected without GPU
injection; required video plugins must still be verified after reboot. Do not
install unrelated audio/codec dependencies just to suppress optional warnings.

## Repository findings and decisions

| Area | Finding | Migration action |
| --- | --- | --- |
| Cameras | config/cameras.yaml has exactly CAM-01…CAM-06 and six distinct RTSP URLs. services/shared/camera_config.py loads per-camera/global .env credentials; all six have credentials configured. | Preserve URLs/IDs/credential precedence. Reuse shared loader, never print credentials. |
| Camera service | services/camera_service/app/runtime.py already implements RTSP/NVDEC and optional mux/EGL wall. Headless bypasses mux. | Adapt this service after platform gate; no new versioned pipeline. Both output modes must exercise mux batch-size 6. |
| Recovery | Only three reconnect attempts. Bus handler logs ERROR/EOS, with no per-source recovery supervisor. | Implement and prove unavailable-at-start/stall recovery and isolation; logging alone is insufficient. |
| Dropping | Camera service and V11 use leaky one-buffer queues and drop-on-latency=true. Some CI checks enforce these. | Camera-only profile needs bounded non-leaky queues, no decoder skipping, no latency dropping; update affected policy tests. |
| Other pipelines | camera_v2, camera_v11, ML capture and SHM variants overlap. V11 step0 creates six independent pipelines without mux/display. | Single owner using existing shared lock. No camera runtime or Docker container was running during audit. Retain history until dependencies justify removal. |
| Plugins | nvurisrcbin, nvv4l2decoder, nvstreammux, tiler, converter, OSD and EGL resolve to host DS 7.1. RTSP/RTP depayloaders and H.264/H.265 parsers are used. | Verify inside 9.1 image. Never mount host NVIDIA libraries or GST registry/cache. |
| TensorRT | Old builders/workers use num_bindings, get_binding_*, set_binding_shape and binding-index execution; launchers isolate TRT86 for GTX 1050 Ti. | Exclude all inference workers/models/imports. API migration belongs to later inference pass. |
| Native ABI | Metadata/PTS bridges compile against discovered SDK paths; cache invalidation uses source mtime, not SDK version. | Never reuse 7.1 .runtime/*.so in 9.1. Rebuild minimal metadata instrumentation/labels against 9.1 if needed. |
| CPU copies | ML capture maps NumPy/BGR; SHM/JPEG/latest-frame branches copy/drop video. | No appsink/NumPy/OpenCV/JPEG/SHM in verification video path. |
| ML | Current FastAPI main reads V11 telemetry; video endpoint returns 503. Legacy capture/detection code remains. | Leave stopped. |
| API | FastAPI/httpx proxy plus monitoring websocket. | Leave stopped until integration stage. |
| Frontend | PySide6 apps depend on MJPEG/SHM/telemetry. Sentinel data.py generates people/events. | Exclude frontend from acceptance; native DeepStream wall with real images and CAM labels. |
| Python | ML uses FastAPI/Uvicorn/YAML/dotenv/NumPy/OpenCV; API uses FastAPI/Uvicorn/httpx; UI uses PySide6. | Host preflight needs existing YAML/dotenv only. Do not install broad ML requirements. |
| Docker | No tracked Dockerfile/Compose. Cached images were DS 6.3/7.1 and CUDA 12.3.1. | Use pinned DS 9.1 image, no old LD_LIBRARY_PATH, GST_PLUGIN_PATH or venv injection. |
| README | Twelve explicit file references do not exist, including canonical launcher. | Add current status; preserve historical content and user edits. |

The scan found 173 stack-assumption lines across 95 files and 111 binding-era TRT
references across 19 files. These are migration candidates, not permission to
blindly delete future features. Runtime inheritance requires dependency review.

## Changes in this audit

- Added config/deepstream-platform.json as exact stack/image manifest.
- Added scripts/preflight_deepstream91.py: camera IDs/URLs, selected GPU/VRAM,
  active driver, Docker runtime and pinned image checks. Optional --container
  checks GPU-container stack/plugins only after host gate passes. It does not
  open streams or mark any camera healthy.
- Added focused gate tests covering unsupported driver/GPU/VRAM, missing GPU and
  mismatched stack. Added README migration banner and this audit.
- Downloaded pinned official image; no host driver/CUDA/TRT changes, no legacy
  code deletion, no camera configuration changes, no additional pipeline.

JSON evidence is `.runtime/deepstream91/preflight.json`. Exit 1 means blocked;
exit 2 means container check needed. Exit 0 means ready for camera implementation,
never a camera or stability PASS. No NVIDIA_REQUIRE_CUDA bypass or driver
forward-compatibility workaround is used.

## Driver handoff

APT simulation succeeded: 22 packages removed, 22 installed, 2 upgraded. This
replaces 580 driver/kernel-module packages without installing host CUDA/TRT.
Passwordless sudo is unavailable. User will authenticate locally and reboot.

```bash
sudo apt-get install nvidia-driver-595=595.91.07-0ubuntu0.24.04.1
# Save work and reboot when ready, then:
nvidia-smi
python3 scripts/preflight_deepstream91.py --container --report .runtime/deepstream91/preflight.json
```

If that exact version becomes unavailable, inspect candidates and simulate again.
Do not install arbitrary CUDA/TRT meta-packages. Keep old host userspace outside
the container; defer package removal until other consumers have been checked.

## Staged implementation and acceptance plan

1. After reboot, validate GPU-enabled 9.1 container and required plugin properties.
   Check NVIDIA's documented RTSP rtpmanager fix in the image; apply its shipped
   additional-install script only if required, never copy a host plugin.
2. Adapt camera_service as sole owner:
   `6 × nvurisrcbin/NVDEC → NVMM → bounded queue → nvstreammux(batch-size=6)`
   then tiler/GPU conversion/label OSD/EGL. Decode precedes mux. Keep camera IDs
   stable during outages and avoid showing cached frames as live. No inference,
   tracking or CPU frame mapping. No frontend integration yet.
3. Start TCP, measured source rates, finite batch push timeout, live-source=1,
   sync-inputs=0, non-leaky queues and async=false sinks. Validate timestamps,
   partial batches and caps against 9.1; fail on missing required properties.
4. Handle source ERROR/EOS/stalls locally with bounded reconnect backoff.
   Shared mux/output failures must fail visibly. Five good sources must keep
   delivering while one starts offline or loses its connection.
5. Native metadata counters for source/decode/output FPS, PTS, batch occupancy,
   queue depth and RTP loss/late statistics. Sample GPU/NVDEC, CPU/RSS and VRAM
   off the video path; no Python per-frame processing.
6. Verify all six labeled real streams. Measure uninterrupted 600 s minimum,
   preferably 1800 s after warmup. Compare early/late RSS/VRAM/CPU, PTS lag,
   queue trends, decoded/delivered/source FPS. Repeating errors, growth, backlog
   and severe frame loss cannot pass on total frames or process-alive alone.
7. Separately interrupt one app source connection (not physical camera settings),
   prove the other five continue, restore and prove reconnect. Test unavailable
   source at startup too; follow recovery with uninterrupted all-six soak.
8. Publish per-camera evidence. Stop at camera acceptance; later AI/UI stages
   remain deferred even when cameras pass.

## Actual results

RTX 3060 detection: YES. Six configurations valid: YES.
CAM-01, CAM-02, CAM-03, CAM-04, CAM-05, CAM-06: CONFIGURED, NOT TESTED.
DS 9.1 GPU execution, source failure isolation, reconnect, real display,
queue/drop behavior, 10-minute stability and VRAM stability: NOT TESTED.

Exact camera-only command used: NONE, because platform migration is blocked.
The executed command for this pass is the platform preflight above. There is
no validated DS 9.1 camera launch or stability command yet.

Verification: all five focused platform-gate unit tests passed. The final live
preflight exited 1 with only the active-driver blocker. The existing README
trailing whitespace was preserved with the user’s pre-existing edits.
