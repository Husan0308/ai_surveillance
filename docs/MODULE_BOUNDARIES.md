# AI Surveillance — Module Boundaries (incremental refactor)

Status: source boundary migration **prepared**, not yet GPU/RTSP verified.
Baseline: \`fix/deepstream91-six-live-preview-20261025\` at \`315ea383a2fc\`.
Refactor branch: \`refactor/gpu-module-boundaries-20261025\`.

## What changes now

This stage relocates only the isolated six-camera PeopleNet live-preview
*control-plane* code and shared lock primitive. It deliberately does **not**
move the frozen NVIDIA native pipeline, ML inference backends, API routing,
Qt components, calibration implementations, or legacy runtime entrypoints.

| Ownership | Canonical path | Responsibility |
|---|---|---|
| GPU live (new) | \`services/gpu/runtime/live_launcher.py\` | Readiness, secure Docker startup/shutdown, camera resource ownership |
| GPU config (new) | \`services/gpu/runtime/live_config.py\` | Mapping, validated PeopleNet config, live DeepStream INI generation |
| Camera locks (shared) | \`services/shared/source_ownership.py\` | Single per-camera flock ownership contract |
| Camera settings (shared) | \`services/shared/camera_config.py\` | One camera list / approved scoped or shared credentials |
| GPU/ML legacy | \`services/camera_v11/\`, \`services/ml_service/app/deepstream/\` | Existing historical producers/capture paths; do not launch simultaneously |
| ML service | \`services/ml_service/app/\` | API for ML metadata and future identity/occupancy logic |
| API service | \`services/api_service/app/\` | HTTP/WS and state handoff; no new direct RTSP reader |
| Frontend | \`services/frontend/app/\` | PySide6 camera display and operator interactions |
| Alternate Qt UI | \`services/frontend/sentinel_v1/\` | Existing UI not selected as new GPU canonical endpoint |
| Old CLI compatibility | \`scripts/six_camera_live/{launch,config}.py\` | Alias to GPU modules; no second active implementation |

### Dependency direction

\`\`\`
NVR / six RTSP sources
       |
       v
services/gpu/runtime/  -- GPU decode -> streammux -> PGIE -> NvDCF -> OSD
       |                                      |
       |                                 NvDs metadata
       v                                      v
Live EGL preview                      ML metadata consumer (later)
                                              |
                                              v
                                     API REST/WS -> PySide6 frontend

services/shared/: cameras, cross-service data contracts, ownership locks
\`\`\`

The existing live viewer currently uses NVIDIA's official \`deepstream-app\`
with EGL sink directly. It does not yet deliver frames to the PySide6 frontend;
**do not describe that integration as complete**.

The GPU pipeline owns detector and tracker execution. ML owns *interpretation*
of produced metadata and future cross-camera identity, face enrollment and room
occupancy. Do not add another PGIE or RTSP reader inside API/frontend.

## Supported commands

New canonical, CPU-only configuration check:

\`\`\`bash
python3 -m services.gpu.runtime.live_launcher --check \\
  --infer-config /absolute/validated/peopleNet-b6.txt \\
  --models-root .runtime/models
\`\`\`

Old command remains supported unchanged:

\`\`\`bash
python3 -m scripts.six_camera_live.launch --check \\
  --infer-config /absolute/validated/peopleNet-b6.txt \\
  --models-root .runtime/models
\`\`\`

Actual camera test is **not authorized** by this document.
\`--run\` still requires validated engine hash, scoped NVR authorization,
independently verified restricted egress, X11/EGL readiness and explicit
physical-camera approval. See \`docs/SIX_CAMERA_LIVE_VIEWER.md\`.

## Do not move or delete yet

- \`services/camera_v11/native/\` historical approved evidence/runtime.
- \`scripts/run_six_camera_v13_wall_candidate.py\` historical YOLO26m path.
- 149 flat historical diagnostic/helper scripts in \`scripts/\`.
- Existing ML, API, frontend entrypoints and legacy UI without import audit.
- \`config/deepstream/\` experimental NvDCF profiles.
- \`.runtime/\` assets and validated DeepStream engine caches (ignored).

Any old path is deleted only after: active entrypoint audit, import test
coverage, pinned artifacts review, equivalent offline comparison, explicit
rollback point and separately approved live validation.

## How to localize failures

| Symptom | First evidence/owner |
|---|---|
| NVR auth or RTSP disconnect | GPU source + NVR ACL/network + camera ID |
| Decoder errors | GPU NVIDIA source/decoder log |
| Missing detection | GPU PeopleNet PGIE vs post-tracker metadata |
| Fragmented ID | GPU NvDCF, not API |
| Same person counted twice across cameras | ML identity/occupancy state (unvalidated) |
| Healthy GPU but API unavailable | ML/API health and IPC contracts |
| Healthy API but blank Qt tile | Frontend preview transport and frame freshness |
| Busy GPU / duplicate live sessions | Shared \`camera-owner-locks\`, existing process and TCP socket inventory |

Include \`camera_id\`, DeepStream \`source_id\`, frame timestamp, pipeline stage
and monotonic observation time when adding telemetry, without credentials.

## Acceptance gates

1. **Source / import gate**: old and new module paths resolve to the same
   objects; shared camera lock is not duplicated; scripts have no separate
   launcher implementation.
2. **CPU tests**: existing \`tests/test_six_camera_live_config.py\`, new
   \`tests/test_gpu_module_boundaries.py\`, existing camera-source ownership
   and camera selection tests pass.
3. **Offline resource gate**: exact PeopleNet batch-six model/engine hash and
   DeepStream 9.1 dependencies available; no GPU replay is implied.
4. **Readiness**: NVR read-only channel permissions, camera-only egress and
   desktop display checked independently. No flag bypass.
5. **Physical live gate**: separate authorization, all six fresh RTSP feeds
   on screen for 600 seconds, operator PASS, clean shutdown.
6. **Future refactor gate**: API/ML/UI internal file movements only after
   these interface contracts and runtime baselines are confirmed.

If a gate fails, fix only that module. Do not combine a broad rename with
model/threshold changes or a new GPU pipeline implementation.
