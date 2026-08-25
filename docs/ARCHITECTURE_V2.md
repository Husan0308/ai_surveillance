# AI Surveillance V2 — Audited Camera Pipeline

Branch: `cleanup/camera-v2-audited-20260825`

## Current milestone

Prove one bounded production hot path before adding more AI:

1. six RTSP streams decode and render continuously;
2. CAM-01 is sampled just-in-time without creating a detector backlog;
3. YOLO26 TensorRT 8.6.1 returns a fresh person detection;
4. that detection is attached as DeepStream object metadata before NvDCF;
5. NvDCF owns per-frame local tracking between detector refreshes;
6. tiler/OSD/display remain independent from detector inference latency.

Cross-camera ReID/global identity, API and frontend are later layers. They remain in
the repository but are not part of this acceptance test.

## Canonical graph

```text
RTSP CAM-01..CAM-06
    |
    v
nvurisrcbin / NVDEC -> NVMM NV12
    |
    v
per-camera tee
    |\
    | \-- CAM-01 detector side path only when requested
    |       queue(max=1, leaky)
    |       -> JIT gate
    |       -> nvvideoconvert
    |       -> 672x384 BGRx letterbox surface
    |       -> appsink(max-buffers=1, drop=true, async=false, sync=false)
    |       -> shared memory
    |       -> isolated TensorRT 8.6.1 process
    |       -> YOLO26 E2E output (1,300,6)
    |       -> fresh person result
    |
    \-- display path
            queue(max=1, leaky)
            -> nvstreammux(batch-size=6, live-source=true)
            -> attach detector NvDsObjectMeta at mux.src
            -> nvtracker / NvDCF (512x288)
            -> nvmultistreamtiler
            -> final queue(max=1, leaky)
            -> nvvideoconvert / RGBA NVMM
            -> nvdsosd
            -> nveglglessink
```

The critical invariant is detector metadata **before** `nvtracker` and tracking
**before** `nvmultistreamtiler`. The audited runtime validates the static links at
startup and prints `CAMERA_PIPELINE_AUDIT status=OK` only when this topology is
present.

## Real-time rules

- one RTSP connection and one hardware decode session per camera;
- TCP RTSP for deterministic NVR transport during stabilization;
- all hot-path queues are bounded/latest-only;
- detector frame capture happens immediately before inference, never prefetched
  during the previous inference/sleep interval;
- detector work is a side path and cannot block the six-camera display branch;
- TensorRT runs in an isolated Python environment/process;
- shared memory is used instead of JPEG/base64 IPC;
- stale detector results are rejected before tracker injection;
- NvDCF, not a Python predictor, owns temporal bbox propagation;
- no ReID, face recognition, Qwen or heatmap is enabled until this milestone passes.

## Geometry contracts

- detector tensor: `672x384`, batch 1, FP32 TensorRT 8.6.1 engine;
- camera content keeps aspect ratio and is centered in the fixed tensor surface;
- detector coordinates are un-letterboxed and mapped to nvstreammux geometry
  before metadata injection;
- NvDCF surface: `512x288`; both dimensions are multiples of 32;
- tiler operates only after tracker metadata has been produced.

## Freshness contract

The scheduler opens the CAM-01 gate only when it is ready to consume a frame. A
healthy run should show:

```text
calls > 0
inputs > 0
timeouts = 0
stale_results = 0
result_age < max_result_age
```

The current debugging floor for detector result age is 350 ms. It is not a target
latency; normal fresh results should stay much closer to TensorRT inference time.

## End-to-end acceptance

With a clearly visible person in CAM-01, the path is accepted only when all of the
following are observed in the same run:

```text
CAMERA_PIPELINE_AUDIT status=OK
boxes > 0
meta_boxes > 0
detector_injected > 0
tracked_now > 0
stale_results = 0
timeouts = 0
```

Only after this passes should ReID/global identity be enabled.

## Canonical entrypoints

```bash
python3 scripts/preflight_cam01_audited_static.py
bash -n scripts/run_cam01_trt86_audited.sh
bash scripts/run_cam01_trt86_audited.sh
```

`python -m services.camera_v2` is kept aligned with the audited runtime as a
secondary entrypoint; the shell launcher remains canonical because it establishes
the TensorRT 8.6 environment and runtime profile explicitly.

## Repository policy

Historical stage-by-stage camera experiments, RF-DETR experiments, old pose/ONNX
alternatives, duplicate CAM-01 launchers, legacy Camera-V2 Sentinel UI and obsolete
heatmap/custom-tracker paths are not kept on the cleanup branch. The branch
`fix/cam01-trt86-e2e-20260825` remains the rollback reference.

Future ReID/global identity code, `services/api_service`, `services/ml_service` and
`services/frontend` are intentionally retained because they belong to the next
milestones rather than the discarded camera-pipeline experiments.
