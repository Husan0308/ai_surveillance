# CAM-01 + CAM-02 DeepStream 9.1 validation

This stage extends the validated CAM-01 baseline to exactly two real RTSP sources.
It does **not** start CAM-03 through CAM-06 and does not load YOLO, TensorRT
inference, tracking, ReID, face recognition, pose, heatmaps or the frontend.

The graph is:

```text
CAM-01 nvurisrcbin -> NVDEC -> NVMM -> bounded queue -> mux.sink_0
CAM-02 nvurisrcbin -> NVDEC -> NVMM -> bounded queue -> mux.sink_1
                                                    |
                                                    v
                             nvstreammux(batch-size=2,
                               live-source=true,
                               batched-push-timeout=50000,
                               sync-inputs=false)
                                                    |
                                                    v
                                  nvmultistreamtiler (1 x 2)
                                                    |
                                                    v
                                      NVMM -> NVENC -> MKV
```

The two source IDs are fixed:

- source 0 = CAM-01
- source 1 = CAM-02

The launcher reads the existing camera configuration and credentials through the
shared loader. Secrets are written only to a mode-0600 ignored runtime file,
mounted into the dedicated validator container, redacted from logs and removed
on exit.

NVIDIA documents `nvstreammux` as the element that batches multiple input
sources; each input uses a requested `sink_%u` pad. This validation therefore
uses one mux with batch-size 2 rather than two independent pipelines. Live RTSP
sources use `live-source=true`; the pair starts with `sync-inputs=false` so
one source is not deliberately clock-gated behind the other.

## 1. Quick visual run

```bash
cd ~/ai_surveillance
python3 scripts/validate_cam_pair.py \
  --duration 45 \
  --out .runtime/cam01-cam02-visual
```

Then open:

```bash
vlc .runtime/cam01-cam02-visual/CAM-01_CAM-02.mkv
```

The output is a 1x2 tiled real recording. Both panes must advance; a frozen
first frame is not acceptance.

## 2. Clean soak

```bash
python3 scripts/validate_cam_pair.py \
  --duration 660 \
  --out .runtime/cam01-cam02-stability

python3 scripts/cam_pair_validation/check_stability.py \
  .runtime/cam01-cam02-stability
```

The checker requires at least 600 seconds after warmup, advancing frames and
PTS for both cameras, roughly 20 FPS per camera, advancing tiled output,
hardware decoder evidence for both sources, no source/shared runtime errors,
no RTP loss/late packets, no queue buildup, bounded RSS/VRAM growth and a
finalized 2560x720 H.264 recording.

## 3. Source-isolation test

Because CAM-01 and CAM-02 are channels on the same NVR IP/RTSP port, a host
network cut would disconnect both simultaneously and would not test per-source
isolation. The validator therefore has an explicit source-level isolation
mode: it sets only the selected `nvurisrcbin` to NULL for the requested
interval while leaving the peer source, mux and process alive, then restores
that source to PLAYING.

Test CAM-02 failure while CAM-01 remains alive:

```bash
python3 scripts/validate_cam_pair.py \
  --duration 80 \
  --interrupt-camera CAM-02 \
  --interrupt-at 20 \
  --interrupt-seconds 12 \
  --out .runtime/cam02-isolation
```

A successful run prints:

```text
PAIR ISOLATION_SUMMARY ... status=PASS
```

Repeat with `--interrupt-camera CAM-01` to test the inverse direction.

This is an application/source-isolation test, not a physical Ethernet/NVR
failure. A later deployment test can cover real network/camera faults when a
fault can be scoped to one source without taking the shared NVR offline.

## Acceptance gate

Do not add CAM-03 until all of the following are true:

- CAM-01 and CAM-02 both use `nvv4l2decoder` and NVMM.
- Both cameras sustain their actual source rate independently.
- `nvstreammux` runs with batch-size 2.
- Tiled output keeps advancing.
- Clean 660-second soak passes.
- CAM-02 isolation leaves CAM-01 advancing and CAM-02 recovers.
- CAM-01 isolation leaves CAM-02 advancing and CAM-01 recovers.
- No AI component is loaded.
