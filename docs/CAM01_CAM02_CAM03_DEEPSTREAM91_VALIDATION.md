# CAM-01 + CAM-02 + CAM-03 DeepStream 9.1 validation

This stage extends the validated CAM-01 + CAM-02 / 100 ms baseline to exactly
three real RTSP sources. AI remains disabled.

## Graph

```text
CAM-01 -> nvurisrcbin -> NVDEC -> NVMM -> bounded queue -> mux.sink_0
CAM-02 -> nvurisrcbin -> NVDEC -> NVMM -> bounded queue -> mux.sink_1
CAM-03 -> nvurisrcbin -> NVDEC -> NVMM -> bounded queue -> mux.sink_2
                                                        |
                                                        v
                                  nvstreammux(batch-size=3,
                                    live-source=true,
                                    batched-push-timeout=50000,
                                    sync-inputs=false)
                                                        |
                                                        v
                                  nvmultistreamtiler (2 x 2)
                                                        |
                                                        v
                                        NVENC -> tee
                                                |-> MKV evidence
                                                |-> UDP MPEG-TS -> host ffplay
```

Source IDs are deterministic: 0=CAM-01, 1=CAM-02, 2=CAM-03. The fourth tile in
the 2x2 layout is intentionally empty.

Default RTSP jitter latency remains the already validated **100 ms**. The live
preview auto-opens in ffplay when DISPLAY and ffplay are available. Use
`--no-preview` for headless/evidence-only runs.

## Step 1 — tests

```bash
python3 -m unittest tests.test_cam_three_stability -v
```

## Step 2 — short real three-camera visual run

```bash
python3 scripts/validate_cam_three.py \
  --duration 60 \
  --out .runtime/cam01-cam02-cam03-visual
```

The ffplay window must show three advancing real cameras. The output recording is:

```text
.runtime/cam01-cam02-cam03-visual/CAM-01_CAM-02_CAM-03.mkv
```

Do not start the 660-second soak if CAM-03 is missing, frozen, unauthorized or
not hardware-decoded.

## Step 3 — clean 660-second soak

```bash
python3 scripts/validate_cam_three.py \
  --duration 660 \
  --out .runtime/cam01-cam02-cam03-stability

python3 scripts/cam_three_validation/check_stability.py \
  .runtime/cam01-cam02-cam03-stability
```

Acceptance requires each camera to sustain its own source rate, hardware decode,
zero RTP loss/late packets, zero runtime errors, advancing PTS and no queue
backlog. The group output must advance and the finalized 1920x1080 H.264
recording must match the native output counter.

## Step 4 — source isolation

After the clean soak passes, test each source independently. The interrupted
source is removed and recreated inside the same process. The two healthy peers
must continue advancing.

CAM-03:

```bash
python3 scripts/validate_cam_three.py \
  --duration 80 \
  --interrupt-camera CAM-03 \
  --interrupt-at 20 \
  --interrupt-seconds 12 \
  --out .runtime/cam03-isolation-three
```

Then repeat with CAM-01 and CAM-02.

A successful interruption prints:

```text
GROUP ISOLATION_SUMMARY ... healthy_peers=2 ... status=PASS
```

Stop after this three-camera gate passes. Do not start CAM-04 or AI in the same
validation step.
