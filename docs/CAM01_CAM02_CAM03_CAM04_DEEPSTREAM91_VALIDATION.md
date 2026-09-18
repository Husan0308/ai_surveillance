# CAM-01 + CAM-02 + CAM-03 + CAM-04 DeepStream 9.1 validation

This stage extends the validated three-camera / 100 ms baseline to exactly four
real RTSP sources. AI remains disabled.

```text
CAM-01 -> NVDEC/NVMM -> mux.sink_0
CAM-02 -> NVDEC/NVMM -> mux.sink_1
CAM-03 -> NVDEC/NVMM -> mux.sink_2
CAM-04 -> NVDEC/NVMM -> mux.sink_3
                         |
                         v
                nvstreammux(batch=4)
                         |
                         v
                nvmultistreamtiler 2x2
                         |
                         v
                      NVENC
                      /   \
                    MKV   UDP -> ffplay
```

Source IDs are fixed: 0=CAM-01, 1=CAM-02, 2=CAM-03, 3=CAM-04.
Default RTSP jitter latency remains 100 ms.

## Quick gate

```bash
python3 -m unittest tests.test_cam_four_stability -v

python3 scripts/validate_cam_four.py \
  --duration 60 \
  --out .runtime/cam01-cam02-cam03-cam04-visual
```

The live ffplay window should show four advancing cameras in a full 2x2 grid.

## Soak

```bash
python3 scripts/validate_cam_four.py \
  --duration 660 \
  --out .runtime/cam01-cam02-cam03-cam04-stability

python3 scripts/cam_four_validation/check_stability.py \
  .runtime/cam01-cam02-cam03-cam04-stability
```

## Isolation

After clean soak PASS, run CAM-04, CAM-03, CAM-02 and CAM-01 individually with
`--interrupt-camera`. The three healthy peers must continue while the target
source is removed and recreated in the same process.

Do not start CAM-05 or AI until this gate passes.
