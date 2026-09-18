# CAM-01 + CAM-02 + CAM-03 + CAM-04 + CAM-05 DeepStream 9.1 validation

This stage extends the validated four-camera / 100 ms baseline to five real RTSP
sources. AI remains disabled.

```text
CAM-01 -> NVDEC/NVMM -> mux.sink_0
CAM-02 -> NVDEC/NVMM -> mux.sink_1
CAM-03 -> NVDEC/NVMM -> mux.sink_2
CAM-04 -> NVDEC/NVMM -> mux.sink_3
CAM-05 -> NVDEC/NVMM -> mux.sink_4
                         |
                         v
                nvstreammux(batch=5)
                         |
                         v
                nvmultistreamtiler 3x2
                         |
                         v
                      NVENC
                      /   \
                    MKV   UDP -> ffplay
```

Source IDs are fixed: 0=CAM-01, 1=CAM-02, 2=CAM-03, 3=CAM-04, 4=CAM-05.
The live layout keeps two cameras per row: three rows by two columns, with the
sixth tile intentionally empty. Default RTSP jitter latency remains 100 ms.

## Quick gate

```bash
python3 -m unittest tests.test_cam_five_stability -v

python3 scripts/validate_cam_five.py \
  --duration 60 \
  --out .runtime/cam01-cam02-cam03-cam04-cam05-visual
```

The live ffplay window should show five advancing cameras.

## Soak

```bash
python3 scripts/validate_cam_five.py \
  --duration 660 \
  --out .runtime/cam01-cam02-cam03-cam04-cam05-stability

python3 scripts/cam_five_validation/check_stability.py \
  .runtime/cam01-cam02-cam03-cam04-cam05-stability
```

## Isolation

After clean soak PASS, test CAM-05 through CAM-01 individually with
`--interrupt-camera`. Four healthy peers must continue while the target source
is removed and recreated in the same process.

Do not start CAM-06 or AI until this gate passes.
