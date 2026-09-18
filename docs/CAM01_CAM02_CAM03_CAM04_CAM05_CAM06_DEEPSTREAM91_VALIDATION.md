# CAM-01 through CAM-06 DeepStream 9.1 validation

This is the final camera-only stage. It extends the validated five-camera / 100 ms
baseline to all six real RTSP sources. AI remains disabled.

```text
CAM-01 -> NVDEC/NVMM -> mux.sink_0
CAM-02 -> NVDEC/NVMM -> mux.sink_1
CAM-03 -> NVDEC/NVMM -> mux.sink_2
CAM-04 -> NVDEC/NVMM -> mux.sink_3
CAM-05 -> NVDEC/NVMM -> mux.sink_4
CAM-06 -> NVDEC/NVMM -> mux.sink_5
                         |
                         v
                nvstreammux(batch=6)
                         |
                         v
                nvmultistreamtiler 3x2
                         |
                         v
                      NVENC
                      /   \
                    MKV   UDP -> ffplay
```

Source IDs are fixed: 0=CAM-01 through 5=CAM-06. The 3x2 layout is fully occupied
and preserves two cameras per row. Default RTSP jitter latency remains 100 ms.

## Quick gate

```bash
python3 -m unittest tests.test_cam_six_stability -v

python3 scripts/validate_cam_six.py \
  --duration 60 \
  --out .runtime/cam01-cam02-cam03-cam04-cam05-cam06-visual
```

The live ffplay window should show all six cameras advancing.

## Soak

```bash
python3 scripts/validate_cam_six.py \
  --duration 660 \
  --out .runtime/cam01-cam02-cam03-cam04-cam05-cam06-stability

python3 scripts/cam_six_validation/check_stability.py \
  .runtime/cam01-cam02-cam03-cam04-cam05-cam06-stability
```

## Isolation

After the clean soak passes, test CAM-06 through CAM-01 individually with
`--interrupt-camera`. Five healthy peers must continue while the target source
is removed and recreated in the same process.

After this gate passes, camera transport/decode/mux/tile validation is complete.
Only then should inference/detection be added.
