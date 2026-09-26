# Six-camera freeze compatibility gate

This gate reuses the validated six-camera DeepStream 9.1 transport/decode/mux
acceptance contract from the 2026-09-18 six-camera freeze baseline without
rolling current CAM-01/CAM-04 analytics code back to the old implementation.

Scope:

- CAM-01 through CAM-06 RTSP ingest
- hardware decode/NVDEC evidence
- six-source nvstreammux
- frame/PTS progress
- queue/loss/error checks
- 660-second stability soak
- per-camera interruption/recovery while five peers continue

Not changed by this gate:

- CAM-01/CAM-04 PeopleNet/MV3DT analytics
- NvDCF recall tuning
- OSNet/global identity
- calibration
- BEV/presence
- production identity logic

The freeze compatibility gate intentionally uses the historical 100 ms camera
transport profile. Low-latency P95 < 40 ms is a separate later gate and must
not be inferred from this stability test.

## One stage at a time

Quick:

```bash
bash scripts/run_six_camera_freeze_gate.sh quick
```

Soak:

```bash
bash scripts/run_six_camera_freeze_gate.sh soak
```

Isolation/recovery:

```bash
bash scripts/run_six_camera_freeze_gate.sh isolation
```

All:

```bash
bash scripts/run_six_camera_freeze_gate.sh all
```

Runtime evidence is written only under `.runtime/`.
