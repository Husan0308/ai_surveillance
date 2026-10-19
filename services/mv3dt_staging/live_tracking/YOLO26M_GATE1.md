# Entrance Room YOLO26m — Gate 1 (STATIC ONLY)

Status: **IMPLEMENTED, NOT YET TESTED ON THE HOST**. Do not label PASS
until the host CPU-only checks complete. Do not start any RTSP/live test here.

## Scope

- Branch: `experiment/entrance-yolo26m-offline-gate1`
- Base: `backup/ai-surveillance-20261013` at `3a6774b18e4dcf4ebb4791d7213a31ae3f557bb4`
- Changes only inside `services/mv3dt_staging/live_tracking/` and its gate tests.
- The six-camera production supervisor, models, configs, tracker and API are unchanged.
- The existing `live` command still defaults to PeopleNet. It is **not** switched to YOLO.
- The `mux-smoke` command can explicitly use `--detector yolo26m --inference`.
- Already-rectified 1920x1080 AMC input clips use `--input-geometry rectified` (default).
  They are **not** rectified twice. Raw 3200x1800 / 2560x1440 clips require
  `--input-geometry raw` and the validated empirical bridge.
- The YOLO26m batch-six config is source-hash checked; a private copy changes
  only `batch-size=6` to `batch-size=2`. The model/parser/ONNX/labels hashes
  must match the existing frozen `EXPECTED` manifest before Docker starts.
- The proposed image is pinned by the existing room experimental image digest,
  and `--pull=never --network none` prevents network use by the inference
  container. Actual TensorRT engine compatibility is **not** presumed.

## Gate 1: CPU-only tests, no GPU

On the Ubuntu machine, after checking that the worktree is clean:

```bash
cd /home/apsidal/ai_surveillance-gate3j
git status --short
git fetch origin experiment/entrance-yolo26m-offline-gate1
git switch -c experiment/entrance-yolo26m-offline-gate1 --track origin/experiment/entrance-yolo26m-offline-gate1

.runtime/full-stack-venv/bin/python -B -m pytest -q \
  tests/test_entrance_yolo26m_gate1.py \
  tests/test_entrance_live_tracking_gate.py
```

Do not switch branches with unrelated uncommitted changes. The .runtime
directory is intentionally ignored and remains local to the worktree.

**PASS condition:** all requested CPU tests pass and the production config
files remain unchanged. Report the test output verbatim. A PASS here only means
the frozen-file and command-construction contract is safe; it says nothing about
GPU performance, detection recall or even successful engine deserialization.

## Gate 2: bounded offline PGIE/NvDCF smoke (NOT RUN; after Gate 1)

Before a GPU run, establish production GPU and latency headroom and verify
the 180-second AMC files really are already rectified. Missing assets or an
incompatible TensorRT engine is **BLOCKED**, not permission to rebuild a
production engine. The command below is a template requiring existing clips:

```bash
cd /home/apsidal/ai_surveillance-gate3j
.runtime/full-stack-venv/bin/python -B -m \
  services.mv3dt_staging.live_tracking.live_bridge mux-smoke \
  --cam02-video /PATH/TO/EXISTING/RECTIFIED/cam_00.mp4 \
  --cam05-video /PATH/TO/EXISTING/RECTIFIED/cam_01.mp4 \
  --input-geometry rectified --frames 8 --inference \
  --detector yolo26m \
  --yolo-artifacts .runtime/sixcam-approved-bbox-20261006-r12-profile/models
```

Pass criteria are real frame-matched MUX, PGIE and NvDCF outputs for both
cameras, no spurious source ID, zero invalid metadata crosswalks, and confirmed
model load. It does NOT validate recall/false positives or authorize live use.

## Later gates

- Gate 3: same-frames PeopleNet vs YOLO26m offline A/B; measure actual
  recall, duplicate boxes, NvDCF ID continuity, GPU and VRAM, not counts alone.
- Gate 4: only if safe, a bounded read-only CAM-02/CAM-05 live smoke using
  the winning experimental detector; no automatic switch or 60-second review.
- MV3DT global association and 2D map stay disabled until independent
  pairwise geometry and source synchronization validation.

The most recent live PeopleNet experiment peaked at 99% GPU, so never
bypass the 95% resource guard or run another sustained GPU workload without
measuring production impact.
