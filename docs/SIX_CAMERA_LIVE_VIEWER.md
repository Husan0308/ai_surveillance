# DeepStream 9.1 — six-camera live visual preview

Canonical GPU runtime: `services/gpu/runtime/`. The old
`python3 -m scripts.six_camera_live.launch` command is retained as a
compatibility wrapper. See [module ownership](MODULE_BOUNDARIES.md).

This is a **new, isolated** source path. It does not change the old YOLO26m
V13 service, existing camera diagnostics, recorded validation evidence or AMC.

## What is implemented

- Six sources in verified order: CAM-01, CAM-02, CAM-03, CAM-04, CAM-05, CAM-06.
- Channel mapping: 101, 201, 301, 601, 501, 401.
- Official deepstream-app, one NV stream mux (batch 6), one PeopleNet FP16
  inference configuration, NVIDIA stock NvDCF and EGL display.
- **Two rows by three columns** at 1920x1080. The older diagnostics intentionally
  retain their three-row, two-column 1920x1620 layout.
- Camera credentials are resolved using the existing approved per-camera
  override/shared fallback in services/shared/camera_config.py.
- Configuration containing RTSP credentials is passed over a pipe into a private
  container tmpfs. No authenticated URIs appear in Docker command arguments,
  tracked files or saved diagnostic reports. Child output is redacted and Docker
  parallel logging is disabled.
- Docker image digest is taken from config/deepstream-platform.json.
- Manual confirmation of **all six moving tiles** starts the 600-second
  observation period. This is an operator confirmation, not independent
  per-source capture-timestamp validation. Closing early is not a PASS.
- No camera recording, background service, custom CUDA, ReID or occupancy logic.

## Required local assets

The validated PeopleNet FP16 **batch-six** nvinfer configuration is an input.
It is not the tracked config_infer_peopleNet_fullframe_diagnostic_b2.txt:
that file is batch **two** and must not be silently substituted.

The selected nvinfer file must have the following fields:

- batch-size=6
- network-mode=2
- network-type=0
- num-detected-classes=3
- filter-out-class-ids=1;2
- Existing model, label and cached compatible engine file paths below /models/.

The /models mount comes from --models-root (normally .runtime/models).
All referenced files must already exist. They are kept locally, not pushed
to a public GitHub repository.

## Step 1: CPU-only configuration preparation

From the repository root, locate the validated Stage 5B nvinfer config.
Supply its real path (do not use the older batch-two diagnostic file):

    python3 -m services.gpu.runtime.live_launcher \
      --check \
      --infer-config /absolute/path/to/validated-peoplenet-b6.txt \
      --models-root .runtime/models

This check does **not** call Docker, RTSP, NVR or GPU inference. It validates
six-camera mapping, approved credential presence, model file locations and
private configuration generation without displaying the RTSP URLs.

Expected check result:
CONFIG PREPARATION: PASS; LIVE NETWORK/ACL/EGL: NOT VERIFIED.

A successful check is not permission to start live cameras.

## Step 2: Administrator readiness

Before any live test, independently verify all of the following:

1. NVR administrator confirms **read-only** permission on exactly the six
   designated channels and at least six concurrent RTSP sessions.
2. Credential owner approves the existing shared credentials or scoped
   overrides; never paste passwords into chat or Git.
3. Host administrator verifies the six-camera network permits egress only to
   the approved NVR endpoint and required transport (TCP RTSP/554), including
   IPv4 and IPv6. Docker bridge creation alone is not an egress allowlist.
4. The dedicated network exists. The launcher refuses Docker host, default
   bridge, none and IPv6-enabled networks, but does not independently prove
   the host firewall's source/destination filtering.
5. Local X11 DISPLAY socket and private XAUTHORITY cookie work under the
   intended desktop user. Do not use xhost +.
6. Existing camera sessions, GPU/VRAM usage, AMC and STT must remain unaffected.
7. Obtain fresh approval for one bounded 10-minute physical test.

The --nvr-readonly-authorized and --network-egress-verified switches below
are **operator/admin attestations**, not substitutes for the actual checks.
They must never be passed unless their facts were verified.

## Step 3: One interactive, explicitly authorized physical-camera run

Only *after* Step 2, and after the CPU-only check passed:

    python3 -m services.gpu.runtime.live_launcher \
      --run \
      --infer-config /absolute/path/to/validated-peoplenet-b6.txt \
      --models-root .runtime/models \
      --network APPROVED_RESTRICTED_NETWORK \
      --expected-engine-sha256 VERIFIED_SHA256_FROM_STAGE5_MANIFEST \
      --authorize-physical-rtsp \
      --nvr-readonly-authorized \
      --network-egress-verified

The launcher checks the cached batch-six engine against the expected SHA256 from the verified offline manifest, then makes a single Docker live-preview process. After all six
moving frames are visually confirmed, enter START in the terminal to begin
a **600-second** window; otherwise it fails closed after the startup timeout.
A successful process exit alone does not establish that seated-person recall,
six-camera connectivity or identity accuracy PASS: final visual acceptance
belongs to the operator.

Manual interruption should lead to cleanup of the named owned container.
Check remaining RTSP sessions separately after exit.

## Known limitations / not yet validated

- The new code has not been compiled or executed with this machine's GPU,
  actual DeepStream 9.1 image or physical RTSP cameras.
- The existing validated b6 inference config and matching engine are external
  runtime assets, not part of the public repository.
- The six-camera restricted egress/firewall policy and NVR channel ACL remain
  prerequisites. We do not relax the older runner's security requirements.
- DeepStream stock tiled OSD/--tiledtext shows Person boxes and numeric tracking
  IDs. Stable explicit CAM-XX captions, independent source health display and
  automated live-readiness timing still need on-device verification.
- This viewer does not perform cross-camera ReID or guarantee that seated
  people remain counted when both cameras lose sight of them.

## Offline unit checks

    python3 -m pytest -q tests/test_six_camera_live_config.py

No new production services or existing validation code should be changed
just to make this preview run.
