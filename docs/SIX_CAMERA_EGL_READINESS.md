# Six-camera live preview — NVIDIA EGL and liveness readiness

Status: **LIVE NOT APPROVED**. Read-only findings from the latest operator
audit: X0 authorization is usable, but NVIDIA EGL routing failed with
NVIDIA_EGL_VENDOR_REQUIRED. No RTSP connectivity has been re-tested.

## Isolate the display problem before touching any camera

A working X11 socket/XAUTHORITY does not prove the NVIDIA vendor EGL library
can create the nveglglessink display inside the pinned DeepStream 9.1 image.

Perform **read-only** checks on the Ubuntu workstation:

    echo "DISPLAY=$DISPLAY SESSION=$XDG_SESSION_TYPE"
    nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv
    ls -l /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || true
    ls -l /tmp/.X11-unix/
    test -r "$XAUTHORITY" && echo XAUTHORITY_FILE_READABLE

Do not output the XAUTHORITY cookie. Do not run xhost +, use --privileged,
switch to unrestricted host networking, or change system drivers in this stage.

**Separate approval required:** if these pass and the machine is otherwise
idle/available, perform ONE synthetic NVIDIA EGL display smoke test using
the pinned DeepStream 9.1 container, local X11 cookie, no NVR/RTSP source, and
Docker --network none. Candidate GStreamer pipeline is
nvvideotestsrc ! nvvideoconvert ! nveglglessink, but first confirm installed
elements/caps with gst-inspect-1.0 in the exact image. Preserve GPU workloads.

Inspect container-visible NVIDIA EGL vendor JSON, libEGL_nvidia libraries,
X11 cookie permissions and actual X11 display vendor. If the NVIDIA EGL path
cannot be confirmed, do not start the six-camera live run.

NVIDIA's developer forums describe Mesa/AMD EGL selection problems in
multi-GPU Docker displays; that is a possible diagnostic hypothesis, not
proof of root cause on this particular host. A fallback that sends video over
an existing approved IPC to the PySide6 preview may be evaluated separately;
do not create an unreviewed RTSP/web streaming endpoint.

## Camera liveness

The official deepstream-app emits per-stream recent/cumulative FPS values in
**PERF lines, at the configured five-second interval.

services/gpu/runtime/live_health.py observes those six numeric recent FPS
values via the existing redacted subprocess stdout. It requires TWO consecutive
full per-source positive-FPS reports (minimum 5 FPS per source) before
the operator can start the 600-second timer, and fails closed on:

- Missing updated reports for more than 15 seconds.
- Zero or too-low FPS in any one source.
- Fewer than six stream metrics in a report.
- Unexpected early process exit.

The terminal prints CAM-01..CAM-06 FPS in source order, which also defines
the 3x2 visual grid row-major mapping. It does **not** draw CAM-XX inside each
camera tile. Object ID text via --tiledtext is not a camera label.

**Evidence limit:** FPS statistics alone are not a proof of exact per-source
frame-index/PTS continuity or of actual frame exposure time. Visual review of
all six moving feeds and separate output display verification remain mandatory.
Do not claim complete metadata-level liveness until a properly reviewed
NvDsFrameMeta probe is installed and validated.

## Independent authorization blockers

The above code changes do NOT provide:
- NVR read-only ACL evidence for channels 101,201,301,601,501,401.
- NVR simultaneous session capacity approval.
- A verified six-camera destination-only host firewall policy.
- An authorized live RTSP test.

Do not set existing authorization flags without the underlying approved facts.
The operator must explicitly approve one bounded live run after the gates pass.

## Minimal regression run (NO GPU / NO RTSP)

    python3 -m pytest -q tests/test_six_live_health.py \
      tests/test_gpu_module_boundaries.py \
      tests/test_six_camera_live_config.py

Report any CPU failure first. Do not automatically start synthetic EGL or
physical-camera runs after a successful CPU test.
