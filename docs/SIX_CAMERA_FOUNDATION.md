# F2 camera-only live foundation

Use the frozen F1 runtime on the host to launch the existing preview worker
inside the digest-pinned DeepStream 9.1 container:

```bash
PYTHONNOUSERSITE=1 .runtime/full-stack-venv/bin/python -B -m scripts.run_preview_foundation \
  --output .runtime/preview-foundation-new-run --duration 300
```

The output directory must not exist. This command does not start inference,
tracking, identity, API, ML, UI, or calibration services. It installs no packages.
The host's legacy DeepStream 7.1 plugin directory is not used. F1 dependencies,
source, and authorized camera environment file are mounted read-only; the pinned
container supplies its Ubuntu GI/GStreamer and DeepStream 9.1 plugins. Preview
slots are shared with the host through `/dev/shm`.

## Ownership and freshness

In camera-only mode one process owns all six cameras. In the later room-pair
analytics topology explicitly select only `CAM-02,CAM-03,CAM-05,CAM-06` for this
worker; the analytics owner must be the only reader of CAM-01/CAM-04. Do not run
this all-six mode concurrently with an analytics owner.

Cooperative per-camera flock locks are acquired before codec discovery. They
prevent overlapping preview owners, including partial camera selections. Keep
the lock files: unlinking them while held breaks exclusion. All cooperating
owners must share `CAMERA_OWNER_LOCK_DIR` (default `.runtime/camera-owner-locks`).
Non-cooperating/native owners must also be checked through process/RTSP socket
inventory; flock alone is not evidence that no external owner exists.

Codec discovery inspects authenticated SDP in PAUSED state, then completes
teardown before opening the one NVIDIA decoder. It does not PLAY a second RTP
video pipeline or decode a probe stream. Live source/decoder properties remain
those in the unchanged camera configuration.
GStreamer's supported [`rtspsrc` SDP signal](https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html#rtspsrc::on-sdp)
provides this information before stream configuration; its callback only signals
the owning thread and performs no state mutation.

Each decode path retains the existing downstream-leaky one-buffer queue and
one-buffer dropping appsink. The single latest shared-memory slot replaces old
presentation frames; analytics frame dropping is not involved in this stage.
Readers reject unpublished, expired, future-timestamped, or malformed frames and
reopen the mapping when a producer replaces its inode. A file's existence is
never proof of live video. Runtime acceptance additionally requires publication
after the current launch and continuing sequence progression.

## Reproducible evidence campaign

```bash
PYTHONNOUSERSITE=1 .runtime/full-stack-venv/bin/python -B -m scripts.freeze_preview_foundation \
  --output .runtime/freeze/F2-six-camera-foundation-new-campaign
```

The campaign runs fresh 30-second smoke, 300-second stability, and 65-second
controlled source teardown/reconnect tests. It records frame-header samples,
decoder counters, sampled queue high-water, runtime element/property inventory,
RTSP sockets, GPU/decoder utilization, memory, fresh/stale transitions, process
shutdown, and F0/F1 protected hashes. The intentional CAM-05 source cycle is
reported as one controlled reconnect, never hidden among spontaneous reconnects.
It tests source-owner recovery, not a physical network outage.

`V11_ENV_FILE` may select an existing authorized credential file. The resolved
file is mounted read-only as `/run/camera.env`; credentials are not embedded in
commands or evidence. Missing authorized credentials fail before opening RTSP.
`--test-reconnect-camera` and `--test-reconnect-at` are explicit test-only source
cycle controls, absent from normal launch.

Stop the attached launcher with SIGTERM/SIGINT. Docker forwards the signal to
the worker, which closes pipelines, unlinks its preview slots, and releases
camera locks. No detached container should remain. Stale prior evidence is
preserved and ignored, not counted as a live success.
