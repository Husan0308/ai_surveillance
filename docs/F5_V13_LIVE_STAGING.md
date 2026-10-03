# F5: exact YOLO26m + V13 live staging

This is an opt-in experimental profile, **not production promotion**. F0–F4
evidence, the PeopleNet room profile, installed native binary, manifest,
calibration, detector settings and V13 remain unchanged. F6 is not started.

## Ownership and deployment

CAM-01/CAM-04 have one native MV3DT RTSP/decode owner, which also produces their
independent latest-only V11 previews. CAM-02/03/05/06 have one preview-only owner.
The real PySide6 wall only consumes shared memory. API defaults to 8100, ML to
8101 via the frozen deployment settings. AutoMagicCalib retains port 8000 and
is not stopped, moved or reconfigured.

F1's `.runtime/full-stack-venv` Python 3.12 / Ubuntu GI is used by every host
service and sidecar. The candidate sidecar explicitly includes the frozen
`services/mv3dt_room` directory for the existing OSNet helper's sibling import;
it does not use a global/Trash interpreter or change identity code.

The experimental supervisor captures the existing frontend configuration before
the shared credential `.env` can inject legacy 33 ms / 2,000 ms timer settings.
It retains the declared 16 ms / 200 ms defaults and honors explicit environment
overrides. This is launcher configuration ownership, not a frontend/preview code
change; the frozen F3 source and its evidence remain untouched. Both launch
environments and the initial failed latency run are retained in F5 evidence.

## Pinned assets

- Frozen tracker: `config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml`
  SHA256 `5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834`.
  The staged tracker is a byte-identical copy, not a threshold overlay.
- YOLO26m raw one-to-many FP16 engine: `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16`.
  RGB 640×640 letterbox, person only, confidence 0.25, NMS IoU 0.45, interval 0.
  Its explicit dynamic batch profile permits the two-source batch=2 configuration.
  Only batch=6→2 changes in the copied nvinfer config; the application's engine
  override also points to YOLO, never the original PeopleNet engine.
- F4 experimental native binary: `b42701d714610a25a1785e5daff0f39e01b500279a7564050fa95e6cba5237bf`.
  Its staged source is `90a36e2528491aa0ab140e2c22aafbe57169c8563df85a6fddd7520cdeb70b52`;
  the unmodified repository source is
  `ec9f86da89b4e41c97173c49b052eca82e15df24886abed02bb012130528257c`.
  Build provenance and SDK inputs are reverified before every launch.
- Same BodyPose3DNet ONNX `0452b785a70fcd6bc5bd4069249bdfd85eb139c9e9216bcf81f89df33945d028`.
  Its old engine is incompatible with F4's validated TensorRT runtime patch
  version. A separate FP16/batch=1 engine is built in staging:
  `d4c7570dff27016aa0c488323a30ef8f6c2f7ed98b92cc1e4b8a2c11a4120b8b`.
  No pose model, V13 pose settings, production engine, or calibration is changed.

The pre-existing bare production source/binary/manifest drift remains an explicit
deployment blocker. This stage verifies a separate hash-pinned experimental
chain; it does not change expected accepted hashes to silence the asset gate.

## Tools

```bash
.runtime/full-stack-venv/bin/python -B -m scripts.run_v13_room_candidate \
  --dataset canonical --output .runtime/new-v13-canonical

.runtime/full-stack-venv/bin/python -B -m scripts.run_v13_room_candidate \
  --dataset person-present --output .runtime/new-v13-person-present

.runtime/full-stack-venv/bin/python -B -m scripts.run_v13_room_candidate \
  --dataset empty-room --duration 700 --output .runtime/new-v13-empty

.runtime/full-stack-venv/bin/python -B -m scripts.validate_v13_full_stack \
  --duration 300 --output .runtime/new-v13-six-camera
```

Each output must be new. Every replay uses complete hash-verified archived inputs
and a fresh SQLite gallery, fresh Kafka topic and isolated MQTT namespace. Model
and calibration mounts are read-only. Containers use the existing GPU runtime
without `--privileged`; host networking is for existing RTSP/Kafka/MQTT access.
Only owned processes/containers are stopped on shutdown. Brokers and AMC remain.

`scripts.v13_monitoring` is an external observer of counters and shared memory;
it adds no work or locks to the CUDA preview worker. It reuses the frozen F3
sequence/session/age freshness checks. Missing native error telemetry remains
unknown rather than invented zero; bus/runtime errors are audited from logs.

After a host reboot, the historical Kafka `/tmp` storage may be absent. If no
broker is listening, `scripts.run_staged_kafka --output <new .runtime directory>`
can create an isolated broker using the installed official configuration with
only `log.dirs` changed. It refuses occupied ports/existing output, formats only
new storage, and never reformats previous broker data.

## Evidence and scope

Evidence lives under `.runtime/freeze/F5-v13-live/`, never in git. Unit/static
checks and real runs are separate. Exact same-frame detector-proposal retention
is **not physical-person recall**. Reviewed physical samples and their missing
boxes/identity changes are reported separately. Empty-room safety requires the
complete 12,031 frames per camera, zero published tracks/IDs/BEV, not merely a
zero-object average. Native PGIE/tracker frame counts are independently checked;
FFmpeg software-decoder error concealment is not used to trim the archived HEVC
input. Raw warnings/errors remain preserved.

Canonical replay requires exactly Person_01/Person_02 and the unchanged formal
auditor. The busy-scene reference has pre-existing identity instability, which
remains explicit for F6; F5 does not redesign identity or claim it is frozen.
The old calibration's coordinates are not independently proven meters. Metric
calibration, real-person live recall, long-soak release acceptance and production
promotion remain separate gates.

## Current disposition: NOT FROZEN

Passing replays do not make the live integration pass. The original F4 binary
passed all complete V13 replay regressions, but a real occupied six-camera run
contained a 50-frame CAM-04 detector-present association deficit. Public metadata
shows the same native target becoming INACTIVE and later returning; NvDCF does
not export the private rejection reason. No geometry/association threshold cause
is asserted solely from that metadata, and V13 has not been tuned.

Two occupied live runs also required forced native shutdown. The captured
backtrace locates a blocked native MV3DT association worker in
`MultiViewAssociator::processReceivedMessagesToCurrentFrame`, with tracker flush
and pipeline teardown waiting behind it. This localizes the blocked component,
but does not establish the mutex owner or a safe fix inside the closed library.

`scripts.build_v13_shutdown_candidate` builds an **unaccepted diagnostic
hypothesis**, not an installed fix: it skips explicit injected EOS only during
RTSP teardown while retaining the normal parent NULL transition. URI replay's
EOS finalization remains unchanged. Repository/SDK sources are not overwritten.
The separately built binary is
`be8e499b05e7628b444a2d21cba49e5fd475d20e1e1ecb50e885e481c7f1ba05`;
its staged SDK application source is
`6fe7d9b929cd8697b91a011c0a3aab211f6d3f519a900b48cc9d8cba6725c9fa`.
The experiment does not change steady-state inference/tracking/preview code.
Optional `--native-build` validates the precise override and all build inputs;
optional `--shutdown-debug` records a container-local, read-only GDB backtrace
only after graceful shutdown has already timed out. Neither option promotes it.

The diagnostic binary exited cleanly in a five-minute **empty live room**, but
that does not prove occupied-scene shutdown fixed. CAM-01 total UI-paint latency
P95 was 40.181756 ms, above the retained 40 ms check. Empty live-room retention is
unmeasured (0/0), never converted to 100%. A separately preserved post-mortem
report is always FAIL and cannot replace the original live acceptance. A large
green strip was visible in the saved CAM-04 startup preview; its source/copy
origin is not yet localized, so no glitch-free visual acceptance is claimed.

Recorded-video RTSP transport diagnostics are explicitly **not live camera
acceptance**. One original-binary run stopped cleanly; two diagnostic-binary runs
stalled, including one with a fresh RTSP server. The latter stopped progressing
inside native association and source reset subsequently waited on the blocked
pipeline. Their compressed B-frame PTS backsteps/correlation misses also make
them unsuitable for latency acceptance. These results do not demonstrate a
causal benefit from the teardown hypothesis. All failing runs are preserved.

`scripts.freeze_v13_live` requires an identical binary/source/tracker/pose/PGIE
chain and fresh galleries across live and all three full replays. It retains
all latency rows, tests empty input independently, and refuses to combine
successful replays from one binary with a different live candidate. F5 must
remain FAIL until actual occupied live tracking, sustained native operation,
clean shutdown, and the full six-camera acceptance are proven. No F6 changes,
production promotion, manifest update, or release tag are authorized here.
