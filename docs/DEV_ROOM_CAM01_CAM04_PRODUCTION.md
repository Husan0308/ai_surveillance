# Dev Room CAM-01 + CAM-04 production profile

This profile keeps MV3DT scoped to CAM-01 and CAM-04. CAM-02, CAM-03,
CAM-05, and CAM-06 are live preview-only tiles in the production UI; they are
not added to the Dev Room detector, tracker, identity, calibration, or BEV
graph.

## Production data flow

```text
CAM-01/CAM-04 replay files or configured RTSP sources
  -> accepted DeepStream PeopleNet 2.6.3 FP16 binary
  -> causal bbox recovery before downstream tracking/projection
  -> NvDCF + ObjectModelProjection + VGGT calibration + MVA
  -> Kafka native observations (native IDs remain debug metadata)
  -> event-driven DeepStream object crops over the crop metadata socket
  -> asynchronous CUDA OSNet micro-batches
  -> GlobalIdentityManager KNOWN/PENDING/NEW_CONFIRMED lifecycle
  -> latest-only application identity state
  -> API service
  -> camera overlays + strict-current-presence shared BEV

CAM-02/03/05/06 configured RTSP sources
  -> NVIDIA decode preview worker
  -> V11 latest-only shared memory
  -> preview-only production UI tiles
```

### Independent room-pair preview prerequisite

The room-pair preview is tapped before analytics, not from tracker output.
CAM-01 uses the decoded source-bin output; CAM-04 fans out at decoder output
before its source queue. CAM-04 requests the preview tee pad first and uses a
forced GPU surface copy on the analytics leg. Both cameras detach the decoded
surface before the non-leaky analytics handoff (2 buffers for CAM-01, 16 for
CAM-04). These bounded queues preserve analytics frames; they are not leaky.
Mux, PGIE, tracker, calibration, and identity configuration is unchanged.

Each preview worker has one replaceable pending frame. Conversion uses a
direct CUDA copy into registered shared memory without a full-frame CPU BGRA
copy. Protocol v4 carries camera-local PTS, decoder reference/output times,
and source sequence through actual UI paint. Decoder readiness gating is a
separate prerequisite consumer: no WARMING/READY barrier is implemented here.

`config/cameras.yaml` selects the validated IP-only low-latency decoder
profiles, including CAM-04's eight extra surfaces. It does not change RTSP
latencies or transport. Do not apply these profiles to a new B-frame stream
without validating decoder compatibility first.

The detector, tracker, projection, calibration, MVA, identity, and BEV code is
identical in replay and live modes. Only the CAM-01/CAM-04 source URI staging
changes.

## Source mode

Select the source mode with one setting:

```bash
export MV3DT_DEV_ROOM_SOURCE_MODE=replay
# or
export MV3DT_DEV_ROOM_SOURCE_MODE=live
```

The equivalent explicit CLI is:

```bash
python scripts/dev_room_mv3dt/run_room_pair.py --mode replay
python scripts/dev_room_mv3dt/run_room_pair.py --mode live --duration 180
```

Replay uses the synchronized configured files and changes only the staged
display sink to clock synchronization. The 2400-frame, 20 FPS inputs therefore
take approximately 120 seconds of wall time. Live resolves CAM-01 and CAM-04
from `config/cameras.yaml` and environment credentials without printing them.

Each invocation creates a new runtime session under
`.runtime/mv3dt/dev-room-cam01-cam04/`. The frontend follows the newest
running session and clears cached rows when the session changes, so replay
observations and markers cannot leak into live mode.

## Production UI

Start the local API/ML services using the deployment's configured ports, then
start the preview-only worker and desktop frontend:

```bash
python -m services.camera_v11.preview_only_runtime \
  --cameras CAM-02,CAM-03,CAM-05,CAM-06
FRONTEND_USE_V11_SHARED_MEMORY=1 \
FRONTEND_SHOW_NATIVE_IDS=0 \
python -m services.frontend.app.main
```

The window contains a 3x2 camera wall and the Dev Room shared BEV. CAM-01 and
CAM-04 are badged `REPLAY` in replay mode. Normal overlays show
`Person_XX | CAM-XX`; native IDs require `FRONTEND_SHOW_NATIVE_IDS=1`.
The BEV renders one marker per canonical ID, lists its currently active camera
sources, and removes it as soon as neither Dev Room camera observes it.

### Production publication readiness

The native `run/logs/probe/readiness.json` is the single room-pair readiness
signal. Both sources must have current mux, PGIE, and tracker progress; the
existing native five-second stall/recovery policy is unchanged. The runner
always enables this signal, including when diagnostics are not requested.

The identity sidecar starts in **WARMING**. Source ingestion, inference,
tracking, crops, OSNet, identity resolution, and independent camera previews
continue normally. Production identity rows/events and current presence are
withheld; `current_state.json` explicitly contains an empty `people` list and
`publication.ready=false`. PENDING/Unknown observations are diagnostic, not
production canonical presence.

When native readiness becomes ready, publication enters **READY**. Only
observations received after that opening can be published. Withheld rows and
events are discarded, not queued or flushed; valid identity/gallery state is
not reset. A loss of native readiness closes the same barrier and clears
current presence. Recovery opens a new publication epoch and requires fresh
observations again. The API independently checks the same native readiness,
including the native age limit if its producer stops, so a frozen worker's
old snapshot cannot remain current BEV/overlay output.

Transitions and monotonic opening times are recorded in
`identity-live/identity_path_trace.jsonl`; the state/report includes
`publication`/`publication_barrier`, its epoch, and zero deferred-event queue
depth. This is a Python publication boundary only: no video pad, native
preview worker, decoder, analytics queue, or identity threshold is gated.

## Validation

Verify frozen Dev Room assets:

```bash
python scripts/dev_room_mv3dt/verify_validated_assets.py
```

Audit a completed replay:

```bash
python scripts/dev_room_mv3dt/audit_replay_acceptance.py \
  .runtime/mv3dt/dev-room-cam01-cam04/replay-YYYYMMDD-HHMMSS \
  --preview-stats .runtime/ui-acceptance/final-replay-preview-stats.json
```

The auditor checks canonical allocation, cross-camera coverage, switches,
merges, unresolved final state, strict presence, BEV movement, FPS, queue
bounds, crop delivery, CUDA OSNet, Kafka/MQTT, source readiness, and reconnects.
Preview-only health is non-applicable to a scoped URI replay unless explicitly
supplied. Missing visual telemetry is non-applicable only for skip-render
replay; computed current-presence/BEV remains gating. Supplied incomplete or
failed telemetry cannot be made non-applicable. Native fragment overlap is
diagnostic, not by itself a physical merge or duplicated rendered marker.

The visible identity is always the GlobalIdentityManager application alias.
Face recognition remains separate. Missing or poor crop evidence does not
downgrade a continuous known track or create a new identity by itself.
## Long-gap canonical identity memory

Visible presence and identity memory are separate. When neither Dev Room
camera has a current observation, the worker publishes no person and the BEV
removes the marker immediately. The canonical gallery remains internal in a
bounded SQLite store, whose default location is:

```text
.runtime/mv3dt/dev-room-cam01-cam04/identity_gallery.sqlite3
```

Set `MV3DT_IDENTITY_GALLERY_DB` (or pass `--gallery-db` to the worker) to
select another durable store. The store contains only stable `Person_XX`
mappings and quality-filtered OSNet representatives; it does not store active
tracks or visible positions. A new observation after a long gap enters
PENDING, is matched against the durable gallery with the existing conflict
guards, and only positive novelty evidence can allocate a new application ID.

Validate the captured replay embeddings without waiting for wall-clock gaps:

```bash
python scripts/dev_room_mv3dt/validate_long_gap_reacquisition.py \
  --run .runtime/mv3dt/dev-room-cam01-cam04/replay-YYYYMMDD-HHMMSS \
  --gallery-db /path/to/identity_gallery.sqlite3 \
  --output /tmp/dev-room-long-gap-reacquisition.json
```

The validator covers +5 minutes, +1 hour, +6 hours, strict empty presence
during absence, and a fresh manager reload from SQLite.
