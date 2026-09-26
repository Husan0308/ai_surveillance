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
python -m services.camera_v11.preview_only_runtime
FRONTEND_USE_V11_SHARED_MEMORY=1 \
FRONTEND_SHOW_NATIVE_IDS=0 \
python -m services.frontend.app.main
```

The window contains a 3x2 camera wall and the Dev Room shared BEV. CAM-01 and
CAM-04 are badged `REPLAY` in replay mode. Normal overlays show
`Person_XX | CAM-XX`; native IDs require `FRONTEND_SHOW_NATIVE_IDS=1`.
The BEV renders one marker per canonical ID, lists its currently active camera
sources, and removes it as soon as neither Dev Room camera observes it.

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
bounds, crop delivery, CUDA OSNet, Kafka/MQTT, and all four preview-only
cameras.

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
