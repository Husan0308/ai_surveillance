# Dev Room MV3DT Offline Staging

This is a separate, loopback-only replay adapter for the preserved DeepStream
9.1 MV3DT event log. It does not import production `RoomPairState`, use its
SQLite identity gallery, edit the six-camera frontend, or read/write production
databases. The UI is a separate PySide6 window and the API opens its staging
SQLite database read-only.

Before launch, set `STAGE` to the private evidence bundle and start the API:

```bash
cd /home/apsidal/ai_surveillance-gate3j
export STAGE=/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z
export MV3DT_STAGING_DB="$STAGE/staging/events.sqlite3"
export MV3DT_STAGING_MAP="$STAGE/dataset/map.png"
export MV3DT_STAGING_TRANSFORMS="$STAGE/dataset/transforms.yml"
export MV3DT_STAGING_HOST=127.0.0.1
export MV3DT_STAGING_PORT=18120
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.api
```

In a second terminal, launch the separate Qt staging window:

```bash
cd /home/apsidal/ai_surveillance-gate3j
export STAGE=/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z
export MV3DT_STAGING_MAP="$STAGE/dataset/map.png"
export MV3DT_STAGING_WS=ws://127.0.0.1:18120/ws/replay
export MV3DT_STAGING_SPEED=10
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.qt_ui
```

The default adapter port is `18120`, distinct from the production API/UI ports.
The API rejects non-loopback bind addresses. The Qt client replays the full
recorded event range at 10× nominal speed; use a lower positive speed for a
slower review. Ctrl+C stops the API; closing the Qt window ends that client.

The map shows one primary marker per current MV3DT output ID, using a stable
representative camera observation without averaging positions. The table keeps
every camera observation; "Show camera observation points" exposes their separate
coordinates. Recent rows are labeled and excluded from current counters.

Camera observations, unique global IDs and confirmed visible people have separate
counters. Confirmed people remain unavailable unless a source-hash-bound visual
review covers the exact frame. `frame_reviews.json` records the reviewed final
frame 3599: four observations, three output IDs, two visible people. ID 10 is an
explicit reviewed duplicate of ID 5 in that frame and does not get another primary
person marker. ID 10 remains in the table and source data. No permanent ID alias or
proximity-based merge is made. A different database/video hash disables the review.

All labels and the banner identify these as experimental outputs.
The original event timestamp is a message-publication timestamp. The displayed
media position is only `frame_id / 30`; no source PTS is present in the Kafka
text dump. The exported map scale is not physically verified.

The database importer is only needed to recreate a new staging store:

```bash
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.import_cli \
  --event-log "$STAGE/full-kafka-client.log" \
  --db "$STAGE/staging/events.sqlite3" \
  --fps 30 \
  --cam00-sha256 cf2b6025225cab9df079c9e440d08189f7518cfb4c9d0b6265d68da33bf9a504 \
  --cam01-sha256 2e285eb031d4e9011b85ee2776ca5b474dc8475e3c665247950ed1a09c6a8ace \
  --calibration-sha256 2ed665ab9699a4638ab1393bcd61b0a9d8bcdd2ae1bb13b0f3dd3c976ecbd69b
```

## Entrance Room: Confirmed Startup Adoption Review

The Entrance Room diagnostic uses the preserved 20-second DeepStream run-03
event log and its MQTT `ADOPTED_ID` evidence. It is isolated from the Dev Room
replay and from production. Camera mapping is explicitly `Camera1`/source 0 =
CAM-02 and `Camera2`/source 1 = CAM-05.

Create a fresh private replay database from the preserved Kafka text log:

```bash
DIAG=/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/startup-id-diagnostic-20261019-run-03
STAGE=/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/startup-id-alias-ui-20261019
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.import_cli \
  --event-log "$DIAG/logs/kafka_events.log" \
  --db "$STAGE/events.sqlite3" --fps 30 --expected-frames 600 \
  --camera-profile entrance-room
```

Build the alias sidecar from the preserved alias map, decoded MQTT records,
raw MQTT capture, and the exact imported Kafka-log source hash:

```bash
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.adoption_aliases \
  --alias-map "$DIAG/logs/display_id_alias_map.json" \
  --mqtt-decoded "$DIAG/logs/mqtt_decoded.jsonl" \
  --event-log "$DIAG/logs/kafka_events.log" \
  --mqtt-raw "$DIAG/logs/mqtt_peer.raw" \
  --output "$STAGE/adoption_aliases.json"
```

For a separate loopback-only API instance (the existing port 18120 remains
untouched), start:

```bash
export MV3DT_STAGING_DB="$STAGE/events.sqlite3"
export MV3DT_STAGING_MAP="$DIAG/input/map.png"
export MV3DT_STAGING_TRANSFORMS="$DIAG/input/transforms.yml"
export MV3DT_STAGING_ADOPTION_ALIASES="$STAGE/adoption_aliases.json"
export MV3DT_STAGING_HOST=127.0.0.1
export MV3DT_STAGING_PORT=18121
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.api
```

In another terminal, launch the separate PySide6 review window:

```bash
DIAG=/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/startup-id-diagnostic-20261019-run-03
export MV3DT_STAGING_MAP="$DIAG/input/map.png"
export MV3DT_STAGING_WS=ws://127.0.0.1:18121/ws/replay
export MV3DT_STAGING_ROOM_TITLE="Entrance Room"
export MV3DT_STAGING_SPEED=10
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.qt_ui
```

For a full-duration pre-rendered three-panel replay that includes the two camera
views and Entrance map, set `MV3DT_STAGING_REVIEW_VIDEO` to the exact review MP4
before launching the same Qt client. The video player controls (play/pause and
seek) drive the loopback event replay using the nominal frame-index timeline;
the normalized MV3DT events do not contain source PTS, so this does not establish
camera exposure synchronization. The default map/table-only UI remains unchanged
when the variable is unset.

```bash
export MV3DT_STAGING_REVIEW_VIDEO="/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/id-stability-20261018-B/entrance-three-panel-id-stability-B.mp4"
export MV3DT_STAGING_WS=ws://127.0.0.1:18122/ws/replay
export MV3DT_STAGING_ROOM_TITLE="Entrance Room B — ID correction off (experimental)"
export MV3DT_STAGING_SPEED=1
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.qt_ui
```

The checkbox **Apply confirmed ADOPTED_ID aliases (display only)** is available
only when the manifest is valid and its Kafka-log SHA-256 matches the imported
database. The offline replay can apply a confirmed later adoption
retrospectively to pre-adoption frames. It always displays raw ID and selected
display ID separately; event timestamps, camera/source IDs, boxes, and BEV
coordinates are retained. Alias validation rejects hash mismatch, cycles,
conflicting identities, overlapping same-camera merges, and ambiguous target
lineages. The replay-only alias code is not connected to any live endpoint or
production identity logic. The event publication timestamp is not source PTS,
and a confirmed software adoption event is not independent proof of real-person
identity correctness.

Private UI comparison captures from this run:
`/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/startup-id-alias-ui-20261019/`

Tests:

```bash
.runtime/full-stack-venv/bin/python -m pytest tests/test_mv3dt_staging.py -q
```

