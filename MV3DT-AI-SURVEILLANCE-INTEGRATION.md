# MV3DT offline results → AI Surveillance staging

**Verdict: `STAGING_INTEGRATION_PASS`**

The existing DeepStream 9.1 result events are preserved in a private local bundle, imported into a separate SQLite staging store, exposed through a loopback-only FastAPI replay service, and rendered by a separate PySide6 map UI. The final UI replay consumed all 3,595 frame batches and 12,603 observations from the saved event log. This validates the offline adapter and presentation path only. It does not validate cross-camera identity accuracy, map geometry, or live RTSP behavior.

No calibration, DeepStream inference, camera access, production service, production database, ByteTrack/OSNet identity logic, or four other cameras were touched. The production FastAPI and Qt modules were inspected but not modified. The new package is separate from `services/mv3dt_room/RoomPairState` and its SQLite identity gallery.

## Official MV3DT provenance and preserved artifacts

Used NVIDIA’s installed `deepstream-run-mv3dt` skill, including its custom-dataset, config-generation, and visualization/metadata instructions. The existing run was reused; no MV3DT or calibration job was rerun.

Private permanent bundle (owner-only directory, mode `0700`; files mode `0600`):

`/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z/`

It preserves the full and smoke MV3DT outputs, source/calibration-format dataset copies, full logs and generated configs, visual samples, previous final report, and the original VGGT ZIP. Broker state and stale PID files were intentionally omitted. The copied VGGT ZIP hash matches the original: `2ed665ab9699a4638ab1393bcd61b0a9d8bcdd2ae1bb13b0f3dd3c976ecbd69b`. `SOURCE-MANIFEST.sha256` inventories the preserved bundle.

Key artifacts and hashes:

| Artifact | SHA-256 / properties |
|---|---|
| Full Kafka JSON event log | `f445f7f427d8a755b98432f9c3413cab91301a3c5e0509991cf2e1e3c6d03956` |
| CAM-01 `cam_00.mp4` rectified input | `cf2b6025225cab9df079c9e440d08189f7518cfb4c9d0b6265d68da33bf9a504`; MPEG-4 Part 2, 1920×1080, 30 FPS, 3,600 frames, 120 s |
| CAM-04 `cam_01.mp4` rectified input | `2e285eb031d4e9011b85ee2776ca5b474dc8475e3c665247950ed1a09c6a8ace`; MPEG-4 Part 2, 1920×1080, 30 FPS, 3,600 frames, 120 s |
| VGGT MV3DT ZIP | `2ed665ab9699a4638ab1393bcd61b0a9d8bcdd2ae1bb13b0f3dd3c976ecbd69b` |
| Tiled OSD output | `04407975144c715e0363b2f2700de750c2b5b3bc54af9dc6c2c5f26da341c1ce`; H.264, 1920×1080, 30 FPS, 3,600 frames |
| BEV output | `bfe604b69cb610da8d31f0632f7f78c49e677fb7ea8cd80c2d97525b62ddcded`; 1,156×864, 30 FPS, 1,903 frames / 63.433 s, **partial** |
| Staging SQLite | `0951b4dbea35393ccfe200aa55dc6e28e380d437850e0307a32fcda5ce8370d5` |

Reference run report: [MV3DT-VGGT-TEST-REPORT.md](/home/apsidal/devroom-new-calibration/MV3DT-VGGT-TEST-REPORT.md).

The original result used DeepStream image `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, image ID `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`, PeopleNetTransformer and the MV3DT tracker. The run completed successfully and reached EOS for both file sources. Its runtime evidence remains as reported previously; the adapter only reuses the captured Kafka JSON and artifacts.

## Existing AI Surveillance architecture and adapter boundary

- The production FastAPI forwards health, monitoring, tracked-object and room-pair API calls to the ML service. Its monitoring WebSocket is a separate existing live service contract.
- The ML service obtains six-camera state from native telemetry and exposes latest-only per-camera tracked objects. The frontend is a PySide6 Qt camera wall; the room-pair panel consumes the current room-pair identity response.
- The existing SQLite `IdentityGalleryStore` stores canonical-identity/gallery embeddings, not active per-frame observations. It is not reused.
- The new `services/mv3dt_staging` package has its own importer, SQLite schema, FastAPI app and PySide6 window. The API opens the staging DB with SQLite `mode=ro`/`query_only`; only the offline importer writes it. There is no route to the production API or database.
- The Qt client only accepts loopback `ws://` endpoints; the API launcher rejects non-loopback bind addresses. The staging service uses port `18120`, not a production listener.

The adapter maps MV3DT `sensorId=Camera1` / `source_id=0` to CAM-01 (upper-right), and `Camera2` / `source_id=1` to CAM-04 (lower-left). Each stored row retains camera ID, source ID, output object ID, frame index, event timestamp, nominal frame time, bbox, 2D foot point, world coordinate, MV3DT object-confidence field, and raw source JSON. Missing confidence remains `null`.

Observation uniqueness is `(camera_id, frame_id, mv3dt_global_id)`. Re-importing the same message is idempotent; conflicting duplicate keys fail closed. A shared numeric ID is not collapsed into one coordinate: the map renders separate camera-colored markers and the table retains one row per camera observation. Rows expire from the display after 30 nominal frame indices. The UI labels IDs as MV3DT output IDs, not confirmed people.

Time semantics are explicit: `event_timestamp` is the original message-publication timestamp; per-camera ordering has no timestamp inversions in this log. Kafka JSON has **no source PTS**. Replay position is `frame_id / 30` as a nominal index only; it is not an assertion of source-PTS correspondence or cross-camera synchronization.

## Staging data and display results

Import result: 7,140 frame messages, 12,603 observations, zero duplicate message/observation keys. Counts by camera:

| Camera | Frame messages | Missing messages over expected IDs 0–3599 | Observations | MV3DT output IDs |
|---|---:|---:|---:|---|
| CAM-01 / source 0 | 3,595 | 5 (IDs 0–4) | 6,893 | 0, 1, 5, 6, 10 |
| CAM-04 / source 1 | 3,545 | 55 total (IDs 0–4 plus 50 internal missing IDs) | 5,710 | 0, 1, 2, 3, 4, 5, 7, 8, 9 |

The union has 11 MV3DT output IDs; `0`, `1`, and `5` occur in both camera streams. There is one event with no MV3DT object-confidence value and no missing 3D positions in this log. Those are capture/log facts, not a statement about the validity of the associations.

The map is the preserved 1200×896 `map.png`; the unmodified row-major `T_ov2px` matrix is `[[100.020558,0,0],[0,-100.020558,0],[0,0,1]]`. In the staging store, 12,592 positions fall within the map image and 11 fall outside; the UI now discloses that count and omits those 11 markers rather than placing them on the edge. The map scale (about 100 px per exported coordinate unit) is **not physically verified**, so UI positions are labeled “world units (unverified),” never metres.

At actual stored frame ID 300 (nominal 10 s), the event data contains two output IDs (`0`, `1`) in both camera streams: four independent camera observations. The PySide6 staging view rendered four markers and four camera-specific table rows from that same API frame. The owner-only screenshot is [mv3dt-map-frame-300-qt-final.png](/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z/staging/evidence/mv3dt-map-frame-300-qt-final.png), with exact API rows in [frame-300-api.json](/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z/staging/evidence/frame-300-api.json). This is a metadata-derived map view, not a new video frame or proof of identity/position accuracy. The underlying real tiled video sample at 10 s is preserved at `visual-check/tiled-10s.png`.

The full loopback Qt/WebSocket replay reached frame 3599 and consumed 3,595 frame batches and 12,603 observations. Final smoke log: [qt-loopback-smoke-final.log](/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z/staging/evidence/qt-loopback-smoke-final.log). The API was stopped after verification; no process is left listening on port 18120.

## Launch commands

Run these in separate terminals from the repository. The API binds only to loopback and reads the private staging store.

Terminal 1 — API:

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

Terminal 2 — PySide6 staging UI:

```bash
cd /home/apsidal/ai_surveillance-gate3j
export STAGE=/home/apsidal/devroom-new-calibration/mv3dt-ai-surveillance-staging-20261017T194214Z
export MV3DT_STAGING_MAP="$STAGE/dataset/map.png"
export MV3DT_STAGING_WS=ws://127.0.0.1:18120/ws/replay
export MV3DT_STAGING_SPEED=10
.runtime/full-stack-venv/bin/python -m services.mv3dt_staging.qt_ui
```

The full replay takes about 12 seconds at 10× nominal speed. Stop the API with Ctrl+C and close the Qt window when finished. This starts only the staging API and window; it does not start surveillance, RTSP, DeepStream or model inference.

## Tests

- Focused adapter/API/WebSocket/Qt tests: `6 passed`.
- Complete existing source-free suite: `577 passed`, `219 warnings`, `8 subtests passed` in 13.32 s. Warnings are existing Starlette/httpx and kafka-python deprecations; no failures.
- `py_compile` and `git diff --check`: PASS. An additional whitespace scan of the new package/tests was clean.
- The real-data endpoint/WebSocket/Qt replay reported 7,140 messages, 12,603 observations, 3,595 ordered frame batches, 0 per-camera event-timestamp inversions, and the 11 out-of-map count.

Run tests with:

```bash
.runtime/full-stack-venv/bin/python -m pytest tests/test_mv3dt_staging.py -q
.runtime/full-stack-venv/bin/python -m pytest tests -q
```

## Accuracy boundaries and future live RTSP plan

The previous MV3DT run report already classified its result `MV3DT_RUNTIME_PASS_GEOMETRY_UNVERIFIED`. The earlier VGGT report says the camera-pair 0/1 pairwise calibration did not complete. Same-frame shared-ID coordinate separations and a qualitative visual sample are not ground truth. Camera synchronization was not independently established; the clips were normalized from 20 to 30 FPS, and frame index does not repair that uncertainty. The map’s 100 px/m scale remains unmeasured. No ID-switch or false-merge rate was measured. The BEV MP4 is partial (1,903/3,600 frame pairs), not a complete run render.

Before any live RTSP integration:

1. Keep this adapter in shadow/staging mode; preserve native per-camera tracker IDs, OSNet and existing production identity state unchanged.
2. Establish the rectification transform for each physical feed and apply the same image-coordinate convention used by the exported `camInfo` projections. Never apply rectified-image matrices to raw frames.
3. Carry actual per-camera source PTS plus a shared monotonic capture-time mapping and source generation/epoch. Do not use wall-clock message timestamps or frame index as synchronized capture time. Validate measured inter-camera offset and uncertainty first.
4. Keep `mv3dt_global_id` namespaced separately from local tracker IDs and person identities; retain one `(camera, source epoch, frame/PTS, output ID)` observation per feed. Define stale, gap, reconnect and out-of-order policies before consuming live events.
5. Validate camera-pair geometry with surveyed/held-out floor points and a physically measured map scale. Assess ID switches/false merges against labeled cross-camera trajectories. Shared output labels alone are insufficient.
6. Only after those gates, build a separately reviewed live shadow adapter and test with explicit authorization. Do not write these experimental IDs into the production identity gallery or production UI/database until that review passes.

## Files added

- `services/mv3dt_staging/event_store.py`
- `services/mv3dt_staging/import_cli.py`
- `services/mv3dt_staging/api.py`
- `services/mv3dt_staging/presentation.py`
- `services/mv3dt_staging/qt_ui.py`
- `services/mv3dt_staging/README.md`
- `tests/test_mv3dt_staging.py`
- `MV3DT-AI-SURVEILLANCE-INTEGRATION.md`

All pre-existing dirty worktree changes were left untouched. The new integration is not attested for, selected by, or deployed to production.
