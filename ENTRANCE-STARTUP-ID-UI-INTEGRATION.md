# Entrance Room — Startup ID Alias UI Integration

**Verdict: DISPLAY_ID_STABLE** for the three preserved frame-6 adoption events in this isolated offline replay. This is a display-only result, not identity-accuracy validation or production approval.

## What changed

The separate FastAPI/PySide6 staging replay now optionally applies aliases only when an adoption sidecar is bound to the exact imported Kafka log hash. The sidecar is derived from the preserved diagnostic alias map and MQTT evidence. It verifies each `ADOPTED_ID` (`type=3`) against the decoded MQTT record, source and peer `TRACKLET` records, camera/source mapping, IDs, timestamps, match counts, and source-log SHA-256. Invalid scope, hash mismatch, cycles, duplicate/conflicting identities, ambiguous chains, overlapping same-camera merges, or ambiguous peer identities are rejected. Retrospective aliases are accepted only for `offline_replay_only`; there is no live alias path.

For CAM-02/source 0, the three frame-6 events produce these labels on pre-adoption records through frame 5, then the emitted IDs are used unchanged from frame 6 onward:

| Raw ID before adoption | Display ID | Adoption frame |
|---:|---:|---:|
| 1 | 2 | 6 |
| 3 | 5 | 6 |
| 4 | 0 | 6 |

The UI has an optional **Apply confirmed ADOPTED_ID aliases (display only)** control. Raw MV3DT ID and display ID are separate columns; camera/source, frame, event publication timestamp, bounding box, and unverified world coordinates remain available in the row tooltip. Canonical per-camera BEV observations and coordinates are unchanged. The map keeps its existing one-primary-marker-per-selected-ID presentation and chooses an existing observation deterministically (it does not average or move coordinates); **Show camera observation points** exposes the separate camera observations. A recent pre-adoption row is hidden from the active table only when its adopted current row is present, preventing a duplicate active label; the original raw message remains in the read-only SQLite store and API.

## Verification

- Preserved run-03 MQTT evidence: 4,623 parsed records, 0 parse failures, and 3 `ADOPTED_ID` records. All three source boxes persisted across frames 5→6 (IoU 0.989–0.993); the diagnostic report identifies the peer tracklet matches.
- Camera mapping: `Camera1`/source 0 = CAM-02; `Camera2`/source 1 = CAM-05.
- Source-bound alias manifest: Kafka-log SHA-256 `2c95349ce8bf86758152f55ee16fc2e22c6af645c484daaa93b723654e43afe0`; raw MQTT SHA-256 `1be52128fa4079f0668d7c79204fe8a24ebdda973d258bb54bcfb23f1c72e2d4`.
- Actual FastAPI replay and PySide6 callbacks were exercised offline over the full preserved run range: 595 shared frame IDs (5–599), 1,190 camera frame messages, and 3,420 object observations. Frames 0–4 are absent from the source event log; this is reported rather than filled in. The three aliases applied to the available pre-adoption rows; no same-camera frame contained multiple raw IDs collapsed to one display ID. At the frame-6 boundary, the active table showed six current camera observations with three IDs per camera; three superseded pre-adoption rows were identified and excluded from the active list while remaining in the event store/API.
- The diagnostic contact sheet visibly shows three separate boxed people in CAM-02 at frames 5 and 6 and three separate CAM-05 tracks in its frame-6 view. The aliases preserve three distinct display labels per camera. No within-camera merge of these visually distinguishable tracks was observed. The actual cross-camera person correspondence is **not independently proven**: source PTS is absent and synchronization/calibration remain uncertain. An algorithmic adoption event is not identity ground truth.
- Event database SHA-256 was unchanged by replay. The staging API uses read-only database connections.
- Focused regression suite: **17 passed**, one upstream Starlette/httpx deprecation warning. `git diff --check -- services/mv3dt_staging tests/test_mv3dt_staging.py`: PASS.

## Private screenshots

Owner-only evidence directory: `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/startup-id-alias-ui-20261019/`

- Raw frame-5 IDs: `before-raw-ids-frame5.png` — SHA-256 `aac0c5d9a7fef230345001ab6c72d33b1f3d14833deb9552a56005f7bf323c0d`
- Aliases enabled at frame 5: `after-adoption-aliases-frame5.png` — SHA-256 `02b261f8cd17f41337de4b322303cda8902f9da06243063eafc7e3d62fb64c1c`
- Post-adoption frame 6: `stable-adopted-ids-frame6.png` — SHA-256 `51878d2b7e5c38e52b67a64ae39905e09dda796e61c261f039affd5ddb30e67c`

These are headless/offscreen PySide6 renders of real preserved event data and the preserved Entrance map, not camera-video frames, live camera, or RustDesk evidence.

## Remaining limits and launch

The aliases cover only the three proven startup adoptions. Other relabels in the longer recording remain untouched. The raw event log contains no source PTS; displayed media time is nominal `frame_id / 30`. Camera synchronization, cross-camera identity correctness, and metric map accuracy remain unverified. This does not change tracker settings, calibration, production identity logic, production services, or original recordings. The existing API on port 18120 was not restarted; validation used an in-process test client. The README contains commands for a separate loopback-only API on port 18121 and the Entrance Room Qt reviewer.

The adoption-event interpretation follows NVIDIA's documented MV3DT `ADOPTED_ID` event semantics: [DeepStream 9.1 MV3DT documentation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html). It does not certify that the camera pairing is a correct physical-person match.
