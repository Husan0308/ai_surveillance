# MV3DT duplicate-person display correction

Result: **the reported final replay state now shows two primary people markers, while preserving four camera observations and three raw global IDs.** The corrected offline staging window is open on DISPLAY=:0. No live cameras, calibration, inference, production services or production identity logic were changed.

## Exact frame and visible people

The reported combination occurs at the final frame **3599**. Both original rectified source videos were independently decoded to that exact zero-based index; FFprobe reports source PTS **119.966667 seconds** for both files. This establishes file-frame correspondence, not independently synchronized physical capture.

Exact metadata and visual inspection establish:

| Camera / output ID | What its box covers at frame 3599 | Finding |
|---|---|---|
| CAM-01 / 1 | Seated person in a light shirt | Real person; no ghost-track evidence in this frame |
| CAM-01 / 5 | Seated person in a dark shirt, with a wider box | Real person, duplicated by ID 10 |
| CAM-01 / 10 | The same dark-shirt person, with a narrower overlapping box | Extra detection/track fragment; not a third visible person |
| CAM-04 / 5 | Dark-shirt person from the other camera | Second camera observation of the same reviewed person |

There are **2 visible people, 3 distinct MV3DT output IDs, and 4 camera observations**. These are different quantities.

Private exact-frame evidence is preserved in:

`/home/apsidal/devroom-new-calibration/mv3dt-duplicate-review-XqzjY1/`

- `CAM-01-frame-3599-annotated.png` and `CAM-04-frame-3599-annotated.png`: exact source frames with the actual saved tracking boxes/IDs.
- `cam_00-sample-06.png` and `cam_01-sample-06.png`: corresponding unannotated decoded frames.
- `frame-correlation.json`: exact source PTS and camera/frame/bbox/confidence records for 12 sampled camera frames.
- `duplicate-metrics.json`: ID lifetime, overlap and detector-dump findings.
- `reviewed-frame-3599-api.json`: all four unchanged source observations plus the explicit visual-review annotation.
- `desktop-corrected-staging-final.png`: actual corrected X11 staging window, 1854×1011 capture.
- `desktop-window-properties-final.txt`: window/PID/geometry evidence.

## Proven cause

Two issues combined:

1. The old map renderer drew one filled person marker per camera observation. A shared global ID seen by two cameras therefore appeared twice.
2. The existing MV3DT result also contains upstream duplication/fragmentation. CAM-01 ID 5 last appears before a gap at frame 3157, ID 10 starts at 3187, and ID 5 reappears at 3198. IDs 5 and 10 coexist in 117 frames through 3599. Their median image-box IoU is 0.485690, and the median fraction of the smaller box covered is 0.991420.

At frame 3599, the CAM-01 detector dump itself contains three Person boxes for two visible people. Its two dark-shirt boxes are the same boxes emitted by tracker IDs 5 and 10. Their IoU is 0.520556; 98.8143% of the smaller box lies in the larger box. Detector confidences for those boxes are 0.420778 and 0.382141; the light-shirt box is 0.958070. The separate Kafka object-confidence values are retained with their existing MV3DT field name and are not relabeled as detector confidence.

On the 117 co-occurrence frames, the detector dump has three boxes in 109 frames and four boxes in eight frames. This supports a persistent upstream duplicate/fragmentation problem, but only the explicitly reviewed frame is given a confirmed-person grouping. The exact lower-level association/NMS cause is not established. No calibration matrix or association threshold was changed.

CAM-04's final detector dump also sees the light-shirt person (confidence 0.976891), although the tracker/event output contains only ID 5 for that camera. The other real person remains represented by CAM-01 ID 1. Thus the extra ID 10 is not evidence of another person, and the absent CAM-04 ID 1 is not evidence that the light-shirt person left the room.

## Display correction

- Current observations are grouped by global MV3DT output ID. Each ID has one primary marker, selected deterministically from one camera observation; camera coordinates are not averaged.
- Camera observations remain as separate table rows. The **Show camera observation points** checkbox exposes their individual map coordinates.
- The UI has separate counters for current camera observations, unique global IDs, confirmed visible people and primary markers. Recent retained rows are labeled and excluded from current counters.
- Confirmed visible people is **UNAVAILABLE** for unreviewed frames. A new sidecar `frame_reviews.json` explicitly records the final-frame visual finding: ID 10 is a duplicate fragment of ID 5 in frame 3599. The primary map therefore shows ID 1 and ID 5; ID 10 remains visible in the table as “reviewed duplicate of ID 5.”
- The review is checked against the exact SQLite hash, source-log hash, both source-video hashes, exact frame and expected observation keys. It creates no permanent alias and is never propagated to another frame or dataset. Different IDs remain separate without that evidence.
- End-of-replay WebSocket closure is now normal, and the Qt completion status remains visible instead of being overwritten by a misleading connection error.

Final reviewed state on the actual desktop:

| Counter | Value |
|---|---:|
| Current camera observations | 4 |
| Unique global output IDs | 3 |
| Confirmed visible people, exact frame 3599 | 2 |
| Primary map markers | 2 |

The pre-existing offline staging API and Qt processes were individually verified by UID, working directory, exact module arguments and process handles before being refreshed. Current staging API PID: 944965; Qt PID: 944983; X11 window: `0x7400007`. Only these offline staging services were refreshed. The current Qt log confirms full replay completion: 3,595 frame batches, 12,603 observations, 3 raw IDs at the final frame, 2 primary markers and 2 confirmed visible people. Operator personal visual confirmation is not claimed.

## Evidence integrity and tests

The original SQLite database is byte-for-byte unchanged before and after investigation, API reads and desktop replay:

`0951b4dbea35393ccfe200aa55dc6e28e380d437850e0307a32fcda5ce8370d5`

Original videos, calibration files, map transforms and archived evidence were not changed. Existing unrelated dirty worktree files were preserved.

- Focused staging regressions: **10 passed**.
- Complete source-free suite: **581 passed**, 8 subtests passed, 219 existing dependency deprecation warnings, 17.13 s.
- Python compilation and `git diff --check`: PASS.
- Actual source-database REST and WebSocket review delivery: PASS; neighbouring frame 3598 has no confirmed-person annotation.
- Actual Qt/X11 replay and screenshot: PASS; four evidence rows retained and two primary markers rendered.

Regressions cover same-ID observations from two cameras, distinct people even at identical coordinates, reviewed duplicate IDs, stale-frame exclusion, exact-frame review scope, database-hash mismatch rejection, unchanged SQLite reads, Qt counters/table preservation, and normal replay completion.

Files changed: `services/mv3dt_staging/{presentation.py,qt_ui.py,api.py,README.md}`, new `services/mv3dt_staging/{reviews.py,frame_reviews.json}`, `tests/test_mv3dt_staging.py`, and this report.

## Remaining limits

The UI now correctly represents the two visible people in the reported reviewed state. Upstream duplicate boxes and fragmented MV3DT IDs remain preserved in the offline result and may occur in other frames. Those unreviewed frames show raw unique IDs with confirmed-person count unavailable. This correction does not establish dataset-wide identity accuracy or repair unverified ground-plane geometry, map scale or camera synchronization.
