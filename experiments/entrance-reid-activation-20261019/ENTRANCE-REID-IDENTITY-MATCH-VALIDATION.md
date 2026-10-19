# Entrance Room — Re-ID Identity Match Validation

## Result

**Overall: PARTIAL — persistent identity matching remains BLOCKED.**

The preserved DeepStream run supplies a complete same-run crosswalk from every exported Re-ID vector to its source/frame, detector/tracker box row, and emitted MV3DT target ID. It does not supply independent person-identity labels. Therefore no appearance threshold can be calibrated, no returning-person match can be counted as correct, and no persistent IDs were assigned.

| Gate | Result | Evidence |
|---|---|---|
| `FEATURE_TO_TRACK_CROSSWALK` | PASS, scoped to exposed post-tracker object ID | All 34,622 feature rows matched a same-source/same-frame tracker dump row and corresponding MQTT `targetId`; no duplicate same-frame raw-ID feature rows. The dump is the downstream `NvDsObjectMeta.object_id`, not a separately preserved pre-MV3DT local tracker namespace. |
| `FEATURE_TO_GLOBAL_ID_CROSSWALK` | PASS | All 34,622 feature IDs also occur in the same source/frame MQTT target-ID set. Raw IDs remain run-scoped. |
| `POSITIVE_IDENTITY_LABELS` | 0 confirmed | The four candidate items remain `UNCERTAIN`; the earlier GID 8/GID 11 note is a provisional duplicate-box observation, not a verified identity label. |
| `HARD_NEGATIVE_LABELS` | 0 confirmed | No pair has been independently labeled `DIFFERENT_PEOPLE`. |
| `CORRECT_REAPPEARANCE_MATCHES` | 0 verified | The fail-closed replay made zero automatic identity matches. |
| `FALSE_MERGES` | UNMEASURED | No merge was attempted; the false-merge rate cannot be estimated without labeled examples. |
| `PERSISTENT_ID` | BLOCKED | No calibrated identity matcher or persistent-person assignments. |

## Evidence and scope

- Full offline run: `/tmp/entrance-nvdcf-reid-globalid-20261019-run06/`.
- Same-run analysis: `analysis.json` (SHA-256 `07049b21f25425f35133f919eda5afd241512c9a3707cdd6690f83204ce676e5`).
- Fail-closed identity replay: `identity-memory-replay.json` (SHA-256 `d8ba6e434f84bd5fae4ffc53a71a54405ae01a5f7f67af57eb7e8c70f9f42291`).
- Run C completed on the two 5,400-frame videos. Source mapping was source 0 = CAM-02 and source 1 = CAM-05. The preserved run report records the source hashes and DeepStream 9.1 image digest.
- The run exported 34,622 finite, normalized 256-D vectors. Each feature output row contains the metadata object ID plus the vector; its filename provides source/pad and frame number. In the public DeepStream test5 path, `write_reid_track_output()` reads `NvDsObjectMeta.object_id` and the `NVDS_TRACKER_OBJ_REID_META` (`NvDsObjReid`) from the same object. The matching tracker dump supplies that frame’s box, while MQTT supplies the corresponding `targetId`.
- The run's own `/tmp/entrance-nvdcf-reid-globalid-20261019-run06/run/config_deepstream.txt` explicitly sets `kitti-track-output-dir=tracker-kitti-dump` on line 3; the corresponding per-source/per-frame dump files are present under `run/tracker-kitti-dump/`.
- The probe/application output did not serialize `NvDsFrameMeta.buf_pts` alongside each vector. PTS is recovered from the exact input video by its frame index; both inputs have complete 5,400-frame 30-FPS timelines. Example: frame 2220 maps to source PTS 74.000000 s. This is a deterministic source-timeline join, not a direct per-vector PTS field and not proof of cross-camera exposure synchronization.
- Six `ADOPTED_ID` records are preserved in the run: CAM-02 frame 6, 2→3; CAM-05 frame 6, 4→0 and 5→1; CAM-02 frame 2327, 7→6; frame 2779, 9→8; and frame 4173, 15→12. These are tracker lineage events, not independently verified person identities.
- The prior run report notes 98 of 10,790 common source/frame groups have different complete ID sets between feature rows and tracker dumps; all feature IDs themselves are present in the tracker and MQTT records. This set-level difference remains unexplained and is not hidden by the per-row crosswalk result.

The separate pre-MV3DT/local tracker identity is **not** preserved as another ID field in these downstream records. `NvDsObjectMeta.object_id` at the post-tracker probe is the exposed tracker output ID; MQTT `targetId` and `ADOPTED_ID` provide MV3DT target lineage. A replay at that same public downstream probe would re-emit these same fields, not recover an internal identity namespace that is no longer exposed. If that internal ID is essential, it requires a vendor-supported probe/metadata interface before MV3DT adoption or an application point that observes the tracker before the global associator.

## Candidate review clips

The clips are private (directory mode `0700`, files mode `0600`) and generated from the existing videos plus same-run tracker and Re-ID records; no inference was rerun. Boxes and raw IDs are unchanged. The compact ID legend is color-keyed to boxes. For paired clips, each panel shows its own frame and encoded PTS. The shared playback timeline is only a review aid: camera sync remains uncertain (approximately ±150 ms in prior evidence).

| Candidate to review | Clip |
|---|---|
| Prior B-run GID 7→8 time window (run06 numeric IDs are not equated to B) | `identity-review-run06/return_candidate_7_to_8_timewindow_run06.mp4` (72–93 s; 630 frames; SHA-256 `667e81088aea0e985f6cfec12583ce896552fc39e2f0b1a89c563de10a05adcc`) |
| Prior B-run GID 12→13 time window (run06 numeric IDs are not equated to B) | `identity-review-run06/return_candidate_12_to_13_timewindow_run06.mp4` (119–135 s; 480 frames; SHA-256 `76455cfbdb931a37252d83808226f8468f31648be9eab171c5cc8c749dfdf598`) |
| Prior GID 8 versus 10 candidate, CAM-05 | `identity-review-run06/overlap_candidate_8_vs_10_cam05_run06.mp4` (91.5–96 s; 135 frames; SHA-256 `72f355b9f7f04c3f07f3e5c39d4bc9162313c61e4f10ed6c99316fdd7d6cfa74`) |
| Prior GID 8 versus 11 candidate, CAM-05 | `identity-review-run06/overlap_candidate_8_vs_11_cam05_run06.mp4` (96.5–104 s; 225 frames; SHA-256 `4a8a1333eb1651a7831cdcb171585a23c2c7d03c63ce018f9a547eb892b86726`) |

Annotation template: `identity-review-run06/identity-annotation-template.csv`. All four labels deliberately start as `UNCERTAIN`. The old B-run numeric GIDs cannot be mapped by number to this fresh C run; review the clips and annotate visible person continuity, not ID equality.

## Reappearance and duplicate candidates

- **GID 7→8 and 12→13:** the IDs referenced in the request are from earlier run B. Run06 is a fresh run with a different ID namespace. The supplied windows show the corresponding time regions and current run IDs, but do not prove those old-to-new identities. Both remain `UNCERTAIN` pending visual annotation.
- **GID 8 vs 10:** in the current run, CAM-05 has adjacent-frame transitions 8→10 at frame 2782→2783 (box IoU 0.576) and 10→8 at 2843→2844 (IoU 0.492). This is evidence of ID/box continuity candidates only, not proof of the same person or a correctly matchable reappearance.
- **GID 8 vs 11:** current-run CAM-05 has overlapping different-ID boxes at frames 3054, 3087 and 3088; IoU at frame 3087 is approximately 0.615. This agrees with the earlier visual concern about a possible duplicate box. It is an overlap/duplicate-track candidate, not a disappearance/reappearance example and not sufficient to merge the IDs.
- No candidate above has a user-confirmed `SAME_PERSON` or `DIFFERENT_PEOPLE` label in this validation.

## Why no matcher threshold was fitted

The existing replay consumed all 34,622 crosswalked vectors, found zero invalid vectors, skipped zero for missing tracker/MQTT crosswalks, and retained raw IDs. Its policy had no calibrated threshold profile and no validated ground geometry; automatic re-identification was disabled and it produced zero matches. This is the required fail-closed result.

Appearance similarity alone, especially on visually similar clothing, cannot create a ground-truth label. Camera synchronization is uncertain, and the map/world geometry is not independently validated, so spatial/time gates would not currently be trustworthy either. There are zero accepted positives and zero hard negatives; fitting a threshold or reporting precision/recall would be invalid.

## Next safe step

Review the four clips and edit the CSV with `SAME_PERSON`, `DIFFERENT_PEOPLE`, or `UNCERTAIN`, including brief visual rationale and the current-run frame intervals. Do not label a pair from its numeric GID or embedding score. To validate a matcher, collect multiple independently identifiable return sequences and hard negatives (especially similar clothing and simultaneous same-camera tracks), split them into threshold-calibration and held-out sets, then evaluate gallery matching and false merges. Keep the production/raw MV3DT IDs unchanged throughout.

## References

- [DeepStream MV3DT guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html) — documents TRACKLET, STATES, and ADOPTED_ID event semantics; adoption is tracker lineage, not external person ground truth.
- [DeepStream 9.1 Nvtracker metadata API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/sources_2includes_2nvds__tracker__meta_8h_source.html) — exposes Re-ID feature metadata such as feature size and host/device pointers.
- [DeepStream 9.1 NvDsFrameMeta API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/struct__NvDsFrameMeta.html) — identifies source, frame number, buffer PTS, and NTP timestamp fields; those timestamps were not serialized per feature by this preserved run.
