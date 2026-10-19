# Entrance Room — NVIDIA DeepStream 9.1 Global ID and Re-ID Validation

**Verdict: `PERSISTENT_REID_PARTIAL`**

The isolated DeepStream 9.1 run used MV3DT peer messaging and NvDCF `reidType: 2`; the TAO ReIdentificationNet TensorRT engine loaded, and the tracker emitted valid 256-dimensional normalized features across the complete 180-second clips. This verifies the configured tracker Re-ID inference/output path. It does **not** establish reliable person identity after a terminated track, accurate cross-camera association, or a locally calibrated matching threshold. The fail-closed persistent-identity layer accepted no merges.

No calibration, RTSP connection, production service/database change, tracker deployment, or raw-event modification was performed.

## Results

| Requested result | Status | Evidence / scope |
|---|---|---|
| Official Global ID configuration | **PASS** | Isolated NvMultiObjectTracker run completed with `MultiViewAssociator` type 1, MQTT communicator type 2, reciprocal peer topics, camera-model paths ordered CAM-02 then CAM-05, `enableMsgSync: 1`, Late Peer Re-Assoc enabled, and ID Correction separately set to 0. TRACKLET, STATES, and ADOPTED_ID messages were captured. This is a configuration/protocol pass, not proof that the calibration or physical identities are correct. |
| Real Re-ID inside tracker | **PASS** | NvDCF loaded the Re-ID TensorRT engine, ran as part of the tracker, had `reidType: 2`, `enableReAssoc: 1`, nonzero Re-ID score weight, and emitted feature vectors tied to per-frame tracker records. Individual reassociation decisions/scores are not exposed in the preserved logs, so behavioral benefit is not established. |
| Local lost-target reassociation | **PARTIAL** | Re-ID-based reassociation is enabled. No manually confirmed same-person disappearance/reappearance sequence establishes a successful reacquisition. At 30 FPS, `maxShadowTrackingAge: 162` is about 5.4 seconds; this is finite track retention, not persistent identity. |
| Cross-camera ID consistency | **PARTIAL** | Peer graph and messages worked; eight numeric target IDs appeared on both camera topics. Numeric overlap is not identity ground truth. Pairwise camera geometry and capture-time synchronization remain unvalidated. |
| Persistent person reappearance | **PARTIAL** | Separate `PersonIdentityMemory` replay consumed the feature records but made **0 automatic re-identification matches**. Matching was intentionally fail-closed because there is no calibrated threshold profile, verified world-coordinate gate, or labeled positive/hard-negative set. Raw IDs were not changed. |
| False merges | **UNMEASURED** | No verified identity ground truth supports a tracker false-merge rate. The memory layer performed no merges; that is a safety-policy result, not evidence that MV3DT made no false merges. |

## Configuration and official semantics

The experimental tracker configuration was derived from the preserved ID-Correction-off baseline B (`enableIDCorrection: 0`). The diff was limited to:

- `TrajectoryManagement.minMatchingScore4ReidSimilarity`: `0` → `0.4329`.
- `TrajectoryManagement.matchingScoreWeight4ReidSimilarity`: `0` → `0.8356`.
- Added `ReID`: `reidType: 2`, feature size 256, history size 100, tensor output on, and model-specific inference parameters below.

All other B tracker and MV3DT settings were retained. The two nonzero trajectory values come from NVIDIA’s published NvDCF accuracy example; they are **not** calibrated for Entrance Room and must remain experimental. No threshold search or reduction was performed.

Other relevant settings retained: `maxTrackletMatchingTimeSearchRange: 27` for single-view reassociation; `MultiViewAssociator.enableLatePeerReAssoc: 1`, `enableMsgSync: 1`, `recentlyActiveAge: 178`, `minCommonFrames4MatchScore: 2`, peer `maxTrackletMatchingTimeSearchRange: 1`, and `minPeerTrackletMatchScore: 0.48`. At nominal 30 FPS, 178 frames is about 5.93 seconds. None of these values proves real source-clock alignment.

The camera mapping is verified at the configuration/order level: source 0 / `cam_00` → CAM-02 and `camInfo_00.yml`; source 1 / `cam_01` → CAM-05 and `camInfo_01.yml`. The matrix-to-camera ordering is correct; the matrices’ geometric accuracy is **not** independently validated. MQTT topics were reciprocal (`cam1` publishes/subscribes to `cam2`, and vice versa) on a disposable local broker. The main app used the isolated legacy-mux `deepstream-test5-app -c config_deepstream.txt` path with batch size 2 and 1920×1080 input. `useUniqueID: 0` was retained; IDs are run-scoped outputs and are not durable person identifiers across application restarts.

NVIDIA defines `reidType: 2` as Re-ID-based target reassociation and describes an internal recent-feature gallery. It also documents that targets exceeding `maxShadowTrackingAge` terminate and may receive a new ID on return. MV3DT cross-camera matching compares foot-location tracklets on a common time axis; `messageTimeStamp` and frame numbers support alignment, but do not create source synchronization that the recordings do not contain. Accordingly:

- Local tracker ID, MV3DT `targetId`/global ID, an `ADOPTED_ID` lineage event, and application `persistent_person_id` are separate concepts.
- Preserved outputs contain the post-tracker object/target ID and adoption old→new lineage, but not a separate independent pre-adoption local ID for every observation.
- `ADOPTED_ID` is valid tracker lineage evidence, not independent proof of a real-world person match.
- Re-ID does not imply indefinite identity after track termination. Persistent person identity remains an application-layer feature and must fail closed.

Official references: [MV3DT architecture, alignment, messaging, and reassociation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_MV3DT.html), [Nvtracker Re-ID modes, feature gallery, output tensor, and target lifetime](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html), and the [NVIDIA MV3DT reference application](https://github.com/NVIDIA/DeepStream/tree/main/src/apps/reference_apps/deepstream-tracker-3d-multi-view).

## Isolated run evidence

- Run directory: `/tmp/entrance-nvdcf-reid-globalid-20261019-run06/`
- DeepStream image: `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`, image ID `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`.
- Application reached EOS for sources 0 and 1 and logged `App run successful`; tracker deinitialized. Runtime was about 6m14s. The output tiled MP4 independently probes as H.264, 1920×1080, 30 FPS, 5,400 frames, 180 seconds.
- Each camera had 5,400 tracker-dump frame files and 5,400 Re-ID feature-output frame files spanning frames 0–5,399, with no missing frame index in either sequence.
- Re-ID output: 17,073 vectors for source 0/CAM-02 and 17,549 for source 1/CAM-05; all rows parsed as 256 finite values. L2 norm ranges were 0.9999976–1.0000026 (CAM-02) and 0.9999974–1.0000025 (CAM-05).
- The previously unfeatured late candidate interval from the earlier run is covered by this run’s complete feature-frame range. Its old numeric GIDs cannot be assumed to refer to the same objects in this fresh run without a per-frame visual/identity crosswalk.
- Same-source/same-frame crosswalk: all 34,622 feature IDs were found in the corresponding tracker dump. Of 10,790 common stream/frame groups, 10,692 had identical ID sets; 98 did not and are unexplained. No cross-camera numeric-ID merge was used in this crosswalk.
- MQTT subscriber was active before the application. It captured 45,428 payloads. Known message counts: 121 TRACKLET, 34,501 STATES, and 6 ADOPTED_ID. Remaining type-4 payloads are not assigned a semantic name in this report. The eight target IDs appearing on both topics were `0, 1, 3, 6, 8, 12, 13, 14`; these are tracker outputs, not eight proven cross-camera people.

The six observed ADOPTED_ID transitions were:

| Camera | Frame | Old target ID → adopted target ID |
|---|---:|---:|
| CAM-02 | 6 | 2 → 3 |
| CAM-05 | 6 | 4 → 0 |
| CAM-05 | 6 | 5 → 1 |
| CAM-02 | 2,327 | 7 → 6 |
| CAM-02 | 2,779 | 9 → 8 |
| CAM-02 | 4,173 | 15 → 12 |

These explain tracker ID-lineage changes at those frames; they do not independently label the person. Numeric IDs differ across runs and must not be crosswalked by number alone.

### A/B/C continuity comparison

The existing runs A (ID correction on) and B (off), and this Re-ID-enabled C run use the same preserved 5,400-frame clips. Using the same per-frame tracker dump and deterministic one-to-one greedy IoU matching for adjacent frames:

| Same-camera box continuity surrogate | A: ID correction on | B: off | C: B + Re-ID |
|---|---:|---:|---:|
| Adjacent-frame matched boxes whose raw ID changes, IoU ≥ 0.4 | 13 | 8 | 11 |
| Of those, IoU ≥ 0.8 | 11 | 6 | 9 |
| Same-frame, same-camera, different-ID overlap candidates, IoU ≥ 0.5 | 8 | 2 | 4 |

These are geometric continuity/overlap proxies, not ID switches, fragmentations, duplicate people, or false merges. C is between A and B on the first proxy and worse than B; it does not demonstrate an identity improvement. The separate `analysis.json` also contains a deliberately simpler per-prior-box (many-to-one) counter yielding 13/9 for C; the table above uses one-to-one assignment consistently across all three runs.

The user’s previous candidate labels remain conservative: GID 7→8, GID 12→13, and GID 8 versus 10 are uncertain; GID 8 versus 11 at frame 3087 was judged likely the same visible subject/duplicate box. In the C run, CAM-05 frame 3087 also has an overlapping different-ID candidate (IDs 8 and 11, IoU 0.615). This remains a duplicate-observation candidate, not a validated reappearance or permission to merge. Raw observations are preserved.

No IDF1/HOTA is reported: there is no frame-level annotated identity ground truth. The existing short sequence at frames 2,234–2,235 is useful for continuity inspection, but is not sufficient to calibrate appearance thresholds. The known approximately ±150 ms source synchronization uncertainty, incomplete pairwise calibration evidence, and unverified map scale remain material limits.

## Persistent identity memory

`replay_identity_memory.py` replayed all 34,622 verified vectors after exact source/frame crosswalk into the existing bounded, fail-closed component. The run had no calibrated threshold profile and no validated world coordinates; automatic matching remained disabled. Results: 0 automatic identity matches, 34,622 observations adjudicated, 34,606 raw identities continued, and the remainder retained as provisional/uncertain. Raw MV3DT IDs were unchanged. This is the intended safe behavior, not a successful persistent-Re-ID outcome.

Do **not** select a similarity threshold from the previously uncalibrated OSNet scores, these NVIDIA example weights, the 8/11 duplicate candidate, or ID-change counts. The next justified step is to annotate multiple continuous positive reappearance sequences and hard negatives (including similar clothing and simultaneous same-camera overlaps), preserve source timestamps and tracker lineage, then fit and evaluate a threshold/margin on disjoint calibration and held-out sets. Validate camera geometry/time alignment before allowing world-position evidence to contribute.

No new three-panel “persistent ID” review was generated: there are no validated persistent-ID assignments to display. The real tracker output is available at `/tmp/entrance-nvdcf-reid-globalid-20261019-run06/run/outVideos/tiled_display_raw.mp4`; it does not contain a map or persistent-person identities.

## Configuration and artifact integrity

| Artifact | SHA-256 |
|---|---|
| Preserved B baseline tracker config | `b175565fc715c004ff3f37687e1cc2548fe4c08345db4feef1ed63090205fa82` |
| Run C tracker config | `de8b282329885c070e374ee765b5a72f483b87c50cc55e4154a1fd25a5f99c05` |
| NVIDIA TAO ReIdentificationNet `.etlt` | `0e5b7f702ce7e3734e45f27b819866f15f6b083048481d1afae2089136e20201` |
| TensorRT engine | `9ca1722b5aadd475d6df6ced189a8b9a4e4358893faae48ff3e71b9c6b7b5b19` |
| `deepstream.log` | `f4bab8e5ce9997ae8c0ad5f76c8a15da8db71f388b81688bc6450659244381a2` |
| Captured MQTT payloads | `575222acd9540f712705c245236a505dab3bdd671439c46ddb08114dfaaf96e2` |
| Derived `analysis.json` | `07049b21f25425f35133f919eda5afd241512c9a3707cdd6690f83204ce676e5` |
| Fail-closed identity replay summary | `d8ba6e434f84bd5fae4ffc53a71a54405ae01a5f7f67af57eb7e8c70f9f42291` |

The input-video hashes recorded by the preserved full replay report are CAM-02 `01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160` and CAM-05 `dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813`. This run consumed the same frame-indexed pair; those source hashes are cited from the prior report, not rehashed from an active dataset mount in this final audit.

The isolated run container/network and test MQTT/Kafka listeners are stopped; no run-specific listener remained on ports 18883/19183. Production services were not targeted. The DeepStream log includes nonfatal V4L2/EOS teardown warnings, followed by tracker deinitialization and `App run successful`; outputs and frame coverage were verified independently.

## RTX 3060 performance

- DeepStream’s per-source cumulative average at EOS: **14.59 FPS per camera** (about 0.486× the 30 FPS input rate); offline processing was not real-time.
- GPU sampler: 192 device-wide samples; utilization mean **90.64%**, p95 **99%**, peak **99%**. Device-wide memory mean **4,098 MiB**, peak **4,153 MiB** of 12,288 MiB. These are device totals, not process-exclusive measurements; startup/shutdown samples are included.
- End-to-end per-frame latency was not measured. Do not infer latency from offline throughput.

Focused regression: `python3 -m pytest -q tests/test_mv3dt_person_identity_memory.py` — **12 passed**. `git diff --check -- experiments/entrance-reid-activation-20261019` — **passed**.

## Next justified correction

No tracker threshold or identity merge is justified now. First obtain manually annotated positive reappearance and hard-negative examples with source-time evidence; record both pre-adoption local IDs and post-MV3DT target IDs/adoption lineage if the supported metadata API permits. Validate geometric calibration and synchronization independently. Then evaluate A/B Re-ID settings against held-out identities and count false merges, fragmentation, continuity, IDF1/HOTA, FPS, and latency. Keep this candidate isolated and the persistent-memory matcher fail-closed until those gates pass.
