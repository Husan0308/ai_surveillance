# Entrance Room — Persistent Person Identity Review

**Verdict: PARTIAL**

The tracker configuration does not use appearance Re-ID, and the preserved 180-second event stream contains no appearance embeddings. The isolated memory component and its conservative assignment rules are implemented and source-free tested, but it cannot verify a single same-person disappearance/reappearance match from the available evidence. No inferred identities were merged and no review UI was opened or changed.

## Findings

| Check | Result | Evidence |
|---|---|---|
| MV3DT appearance Re-ID active | **No** | The preserved B tracker config has no `ReID:` section or model path; `minMatchingScore4ReidSimilarity: 0` and `matchingScoreWeight4ReidSimilarity: 0`. `reidType`, `reidHistorySize`, and `outputReidTensor` are absent, so NVIDIA's documented default `reidType=0` (DUMMY) applies; the documented default history size is 100 but is unused without Re-ID. |
| Tracker memory after loss | **Short-lived only** | `maxShadowTrackingAge: 162` frames (about 5.4 s at 30 FPS); `TrajectoryManagement.maxTrackletMatchingTimeSearchRange: 27` frames (about 0.9 s); `MultiViewAssociator.maxTrackletMatchingTimeSearchRange: 1`; `recentlyActiveAge: 178` frames (about 5.93 s). `enableReAssoc=1` and `enableLatePeerReAssoc=1`; B has `enableIDCorrection=0` (A has it enabled). These settings do not establish long-gap persistent identity after both views lose a person. |
| Embeddings in replay | **Unavailable** | Read-only scan of all 34,539 B observations: `embedding` was an empty object in all 34,539; non-empty vectors: 0. The raw event stream also has no source PTS or local tracker lineage. |
| Available OSNet | **Separate model, not wired into MV3DT** | A local `osnet_ain_x1_0_msmt17.pth` exists in the separate AI Surveillance experiment tree (SHA-256 `8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f`). It loaded for a small CPU crop-similarity probe; it is not the active DeepStream tracker model and has not been accuracy-calibrated for these recordings. |
| Same-person examples | **Not confirmed** | Sampled windows show candidate track returns with broadly similar light tops/dark trousers, but not unique visual evidence. Camera timing is only a nominal replay timeline; source-clock alignment uncertainty remains about ±150 ms. The room geometry/map scale is not independently validated. |

The preserved B event database is `/tmp/entrance-180s-review.9WdL9e/B-events.sqlite3`, read in SQLite read-only mode. Its metadata binds it to source log SHA-256 `3046c56da870b14084eff0ee8fc9981986fd0aef657d6894ecf3635a579a61f7`; B tracker-config SHA-256 is `b175565fc715c004ff3f37687e1cc2548fe4c08345db4feef1ed63090205fa82`.

## Candidate disappearance/reappearance windows

These are *candidates for annotation*, not visually confirmed person identities. The temporary IDs below are separate provisional memory labels, not names or claims of unique real people. Memory was run on the full replay without embeddings and therefore left every raw global ID separate.

| Camera / candidate | Raw MV3DT ID and interval | Provisional memory ID | Assessment |
|---|---|---|---|
| CAM-05, nominal 30 FPS | Raw `7`: frames 1,969–2,521 (65.63–84.03 s); raw `8`: frames 2,633–3,088 (87.77–102.93 s). There are 111 unobserved frame indices; endpoint delta is 112 frames, about 3.73 s nominal. | `7` → `person-provisional-00008`; `8` → `person-provisional-00009` | **Uncertain.** Sampled OSNet similarity was 0.626 between CAM-05 crops (0.748 for a cross-camera crop comparison); similar clothing and no distinctive corroborating feature. No merge. |
| CAM-05, nominal 30 FPS | Raw `12`: frames 3,652–3,704 (121.73–123.47 s); raw `13`: frames 3,787–4,582 (126.23–152.73 s). Gap: 82 frames, about 2.73 s. | `12` → `person-provisional-00013`; `13` → `person-provisional-00014` | **Uncertain.** Sampled OSNet similarity was about 0.673; no manually confirmed identity label or independent movement/timing evidence. No merge. |
| CAM-05, competing candidate | Raw `7` ends at frame 2,521; raw `10` appears at 2,788–2,843 (92.93–94.77 s), an 8.90 s interval from the end of `7`. | `7` → `person-provisional-00008`; `10` → `person-provisional-00011` | **Do not merge.** The OSNet crop score from `7` to `10` was 0.792, but raw `8` is concurrently present in CAM-05 over frames 2,788–3,088. Raw `10` also has a competing appearance score to another fragment. The score is exploratory and uncalibrated. |

Same-camera interval overlap is an explicit conflict signal for the memory component. For example, raw IDs `8` and `10` are both recorded in CAM-05 over frames 2,788–2,843; raw IDs `8` and `11` overlap over frames 2,941–3,088. These may be distinct people or tracker duplicates; the event stream alone cannot distinguish them, so they remain separate.

The only preserved `ADOPTED_ID` alias manifest comes from a different diagnostic run and has a different source-log hash. It was not applied to B. An adoption event can explain a tracker label transition, but is not independent proof that two observations depict the same real person.

## Isolated implementation and validation

Added `services/mv3dt_staging/person_identity_memory.py`, a test-only, in-memory component. It keeps raw IDs unchanged and separately returns `persistent_person_id`, `display_id`, match status, and candidate similarity. It bounds gallery size and retention; requires high-quality embeddings and an explicitly identified calibrated threshold profile before accepting appearance matches; by default it also requires validated location evidence (that guard is configurable but was not relaxed here); rejects active identity conflicts; keeps uncertain observations provisional; and treats hash-bound `ADOPTED_ID` records as display aliases only. It stores no image crops or embeddings on disk and is not wired into production tracking or the existing review UI.

The implementation also handles a late first usable embedding: a raw track first seen without a feature can be compared once its first high-quality feature arrives. Regression coverage verifies this path as well as raw-ID isolation, confident versus ambiguous synthetic matches, simultaneous-track conflict rejection, low-quality embeddings, movement/location gating, expiry, camera scoping, adoption hash/cycle checks, and ordering.

Full B replay through this component:

- Input observations: 34,539; raw global IDs: 16.
- Embeddings present: 0; automatic appearance matching enabled: false.
- Result: each raw global ID received its own provisional memory ID; accepted re-identifications: 0; raw events unchanged.
- This is a fail-closed behavior check, not an identity-accuracy result.

Focused tests: **30 passed**, with one Starlette/httpx deprecation warning. `git diff --check` passed. The plain system Python could not collect the FastAPI UI tests because FastAPI is not installed there; the repository's `.runtime/full-stack-venv/bin/python` ran the intended suite successfully. No GPU inference, calibration, camera access, tracker change, production-service change, or review-UI launch occurred.

## NVIDIA behavior and next safe step

NVIDIA documents tracker Re-ID modes (`reidType` 0–3), a per-target Re-ID feature gallery, the `reidHistorySize` setting, and feature export through `outputReidTensor: 1` plus `reid-track-output-dir` when a supported Re-ID mode is active. NVIDIA also documents that a target lost beyond `maxShadowTrackingAge` can receive a new tracker ID even with Re-ID, so the configured short tracker lifecycle is not a guarantee of indefinite identity. MV3DT `ADOPTED_ID` records describe peer ID adoption and should be retained as tracker lineage, not treated as independent identity ground truth.

The inspected visual samples are `/tmp/entrance-persist-id/b-candidate-gaps.png` and `/tmp/entrance-persist-id/osnet-candidate-crops.png`; they support appearance ambiguity, not same-person ground truth.

The smallest useful next step is a new, isolated offline instrumentation run that uses a DeepStream-9.1-compatible Re-ID model/configuration, exports the actual per-frame embeddings, and is paired with manually annotated same-person and hard-negative intervals (including similar clothing and overlapping tracks). Fit and validate threshold/margin profiles on separate examples before allowing any persistent-person merges. Keep source PTS/synchronization and world-coordinate evidence as explicit inputs; current map geometry and source-clock alignment are not strong enough to serve as validated match gates. Until that evidence exists, display candidates as provisional and keep raw IDs separate.

Official NVIDIA references:

- [DeepStream 9.1 Multi-View 3D Tracking](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html) — peer tracklets, common-time-axis assumption, `ADOPTED_ID`, Late Peer Re-Association, and ID Correction.
- [Gst-nvtracker documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) — Re-ID modes, feature gallery, shadow-tracking lifecycle, and feature output requirements.
