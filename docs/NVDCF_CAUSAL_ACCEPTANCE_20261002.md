# NvDCF retention and canonical identity evidence, 2026-10-02

Production is unchanged. The experiment remains on
`wip/fix/nvdcf-retention-20261002`, starting at `4cd78c0f`.
The candidate is `config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml`
with the experimental identity manager and capture-reader fixes.

**FINAL STATUS: FAIL for complete metric deployment.** The requested tracker
replay gates and canonical identity gates **PASS**. The independent metric
calibration blocker remains; production promotion is not justified yet.

Absolute metric calibration remains blocked: no measured floor distance exists.
The A→B segment in
`.runtime/fresh-room-calibration-VxTpAvnl/measurement/floor-distance-reference.jpg`
needs a tape/laser measurement recorded with units. Neither 24.22 px/m nor VGGT
scale is a measured metric anchor. All geometry numbers below are existing
projection/policy units, even where legacy code keys end in `_m`. The unchanged
auditor's `maximum_bev_jump_at_most_3m` result does **not** establish a metric
distance. No calibration or detector parameter was changed.

## Reproducible evidence

All new evidence is under `.runtime/nvdcf-causal-20261002/`; older evidence was
preserved. Each replay has a new isolated SQLite gallery, a snapshotted identity
implementation, protected production hashes, input/model manifests, and a
`causal-provenance.json`. No gallery or public ID is reused across replays.

The baseline is the actual recorded V8 tracker, rather than a comment in its
template. The tracked V8 template contains `earlyTerminationAge: 1`; its archived
runner overrides this to **2**, as do both archived V8 replay configs. V8 is
untouched. New configs use the effective value 2. Regression tests parse YAML
values instead of matching a comment.

Canonical inputs are the complete original 2,400 frames/camera, 20 FPS, 120 s:

| Camera | Input SHA256 |
|---|---|
| CAM-01 | `046c7b930a14c850d0e3881440b59ed5427a5dd9829fc6a93d1ea68a07a9cd72` |
| CAM-04 | `6e07b04421730b2d81e64455b5db56312b2a75f5190d549a2f1f36a061555362` |

Person-present input hashes:

| Camera | Input SHA256 |
|---|---|
| CAM-01 | `1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1` |
| CAM-04 | `2b73ecfd4c9361c02442b67966fe78e17aa2352e7b6f39eff0f0258de2ff425f` |

Diagnostic prefixes contain 520 frames. An attempted stream-copy prefix lost a
boundary B-frame and was rejected **before** launch; its evidence remains saved.
The replacement uses lossless H.264 and checks every decoded frame and PTS against
the original. Paired prefix hashes are `95609f8c46df307beb4f12592ce7a16349bf9d7997869f67fa018cb4e6fda8b3`
and `0c0a5c6fffec9fea1eedf4004de79ef36381344e141180dd36c0c0da1853c8f4`.
Prefixes are diagnostic only; final acceptance uses complete original videos.

## Canonical tracker root cause

V8 has 135 proposal-deficit windows: 94 in CAM-01, 41 in CAM-04.
[Every window](../.runtime/nvdcf-causal-20261002/baseline-reviewed-deficit-windows-complete.json)
contains frame counts, all detector confidences, proposal boxes, neighboring native
IDs/confidences, recorded lifecycle samples, overlap evidence, physical review,
and explicit unavailable private fields. A compact
[CSV table](../.runtime/nvdcf-causal-20261002/baseline-deficit-window-table.csv)
is also available. Raw evidence remains in the original V8 replay.

All 370 unmatched canonical proposals were source-crop reviewed. 368 visibly
describe the two real people, including truncated/head/upper-body proposals.
Two CAM-04 source frames, 455 and 1654, have corruption/obscured content and remain
unresolved. Nothing was removed from evaluation. None of these reviews supports
raising the detector threshold to discard the loss.

The 26-frame CAM-04 window 274–299 combines **two** physical losses:

| Physical trajectory | Native before | Recorded loss/recovery | Evidence |
|---|---|---|---|
| Moving black shirt | 9, last published 273 | replacement 16 first published 282 | Detector remains present at confidence about 0.58–0.916; inspected source crops are real people. |
| Seated white shirt | 3, last published 280 | INACTIVE samples 281–299, same 3 recovers 300 | Detector confidence about 0.91; recorded tracker confidence 0.282–0.432 exceeds 0.20. The projected/shadow box is about 36×60 against a detection about 178×250, IoU about 0.048. |

Thus this is not a single person disappearing for all 26 frames, nor detector
intermittency. Association was absent while the old white-shirt target remained
inactive. The SDK does not export private association scores, exact new-target
suppression decisions, shadow age, or termination reason. The small public
projected box is not claimed to be the SDK's private association box.

CAM-01 has short, repeated native breaks and publication delays instead of one
long blackout. Among its 239 unmatched proposals, 56 have matched tentative
samples, 175 inactive samples, and 8 lack a decisive lifecycle sample. CAM-04 has
29 tentative, 92 inactive, and 10 unresolved samples among 131 losses. These are
observed lifecycle classifications, not invented private rejection reasons.

Controlled results identify two mechanisms. Raising new-target overlap allowance
from 0.35 to 0.50 permits an associated replacement white-shirt target during the
old target's inactive interval. Removing the two-frame probation publication cost
then retains detector-supported fragment births. Shadow lifetime stays 162;
tracker confidence stays 0.20; the detector and all final association thresholds
stay unchanged.

## One-factor tracker experiments

| Candidate | Change relative to parent | Scope | CAM-01 / CAM-04 retention | Maximum gaps | Outcome |
|---|---|---|---|---|---|
| V8 control | Effective archived V8 | 520-frame prefix | 91.739% / 94.712% | 9 / 19 | Control |
| V9 | Pose estimator 1→0 | full canonical | 95.178% / 97.367% | 9 / 19 | Rejected |
| V10 | V8 local matching IoU 0.10→0.04 | identical prefix | 92.411% / 95.192% | 9 / 19 | Rejected |
| V11 | V10 size score 0.30→0.04 | identical prefix | 92.219% / 95.192% | 9 / 19 | Rejected |
| V12 | V8 `minIouDiff4NewTarget` 0.35→0.50 | identical prefix | 92.603% / 96.538% | 8 / 5 | CAM-04 suppression mechanism supported |
| V13 | V12 `probationAge` 2→0 | identical prefix | 99.616% / 100% | 1 / 0 | Full acceptance required |
| V13 | Same tracker, original identity decisions | full canonical | 99.711% / 99.937% | 2 / 1 | Tracker passes; four canonical IDs fail |

V12's white-shirt replacement 18 is published and detector-associated from 283
through 299; original 3 returns at 300. This is real published association,
not shadow metadata counted as a normal track.

One V12 sidecar launch failed an import and was preserved as invalid evidence.
The wrapper was corrected and smoke-tested before its valid replay. No invalid
run is presented as an acceptance pass.

## Why V8 created Person_03, Person_04, and Person_05

[The creation timeline](../.runtime/nvdcf-causal-20261002/baseline-complete-evidence/identity-creation-timeline.json)
lists **every** candidate, ReID score, time delta, native/global association
evidence, geometry, predecessor, and exact rejection. The independently reviewed
[physical timeline](../.runtime/nvdcf-causal-20261002/reviewed-identity-timeline.json)
and source overlays establish the following mapping:

| ID | Creation frame/camera/native | Physical person | Relevant previous candidate and rejection |
|---|---|---|---|
| Person_01 | 83 / CAM-04 / 3 | seated white shirt | No existing ID; valid initial creation. |
| Person_02 | 83 / CAM-04 / 9 | moving black shirt | Person_01 cosine 0.432; simultaneous different person, correctly rejected. |
| Person_03 | 193 / CAM-01 / 11 | same white shirt as Person_01 | Person_01 current-camera cosine 0.743, peer cosine 0.664; preceding CAM-01 native 1 at 188, box IoU 0.545. `same_camera_simultaneous` rejects retained memory after a native break, despite no current predecessor assignment. |
| Person_04 | 2101 / CAM-01 / 76 | same black shirt as Person_02 | Person_02 cosine 0.600 / peer 0.557; predecessor 73 at 2097, IoU 0.166; partial-body projected-foot jump about 9.5 units. Same retained conflict rejection. |
| Person_05 | 2105 / CAM-01 / 77 | same black shirt as Person_02/04 | Person_02 cosine 0.651 / peer 0.608, gap 8, IoU 0.499; Person_04 cosine 0.686, gap 2, IoU 0.298. Both rejected as same-camera simultaneous. |

All extra allocations take `positive_novelty_evidence` /
`all_existing_identities_incompatible`. Shared native-MVA evidence is false at
these breaks. The logs do not expose private NVIDIA cross-view rejection scores.
There is no evidence that simply lowering ReID thresholds would solve these
hard-conflict rejections. Source review of the creation windows found no merge of
different physical people.

Tracker fragmentation triggers the reassociation attempts. Global logic then
mistakes retained memory and nested partial/full proposals for a different
simultaneous person. Changed box extent also changes projected feet and defeats
the old motion-distance check. This is a combination of local fragmentation,
global reassociation rejection, and projection sensitivity; no proof requires
replacing YOLO26m or relaxing broad ReID thresholds.

## Narrow identity and runtime fixes

`GlobalIdentityManager` now has opt-in decision context, controlled by
`MV3DT_IDENTITY_DECISION_DETAILS=1` or its constructor argument. It records retained
and current observations, image overlap, source time, geometry, and scoring
evidence. Instrumentation is non-mutating and defaults off; tests compare both
decisions and complete manager state with it enabled/disabled.

The reassociation change recognizes an anchored nested partial/full-body pair
only when all independent support exists:

- A short break within the existing 12-frame window, or a coexisting pair.
- At least two usable crops.
- Both the current-camera and peer-camera gallery scores pass the existing 0.50
  gate.
- At least 90% containment of the smaller image box, a stable top edge, and a
  stable horizontal edge. Top jitter is normalized to the full-body extent.
- A currently assigned peer passes the existing 50 ms and 8.5-unit gates.
- Every other current same-camera target for the candidate must satisfy the
  anchored nested-box/source-time check. A distinct simultaneous person vetoes it.

The existing scorer still chooses the identity. No threshold table changes,
post-hoc merges, aliases, public ID deletions, fixed person count, or future frames
are involved. Tests include actual early and late failing boxes and independently
missing image, time, geometry, gallery, crop, and peer support.

The first identity rule passed the early prefix but failed the full clip at
2099–2103. Source review showed a head/torso proposal followed by a coexisting
fuller box of the same black-shirted person. The final rule covers that measured
case instead of treating every current predecessor as a conflict.

A separate live capture race was then observed: the worker advanced its JSONL
offset through an unfinished line, counted both halves as parse errors, and lost
frame messages. `live_identity_worker.py` now leaves the partial tail unread until
the newline arrives. Tests prove exact-once delivery across writes and preserve
error counting for genuinely malformed complete lines. Final occupied replays
receive all 4,800 messages with zero parse errors.

## Final replay gates

| Gate | Result | Evidence |
|---|---|---|
| Person-present CAM-01 retention | PASS, 99.949362% | `v13-final-person-present/runs/v13-final-person-present/reports/summary.json` |
| Person-present CAM-04 retention | PASS, 99.926671% | Same report |
| Canonical CAM-01 retention | PASS, 99.711111% | `v13-final-canonical/runs/v13-final-canonical/reports/summary.json` |
| Canonical CAM-04 retention | PASS, 99.937317% | Same report |
| Longest gaps CAM-01 / CAM-04 | PASS, canonical 2 / 1; person-present 1 / 2 frames | Exact proposal association audits, no frame filtering |
| Canonical identity set | PASS, exactly Person_01, Person_02 | Final unchanged acceptance audit |
| Canonical switches | PASS, 0 | Manager reassignment/event audit |
| Canonical false merges | PASS, 0 confirmed, 0 unresolved spatial candidates | Unchanged acceptance audit plus source-crop review |
| Person-present false merges | PASS, 0 sampled | Frozen physical-keyframe reference analyzer |
| BEV duplicate/ghost markers | PASS, 0 / 0 in occupied runs | Unchanged production presence reducer audit |
| Empty-room tracks/IDs/markers/shadows | PASS, all zero | Complete 12,031 frames/camera; 24,062 diagnostic shadow files, all empty |
| FPS | PASS, canonical/empty 20 / 20, person-present 20.02 / 20.02 | DeepStream per-source PERF/probe evidence |
| Pipeline/runtime errors | PASS in all final runs | Error scans and zero capture parse errors |
| Production unchanged | PASS | Before/after SHA256 and clean protected-file git diff |
| Absolute metric BEV/calibration | BLOCKED | No physically measured distance supplied |

Final canonical replay:
`v13-final-canonical/runs/v13-final-canonical/replay-20261002-195755`.
Final person-present replay:
`v13-final-person-present/runs/v13-final-person-present/replay-20261002-200135`.
Final empty-room replay:
`v13-final-empty-room/runs/v13-final-empty-room/replay-20261002-200642`.
All three use identical final manager and worker snapshots. Identity creation order is
asynchronous in fresh galleries; the final run labels moving black shirt Person_01
and seated white shirt Person_02. Within-run physical mapping is consistent.

[Final physical trajectories](../.runtime/nvdcf-causal-20261002/final-physical-trajectory-table-r3.json)
include every native ID and its first/last frames. CAM-01 has 44 white-shirt and 66
black-shirt natives. CAM-04 has five white-shirt and 45 black-shirt natives, plus
one corrupt/obscured unassigned sample. All canonical-assigned native tracks have
independently reviewed source crops. Label reuse between runs requires matching
camera/native/frame and ≥0.90 crop IoU; changed numbering was reviewed separately.
These reviews and zero spatial conflicts support the observed merge gate; dense
full-video physical-person ground truth is **not** claimed.

## Risks and promotion

Local native fragmentation remains: exact association retention does not mean
one stable NvDCF ID per person. V13 increases native counts from V8's 66 / 34 to
110 / 51. Current global reassociation handles the tested canonical breaks, but
zero probation may expose detector false positives on other footage. The complete
empty-room replay passes with no detections, published tracks, IDs, markers, or
nonempty diagnostic shadow metadata; it cannot prove behavior on every unseen
false-positive scene.

The five-person person-present dataset still has two sampled identity changes.
V8 also had two, but one trajectory differs: the candidate has a near-left-seated
break at frame 1690; both have a mid-left-white-seated change at 2231. The
person-present retention/false-merge/BEV gates pass; this is not a claim of perfect
global identity on that separate dataset. Those cases remain visible in the frozen
analyzer output and are not relabeled or deleted.

The near-left-seated extra identity is allocated earlier, at CAM-01 frame 442,
native 32. Its Person_05 candidate has current-camera cosine 0.9854, retained
native 1 at frame 441, box IoU 0.6422, and a projected shift of 3.6409 units.
The exact rejection is `same_camera_simultaneous`; the new narrow rule lacks a
peer-camera gallery score for this candidate. This is a separate partial-fragment
case with incomplete peer appearance support, not evidence for lowering broad
ReID thresholds or forcing a merge. The frame-2231 seated-white sampled change
also exists in V8; the frozen reference output remains the evidence and does not
provide dense physical trajectories. Neither case is hidden by a metric change.

Final canonical peak VRAM is 1,836 MiB versus V8's 1,716 MiB (+120 MiB, about 1% of
12 GB card capacity); GPU utilization p95 is 22%, both streams remain 20 FPS.
Person-present peak VRAM is 1,860 MiB versus V8's 1,704 MiB (+156 MiB); utilization
p95 is 24% versus V8's 29%. Empty-room peak VRAM is 1,825 MiB, utilization p95 15%.
No throughput regression or memory pressure was observed. The final canonical
99% utilization peak is transient; sustained p95 is below V8's 23.9%.
No inference engine/detector replacement was needed.

Production promotion is not performed. Absolute metric calibration needs the
measured A→B distance; wider person-present identity behavior also deserves
separate acceptance before claiming the complete deployment objective.

[Consolidated final acceptance and provenance](../.runtime/nvdcf-causal-20261002/final-acceptance.json)
records all replay assertions, the three distinct gallery paths, protected hashes,
code snapshots, and the separate metric blocker. The unchanged canonical audit
with verified final-run physical labels is
`v13-final-canonical/acceptance-reviewed-r3.json`.

119 relevant CPU tests pass, including exact YAML lineage, diagnostics neutrality,
real fragment regressions, negative simultaneous-person cases, split JSONL tails,
prefix boundary corruption, acceptance reducers, and existing identity suites.
The canonical diagnostic script and its tests were already untracked at takeover
and are retained without changing their contents as dependencies of the reports.

Protected production SHA256:

| File under `config/mv3dt_dev_room/` | SHA256 |
|---|---|
| `config_tracker.yml` | `9b7b9bddb96abafdb5fdb80b09ea88d7d8804b5f48cd9ee3f5e8296aae945ac0` |
| `config_deepstream.txt` | `32e89cd1fd2be8ffd0599403b994a804229fe1e2a8732d7f382b52401a580c28` |
| `config_pgie.txt` | `c9cb107445ba2b6c52c8602dd00a7d638f7b35e7706507772995d1b88f837abf` |

## Exact changed files and implementation commits

`64ef4ec33820fab4b8b21c8b5226926fe9fa994e` records experimental tracker
lineage, replay/report tools, and their tests:

```text
config/deepstream/config_tracker_NvDCF_yolo26m_retention_v9_bbox_anchor.yml
config/deepstream/config_tracker_NvDCF_yolo26m_retention_v10_iou004.yml
config/deepstream/config_tracker_NvDCF_yolo26m_retention_v11_size004.yml
config/deepstream/config_tracker_NvDCF_yolo26m_retention_v12_newtarget050.yml
config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml
scripts/dev_room_mv3dt/analyze_nvdcf_causal_experiment.py
scripts/dev_room_mv3dt/diagnose_canonical_retention.py
scripts/dev_room_mv3dt/report_nvdcf_causal_evidence.py
scripts/dev_room_mv3dt/run_nvdcf_causal_experiment.py
tests/test_canonical_retention_diagnostics.py
tests/test_nvdcf_causal_candidate.py
tests/test_nvdcf_causal_evidence.py
tests/test_nvdcf_causal_runner.py
```

The canonical diagnostic script and its test were present untracked at takeover;
their contents were preserved and adopted because the new report imports them.

`8cba149c55525fb045342d426e8fc58bc28ccec4` records identity reassociation
and opt-in diagnostic context:

```text
services/mv3dt_room/global_identity_manager.py
tests/test_identity_decision_details.py
tests/test_identity_partial_fragment.py
```

`39d025d7d8b39355272f8cd426420606f52225bb` records the live-file reader fix:

```text
services/mv3dt_room/live_identity_worker.py
tests/test_identity_kafka_tail.py
```

The remaining changed tracked file is this report,
`docs/NVDCF_CAUSAL_ACCEPTANCE_20261002.md`. Runtime evidence is retained locally
under the ignored `.runtime/nvdcf-causal-20261002/` tree and is not committed.
No production config, V8 template, archived runner, calibration, inference model,
or detector config is included in these changes.
