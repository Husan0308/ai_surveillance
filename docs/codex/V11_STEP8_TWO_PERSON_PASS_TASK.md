# V11 Step8 Two-Person Identity Acceptance — Codex Task

## Goal
Make the real CAM-01/CAM-04 two-person Step8 acceptance pass for the correct architectural reasons. Do not weaken the checker, lower assertions, hide conflicts, or change expected counts to manufacture PASS.

Work on the existing branch and repository state. Inspect the code first, reproduce all deterministic/unit failures you can locally, patch conservatively, and keep iterating until the implementation is technically ready for the live camera acceptance. When running on the actual camera machine, continue through the live test and use its logs to iterate until the real Step8 result is PASS.

## Current live failure
The latest real run reached all phases but ended:

```text
V11_STEP5_GLOBAL_SHADOW_V1 RESULT=PASS created=2 provisional=0 confirmed=1 observations=47 conflicts=1 expired=1 active=1 member_tracks=2 track_successor_reuse=1 current_member_per_camera=1
V11_STEP6_GLOBAL_VERIFY_V1 RESULT=PASS records_created=1 pending=0 verified=1 hold=0 expired=0 verified_total=1 hold_events=1 recovered_total=1 persistent_conflicts=0
V11_STEP8_CAM01_CAM04_TWO_PERSON_V1 RESULT=FAIL reasons=step5_confirmed=1/expected2;step5_conflicts=1/expected0;step5_expired=1/expected0;step5_active=1/expected2;step5_member_tracks=2/expected4;step6_records_created=1/expected2;step6_verified=1/expected2;step6_verified_total=1/expected2;step6_hold_events=1/expected0;step6_recovered_total=1/expected0;global_conflict_rows=1/expected0;global_expire_rows=1/expected0;verify_hold_rows=1/expected0;verify_recover_rows=1/expected0;phase_a_verified_ids=0/expected1;phase_b_new_confirmed_ids=0/expected1;phase_c_new_global_ids=GSH-000001,GSH-000002;global_confirmed_ids=1/expected2;global_verified_ids=1/expected2
```

The live detector/tracker/ReID performance itself is healthy enough:

```text
Step1 PASS, six cameras smooth
Step2 PASS, min avg detect rate ~=1.95 Hz
Step3 PASS, tracker_p95 ~=0.532 ms
Step4 ReID quality PASS
Step4 gallery PASS, infer_p95 ~=18.1 ms
Step4 pair scorer PASS
Step4 same-room matcher PASS
```

So do not "fix" this by touching camera ingest, detector scheduling, or GPU policy unless hard evidence proves they cause the identity failure.

## Existing architecture and important files
Inspect at minimum:

- `services/camera_v11/step4_camera_tracklet_v1.py`
- `services/camera_v11/step4_reid_same_room_matcher_v1.py`
- `services/camera_v11/step5_global_shadow_v1.py`
- `services/camera_v11/step5_global_shadow_worker_v1.py`
- `services/camera_v11/step6_global_shadow_hysteresis_v1.py`
- `services/camera_v11/step6_global_shadow_runtime_v1.py`
- `services/camera_v11/step8_two_person_debug_runtime_v1.py`
- `scripts/test_camera_v11_step4_camera_tracklet_v1.py`
- `scripts/test_camera_v11_step5_global_shadow_v1.py`
- `scripts/test_camera_v11_step6_global_shadow_hysteresis_v1.py`
- `scripts/check_camera_v11_step8_cam01_cam04_two_person_v1.py`
- `scripts/run_camera_v11_step8_cam01_cam04_two_person_acceptance_v1.sh`

Also inspect TSV event ordering for the failing run when available:

- `step4_pair_scores_v1.tsv`
- `step4_same_room_matches_v1.tsv`
- `step5_global_shadow_v1.tsv`
- `step6_global_verify_v1.tsv`
- `step8_phase_markers.tsv`

## Frozen boundaries
Do not regress or redesign Step1-3. The guard must stay PASS:

```text
V11_FROZEN_STEP123_GUARD RESULT=PASS sha=d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51 files=19
```

Keep these constraints:

- no new frame FIFO/backlog
- no extra detector inference
- no extra ReID inference for the debug preview
- no new RTSP streams for debug UI
- no latency regression in the production camera wall
- Step8 debug overlay is read-only and must not mutate tracker/ReID/global identity state

## Required identity semantics
A physical person is not an immutable local tracker-ID pair.

For two people A and B in overlapping CAM-01/CAM-04 views:

- A must own exactly one global shadow ID.
- B must own exactly one different global shadow ID.
- Local T-ID or stable CT-ID succession may happen without creating a new global person.
- A successor track may attach to an existing global identity only when evidence anchors it safely to that same physical person.
- Never merge two already-owned different global identities just because a new pair proposal connects them.
- Simultaneous same-camera tracks that represent two visible people must not be treated as successor aliases of one identity.
- After crossing/occlusion, the system must not swap A and B global identities.
- Historical aliases may remain as history, but there must be at most one current member per camera per global identity.

Use reciprocal/one-to-one association and temporal constraints where appropriate. If the current Step4 mechanics-only matcher produces ambiguous proposals, resolve ambiguity using lifecycle-safe logic and measured evidence; do not just reduce thresholds.

Reference conceptually relevant NVIDIA behavior: target re-association links newly appeared targets to recently lost trajectories using motion/ReID evidence; peer re-association and ID correction exist specifically because wrong associations can occur with close/similar targets. Do not copy DeepStream blindly; preserve this project's lightweight V11 architecture.

## Suspected failure classes to investigate
Do not assume one cause. Prove the exact cause from event timelines and tests.

Check for:

1. Person B never reaching enough consecutive/clean evidence before a provisional expiry.
2. A/B pair proposals temporarily crossing during occlusion and generating a conflict that suppresses one identity.
3. Same-camera successor reuse incorrectly attaching Person B to Person A after a short gap.
4. An expired provisional GSH preventing later re-use/confirmation of the correct B hypothesis.
5. Step6 HOLD/RECOVER behavior reacting to a transient Step5 conflict that should never have been emitted.
6. Phase-A verification lag: A is confirmed but not yet Step6 verified before Phase A ends.
7. Checker phase attribution using only event creation/pass rows in a way that misses already-existing verified identities. If checker logic is semantically wrong, fix it only if you can prove the checker is misreading correct state. Do not relax expected identity invariants.
8. Current-member replacement logic losing one camera member when one local/stable track succeeds another.

## Improve observability before guessing
The Step8 preview already shows bbox plus local T-ID / CT-ID / GSH / verification state. Preserve it.

Add cheap diagnostic logging if needed so every Step5 proposal decision explains one reason such as:

```text
create_new
refresh_existing
successor_attach
same_camera_overlap_reject
two_owner_conflict
historical_alias_reject
provisional_expire
confirm
```

For any conflict/expiry, log the involved current owners, camera members, cycle, and last-seen cycle. Diagnostics must be bounded and non-blocking.

## Test requirements
Add deterministic unit/regression tests before relying on another live run.

At minimum cover:

1. Two independent people create two independent globals and both confirm.
2. A local successor for Person A reuses A's global ID.
3. A local successor for Person B reuses B's global ID.
4. A and B visible simultaneously in one camera cannot be successor-merged.
5. Crossing proposal noise between already-owned A/B identities never merges them.
6. A transient wrong cross-pair does not destroy/expire the two good confirmed identities.
7. An unconfirmed provisional identity can expire without expiring or stealing a confirmed identity.
8. Step6 verifies both identities after enough clean observations.
9. Step6 does not HOLD a global because of a proposal that Step5 should classify as harmless ambiguity/non-action.
10. After local-ID changes on both cameras, each global still has exactly one current member per camera.
11. Hidden ID swap after crossing is caught by the Step8 phase checker.
12. Phase checker correctly recognizes an identity that was verified before the isolation window and then only observed during that window, if that is the intended state semantics.

Run all relevant tests repeatedly until stable.

## Live Step8 protocol
The real acceptance runner is:

```bash
V11_STEP8_TWO_PERSON_ACK=1 bash scripts/run_camera_v11_step8_cam01_cam04_two_person_acceptance_v1.sh
```

The live phases are ground truth:

- Phase A: only Person A visible in both CAM-01/CAM-04.
- Phase B: A stays; B joins; both separated and each visible in both cameras.
- Phase C: both cross, briefly occlude, turn, swap positions.
- Phase D: only A remains in both cameras.
- Phase E: only B remains/returns alone in both cameras.

Do not shorten phases or lower confirmation/verification requirements merely to get PASS. It is acceptable to make the runner interactive (explicit operator acknowledgement between phases) if that improves ground-truth reliability.

## Final acceptance invariants
A legitimate PASS must end with these identity semantics, or a strictly stronger equivalent:

```text
Step5:
created=2
provisional=0
confirmed=2
conflicts=0
expired=0
active=2
member_tracks=4
queue_dropped=0
worker_errors=0

Step6:
records_created=2
pending=0
verified=2
hold=0
expired=0
verified_total=2
hold_events=0
persistent_conflicts=0
verify_worker_errors=0

Step8:
physical_people_expected=2
confirmed_ids=2
verified_ids=2
wrong_merge=0
id_swap=0
cross_person_successor=0
conflicts=0
holds=0
expiries=0
current_members=4
RESULT=PASS
```

## Forbidden shortcuts
Do not:

- change expected 2 people to 1
- allow conflicts/holds/expiries in the final acceptance just to pass
- remove conflict assertions
- lower ReID similarity/margin thresholds without measured evidence and new regression tests
- increase expiry timeouts as the primary fix without proving lifecycle timing is the root cause
- hardcode `GSH-000001`/`GSH-000002`
- use clothing labels or manual identity assignments in production logic
- add face recognition for this Step8 fix
- add geometry/world calibration for this Step8 fix unless the current failure is proven impossible to solve safely without it
- mutate frozen Step1-3
- claim PASS from unit tests only

## Working style
Do not stop after analysis. Make the smallest correct patch, run tests, inspect failures, patch again, and continue.

Before finishing:

1. show the root cause supported by exact code paths/log evidence;
2. list changed files;
3. show unit/regression test outputs;
4. show Step1-3 frozen guard still PASS;
5. if running on the real camera machine, show the final real Step8 PASS line;
6. if the Codex environment cannot access the physical cameras/operator, do not claim live PASS — leave the branch in a state ready for the one-command live test and state exactly what remains external.

Do not create a new architecture unrelated to this failure. Prefer a conservative state-machine/lifecycle fix with explicit invariants and tests.
