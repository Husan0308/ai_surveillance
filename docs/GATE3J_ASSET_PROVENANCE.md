# Gate 3J source and deployment disposition

## Branches are different validation lineages

Continue on `rebuild/clean-production-v1-20260920` from
`8a4769b8f9f2ddbc70f449b8dbd23bc0b511ee54`. The clean continuation worktree is
`.runtime/gate3jr-DV3lTHm8/recovery` inside `ai_surveillance-gate3j`.

The original `ai_surveillance` WIP checkout and the
`wip/gate3j-readiness-barrier-20260930` checkout remain untouched.
Published history and the freeze branch are not rewritten.

```text
bfe3055 ── 8a4769b                 validated recovery + publication barrier
       └─ 6a375c2 ── 2536256 ── ab48a9f   preserved alternative watchdog work
```

`2536256` adds fresh-stage recovery criteria, retry scheduling, parent-state
checks, and GLib-dispatched source reset; `ab48a9f` tests those changes.
These are not the final candidate's compile inputs. The saved C1 experiment
used that source (`ec9f86...`) and binary `214f9f...`, briefly recovered, then
stalled again and required owner restart. The branch is preserved, not reset.
Do not replace the proven `8a4769b` source with it simply to reconcile hashes.

`8a4769b` retains the independent preview prerequisite, the publication-only
WARMING/READY barrier, source-local frame-number continuity before tracker/MVA,
and suppression of transient EOS only at reconnectable live analytics inputs.
It preserves URI replay EOS, flush/segment events, frame buffers, queue policy,
and all inference/tracking/identity thresholds. Its final candidate passed the
ten sustained-recovery cycles plus independent confirmation, real RTSP recovery,
canonical replay, and six-camera latency. Some recoveries took approximately
21 seconds; this is not a claim of uniformly one-second recovery.

## Exact artifacts and provenance

| Role | Binary SHA256 | ELF build ID |
| --- | --- | --- |
| This branch's historical accepted binary | `555541b73468147743ae7cdfc7f48df9e365eb22f08fbab3ade8124820ee3c7d` | `254071a721988d3440ec5a90ba2c80bece0cee5f` |
| Installed shared-path WIP binary, not accepted by this branch | `d7dc4c88fba7d652d5d590fa2425043efacb8af45f6a8db7f4f4d1656fe98398` | `c385e972aa7c1f51feecffd4699ad6b1e1a1330d` |
| Final Gate 3J-R staging candidate | `6e9604d1aa31e2473303edcc2ef05f3e7812989361b8aad98de703a56a3c8817` | `666f6452dac25d37bb452cc64842bc749484fa5b` |

The `555541...` file was found, intact, beside the shared installed binary as
`deepstream-test5-pn263-global-id-proto.pre-stale-fragment-promotion-20260928T085843Z`.
Its recorded mtime is September 24. The original checkpoint `f5a1731` and
saved September 26 replay metadata identify it as the historical accepted
binary. No matching original full build-input inventory was located, so its
exact native compilation snapshot is not asserted. `74d37cf` intentionally
rebaselined diagnostic source to `99399c...` without rebuilding that binary.

The shared installed `d7dc...` file is supported by the preserved
`fragmentation-fix-candidate-final-20260928/build.json`, production-candidate
metadata, WIP promotion commit `1dae013`, and the September 29 rollback
snapshot. Its native source was `0762ee5...`; its identity manager was
`2f2047d...`. That WIP deployment also accepted different PGIE/tracker hashes.
It cannot be accepted by copying only its binary hash into this branch.
Its timestamp reflects the later rollback copy, not an unexplained new build.

The final `6e960...` candidate was rebuilt twice from the exact native source
at `8a4769b`:
`6b7470245be42193bbe58f59476fd20422d02278860f681681ee661c398318cc`.
Both rebuilds are byte-identical, including ELF build ID. The reproducible
toolchain is DeepStream 9.1 image
`nvcr.io/nvidia/deepstream@sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`,
GCC/G++ 13.3.0, and GStreamer 1.24.2. The SDK sample `deepstream_utc.c` input
hashes to `ab1f1096...`; the different vendor copy is not used. The new build
wrapper records all native/SDK source/header hashes, inner build script,
compiler versions, image ID, revision/dirty status, and output hash. It refuses
an existing output directory, any output outside `.runtime`, an unpinned image,
changed inputs during build, or a binary differing from the validated pin.

Preserve `.runtime/gate3jr-DV3lTHm8` and
`.runtime/gate3j-consistency-2gZYkIvS`. Final recovery, RTSP, latency, and replay
evidence identifies `6e960...`, not earlier `214f9f...`, `ce0e75...`, or
`13fac7...` candidates.

## Fail-closed production versus explicit staging

The production runtime reference now selects the recovered, matching
historical `555541...` backup instead of the unrelated shared-path `d7dc...`
file. Neither binary was overwritten. Accepted manifest hashes remain unchanged.

Bare validation checks the runtime-selected binary even without `--binary`.
At this stage it is **expected to fail only the unaccepted native-source hash**:
the checkout contains the Gate 3J fix, while the accepted manifest still names
`99399c...`. Do not roll back the fix or declare bare production acceptance.

Staging is explicit and does not imply production acceptance:

```bash
python3 scripts/dev_room_mv3dt/build_staging_candidate.py \
  --output .runtime/new-gate3j-build

python3 scripts/dev_room_mv3dt/verify_validated_assets.py \
  --staging-profile config/mv3dt_dev_room/gate3j_candidate.json

python3 scripts/dev_room_mv3dt/run_room_pair.py \
  --staging-profile config/mv3dt_dev_room/gate3j_candidate.json \
  --mode replay --skip-render
```

The staging profile expects the candidate in its hash-addressed `.runtime`
archive. A newly reproduced binary may be copied there only after verifying
its exact hash; never overwrite an existing artifact. Every run records
`execution_assets.json` with actual/expected binary hashes, all checked asset
hashes, role, and `production_accepted=false` for staging.

`MV3DT_FRAME_AUDIT_LOG` and source-health diagnostics never waive hashes.
The alternative diagnostic environment requires all three of `MV3DT_BINARY`,
`MV3DT_DIAGNOSTIC_SOURCE_SHA256`, and `MV3DT_DIAGNOSTIC_BINARY_SHA256` together.
Only the explicitly pinned native main source is staged. Identity, bbox,
calibration invariants, and accepted PGIE/tracker configs remain checked.
Identity bypass, partial pins, wildcard hashes, and mixed staging selectors
are rejected before any camera starts.

## Backend and ownership

API/ML/frontend defaults use 8100/8101. Calibration on 8000 is separate and
untouched. Private `.env` files still contain obsolete explicit port values;
they are not edited or committed. Managed API/ML startup must select canonical
ports (and the API's ML URL) explicitly if using those private files.
Frontend defaults need no manual URL override.

`preview_only_runtime` defaults to CAM-02/03/05/06 and rejects room-pair or
duplicate IDs before loading credentials/starting captures. CAM-01/04 remain
owned only by MV3DT. This changes ownership validation, not decoder/queue/FPS
settings. The old Sentinel architecture test referenced a TRT8.6 experiment
deleted by `9bfb805`; it now verifies current ownership and the unchanged
telemetry-only UI client instead of restoring that obsolete runtime.

The ML health response requires both `ready=true` and `status=ready`; a
contradictory status cannot advertise healthy publication.

## Non-human revalidation on September 30

The final source/recovery/publication implementation is unchanged from
`8a4769b`. Revalidation used the exact `6e960...` binary, not a newer native
experiment. Two independent builds reproduced it byte-for-byte. The full
repository unit/integration suite passed **213 tests**, and all **5 native bbox
history/lifecycle scenarios** passed; syntax and whitespace checks passed.
The stale-preview UI test now uses an isolated IPC fixture instead of reading
an unrelated live producer's `/dev/shm` payload.

Three fresh startup runs reached READY with advancing mux/PGIE/tracker counters.
Two new seven-second freezer-pause cycles closed publication, discarded stale
presence, and recovered without an owner restart or second stall, remaining
READY for at least 65 seconds. Recovery after unpause took 1.02 and 1.12
seconds. These corroborate, rather than replace, the preserved ten-cycle
Gate 3J-R campaign. A separate real TCP interruption reset only the two
verified native-owner RTSP sockets and also sustained advancing analytics
for 65 seconds; it recovered within the health timeout, so it did not falsely
claim a prolonged NOT READY transition. The four preview-only sockets were
not interrupted.

The fresh real PySide6 wall was restored from its minimized state and held
visible for 330 seconds. API and ML were healthy on 8100/8101, the wall was 6/6,
and Dev Room was READY. All timing rows were retained in the normal latency
auditor:

| Camera | Valid samples | P50 ms | P95 ms | P99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAM-01 | 6050 | 22.14 | 34.80 | 41.32 | 51.81 |
| CAM-02 | 6197 | 21.66 | 34.81 | 41.15 | 57.12 |
| CAM-03 | 5952 | 21.61 | 34.29 | 40.71 | 65.36 |
| CAM-04 | 6012 | 17.63 | 30.64 | 36.38 | 55.46 |
| CAM-05 | 6309 | 16.07 | 29.31 | 34.91 | 50.82 |
| CAM-06 | 5906 | 16.13 | 29.60 | 36.18 | 56.12 |

The metric remains decoder-reference to actual UI paint, not sensor-to-screen.
Decoder-out to publish P95 was 2.23 ms (CAM-01) and 2.27 ms (CAM-04).
Sources/analytics averaged 19.87–20.02 FPS. Reconnects, malformed/invalid rows,
PTS misses/backsteps, and non-monotonic UI timestamps were zero. Queue limits
remain 2/16, non-leaky; sampled maximum occupancy was 1 for each source. There
was one CAM-01 queue-full notification, no CAM-04 overrun, no growing backlog,
and no intentional READY-state analytics dropping.

The earlier minimized/obscured runs are preserved too. One measured CAM-01/02
P95 at 40.55/40.71 ms while the wall was hidden; it is not erased or represented
as acceptance. Restoring the actual window, with no refresh/preview parameter
change, preceded the fresh complete measurement above. This does not establish
visual smoothness: latest-frame painted FPS was 17.84–19.06 with uneven arrival
and some >150 ms gaps. Cadence remains a separate known issue, not a newly
declared smooth-preview PASS.

Fresh isolated-gallery canonical replay acceptance produced only Person_01
and Person_02 in both cameras, with no switches, unnecessary IDs, confirmed
physical merges, duplicate/ghost markers, or presence disagreements. The
Person_06 regression still rejects stale pooled similarity approximately
0.7056 and supports genuine reacquisition approximately 0.9288. Recorded replay
is not proof of live-person detector recall.

## Source-controlled change inventory

Only these files changed in this continuation:

- `.env.example`
- `config/mv3dt_dev_room/runtime.yaml`
- `config/mv3dt_dev_room/gate3j_candidate.json`
- `docs/GATE3J_ASSET_PROVENANCE.md`
- `scripts/dev_room_mv3dt/build_staging_candidate.py`
- `scripts/dev_room_mv3dt/run_room_pair.py`
- `scripts/dev_room_mv3dt/run_startup_gate.py`
- `scripts/dev_room_mv3dt/verify_validated_assets.py`
- `services/runtime_ports.py`
- `services/api_service/app/config.py`
- `services/ml_service/app/main.py`
- `services/frontend/app/config.py`
- `services/frontend/sentinel_v1/ui.py`
- `services/camera_v11/preview_only_runtime.py`
- `tests/test_acceptance_ui.py`
- `tests/test_candidate_build_provenance.py`
- `tests/test_monitoring_services_v1.py`
- `tests/test_preview_only_ownership.py`
- `tests/test_production_ui_source_mode.py`
- `tests/test_room_pair_asset_guards.py`
- `tests/test_sentinel_monitoring_realtime_v1.py`
- `tests/test_service_port_defaults.py`

Native recovery/publication/identity algorithms and the accepted manifest are
unchanged. No runtime evidence, generated asset, secret, or WIP cosmetic edit
is included in these commits.

## Remaining acceptance

All non-human readiness/recovery/identity/preview checks can be completed with
the staged candidate. Formal live PeopleNet person recall remains **PENDING /
NOT RUN** until the scheduled real-person session, October 1 at 11:00 local
time. Empty-room previews and recorded replay do not satisfy it. No detector
threshold, model, tracker/MVA/calibration, or identity decision is changed.

The first real-person run may validate the staged candidate, but it must not
be mislabeled as accepted-production recall. After that candidate gate and
required promotion approval, acceptance must
back up the current production artifact and manifest, atomically install the
validated binary, rebaseline only deliberately accepted native/binary entries,
and run bare and post-promotion gates, including fresh person-present production
recall without staging overrides. Until then final freeze/tag is blocked;
no release tag is created and the recovery freeze branch remains unchanged.

Reference constraints: [NVIDIA MV3DT cross-stream alignment](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html)
and [GStreamer state transitions](https://gstreamer.freedesktop.org/documentation/additional/design/states.html).
The validated fix preserves the existing SDK source-reset architecture rather
than introducing another reader or changing paired tracker settings.
