# V1 cleanup inventory — first static pass

Branch: `refactor/v1-clean-20261013`. Baseline: `32d9ba9d804208d2b214ca80c326b820b07beaea`.

## Scope and evidence

This inventory comes from a static review of the preserved GitHub backup tree, **not** a local host build, an execution trace, or a six-camera runtime qualification.

- 422 tracked files: 181 under `scripts/`, 100 under `services/`, 91 under `tests/`, 14 under `docs/`.
- 77 script paths have names suggestive of candidates, staging, replay, or diagnostics. This **does not prove** they are unused.
- The active F3 supervisor entrypoint, `scripts/full_stack_runtime.py`, sets `SURVEILLANCE_ANALYTICS_ENABLED=0`; it must not be presented as full face or global-identity integration.
- `services/frontend/app/main.py` is a six-camera Qt UI; `services/frontend/sentinel_v1/main.py` is a separate, staged Sentinel UI. Both may have launchers and tests.
- `services/frontend/sentinel_v1/data.py` previously generated fictional Known/Unknown identities, events, and notifications. This branch removes those fabrications, retaining empty collection contracts. Real profiles and events are **not** implemented by that change.
- `services/frontend/sentinel_v1/ui.py` dynamically executes concatenated `ui_parts/part_*.pyfrag` files. Do not remove parts or replace them without test coverage.
- `services/mv3dt_room` contains a scoped CAM-01/CAM-04 identity experiment; whether it is needed by final product must be tested, not guessed.
- No first shared downstream cause of the R13 >3-second physical output stalls has been established.

## KEEP until verified replacement

- `services/camera_v11/native/six_camera_v13/` and native DeepStream 9.1 source/recovery/preview logic.
- YOLO26m config/parser and NvDCF stable tracker config, pinned platform manifest.
- `services/camera_v11/tracked_object_ipc_v1.py`, `ui_preview_ipc_v1.py`, and frontend `camera_wall.py`.
- `scripts/full_stack_runtime.py` and every currently reachable production launcher.
- API/ML service entrypoints and their genuine integration tests.
- Original evidence and user data, including real identity profiles/model assets and recordings.

## REFACTOR / VALIDATE

1. Determine the ONE canonical production launcher and which native binary/sources it builds/starts.
2. Resolve dual frontend strategy only after verifying actual launch usage and retaining the desired Qt UI.
3. Port and verify the five-value `bbox_source` fix already proven offline in another working tree; baseline backup may not include its uncommitted changes.
4. Define real Face Recognition as **NOT_INTEGRATED** unless model + enrollment + persistence + live identity joins can be demonstrated end to end.
5. Treat detector, tracker, and global ReID as separate readiness gates.
6. Check exact Python/native contracts and source identity before pruning historical scripts.

## ARCHIVE CANDIDATES — NOT YET REMOVED

- Old CAM-05 candidate builders and auxiliary inference experiments.
- Historical R13/R14/R15 candidate construction and test harnesses.
- Obsolete recording/calibration experiments whose results have frozen references.
- Superseded UI experiments.

Review actual imports, runtime commands, CI jobs, tests, docs and dynamic imports before deletion. Git history is the archive of record; do not delete live runtime data.

## Already changed

- `services/frontend/sentinel_v1/data.py`: remove simulated people/events/notifications while keeping empty public lists for consumers.

## PASS definition

A static cleanup commit is never a runtime PASS. For each deletion batch:
1. Compare baseline and changed tree, inspect import and runtime references.
2. Compile and run focused regression suites in an isolated worktree.
3. Run full practical unit suite and relevant native compile/preflight without physical cameras.
4. Fix legitimate regressions without disabling/removing meaningful tests.
5. Record exact commands, failure list, hashes, PASS/FAIL/REVIEW_REQUIRED.
6. Reserve real-camera GPU-latency, reconnect, six-camera recovery, tracker/ID continuity and face recognition PASS for separately authorized physical/hardware validation.

## Known blockers to close sequentially

- Tracking canonical IPC five-source contract.
- Clean six-camera native build and unchanged detector/track behavior.
- R13 source-recovery shared output stall (first causal stage unknown).
- Qt preview latency and matched historical sub-40-ms measurement definition.
- Cross-camera ReID true live readiness (not merely CAM-01/CAM-04 experiment).
- Face enrollment / recognition / Known-Unknown persistence (not fabricated demo profiles).
