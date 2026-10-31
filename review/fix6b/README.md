# Fix 5E + Fix 6B source review snapshot

This is a source-only review snapshot for the isolated DeepStream 9.1 Fix 5E frame-generation ownership and Fix 6B upstream progress diagnostics. It is additive under `review/fix6b/`; it does not replace production sources. `SOURCE_PATH_MANIFEST.tsv` maps each file to its original isolated-source path and SHA256.

## Contents and application

- `native/` contains the modified NVIDIA DeepStream source files at their original SDK-relative paths, plus the CPU native pad harness.
- `integration/isolated-host/` contains the isolated launcher copy. It is deliberately named as isolated and is not the production `services/gpu/runtime/live_launcher.py`.
- `integration/project/` contains the isolated validation wrapper and relevant Python regressions. The branch root already carries shared project dependencies such as `services/gpu/runtime/live_health.py`; that file was identical to the source used by this Fix 6B review when this snapshot was prepared.
- `patches/` retains the exact Fix 5E + Fix 6 native patch, individual patch layers, integration-source patch and original checksum manifest.
- The root repository `LICENSE` is copied here; NVIDIA SPDX copyright/license headers remain in native source files.

This snapshot omits binaries, object files, model/engine data, camera configuration, credentials, RTSP URLs, runtime logs, screenshots, recordings, caches and virtual environments. The candidate executable was verified locally as SHA256 `3cea6475a2474f903400825dba8f6f5f705db2fc93d9f49d7b6a5202f582f3b4`, Build ID `eee53522bf57a62ddba9e94e5783b56ea275accb`, but is intentionally not included.

## Build and test environment

- DeepStream SDK: 9.1
- CUDA: 13.2
- GStreamer: 1.24.2
- Local pinned DeepStream 9.1 Triton development image: `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`
- Validation runtime image is separate and pinned in the existing project runtime configuration: `sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f`
- Offline regression host: Python 3.12.3, pytest 9.1.1, NumPy 2.5.3, PySide6 6.11.2. These are recorded for provenance; the virtual environment itself is not included.

Start from the matching NVIDIA DeepStream 9.1 source tree. Apply `patches/fix5e-plus-fix6-native.patch` to its root using `patch -p1`. The existing deepstream-app Makefile includes the added `six_live_upstream_progress.c` via its C source wildcard. Build only in the pinned development image. For the CPU pad test, compile `native/apps/sample_apps/deepstream-app/six_live_upstream_progress.c` together with `native/tests/fix6_upstream_probe_test.c` and the GStreamer headers/libraries in that image; this does not require cameras or GPU inference.

Offline Python regression command used from the project root:

```sh
.runtime/full-stack-venv/bin/python -B -m pytest -q \
  tests/test_fix6_upstream_diagnostics.py \
  tests/test_fix5_source_generation.py \
  tests/test_native_freshness_regression.py \
  tests/test_six_live_health.py \
  tests/test_six_live_lifecycle.py \
  tests/test_six_live_native_runtime_guard.py \
  tests/test_validate_18m.py \
  tests/test_rtsp_shutdown_pause_optin.py \
  -k 'not real_transient'
```

That run passed 157 tests; three real-transient-service tests were deselected. The native pad harness passed 58 assertions. Neither result verifies actual NVIDIA nvstreammux/PGIE/NvDCF plugin propagation; Fix 5F remains blocked. CAM-01 stall root cause is unconfirmed. No live validation is represented as having passed.
