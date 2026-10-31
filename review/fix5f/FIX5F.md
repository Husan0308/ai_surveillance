# Fix 5F synthetic NVIDIA-plugin fixture

This is a review copy of the guarded Fix 5F fixture. It constructs six generated appsrc inputs and is intended to test token propagation through DeepStream 9.1 nvstreammux, PeopleNet PGIE, NvDCF, and fakesink. It uses no camera, NVR, RTSP, or external network input. A GPU-backed plugin run has **not** been performed as part of this publication task.

The fixture includes the Fix 5E `six_live_monitor.c` implementation and audits token acceptance at pre-mux, mux, PGIE, and tracker stages. It exercises generations 0→1→2 and stale, missing, malformed, duplicate, and wrong-source tokens. It does not prove that a real RTSP decoder/source-bin NULL transition drains in-flight buffers.

## Candidate distinction

The guarded runner is pinned to the original Fix 5E viewer SHA256 `87a22a8ddd380676c37229df5791c53b62557356fd28996598561ddf3879ee35` / Build ID `8fb26e4f5f5c791e0a4259a887f8ba36eb7af3c6`. The newer Fix 6 viewer is SHA256 `3cea6475a2474f903400825dba8f6f5f705db2fc93d9f49d7b6a5202f582f3b4` / Build ID `eee53522bf57a62ddba9e94e5783b56ea275accb`. The Fix 5F runner is deliberately **not** a Fix 6 runner: its candidate path, source hashes, and fixture include are pinned to Fix 5E. A Fix 5E test result must not be treated as proof for Fix 6.

The source-level generation-validation API names used by the fixture are still present in the Fix 6B source snapshot, so both source sets can coexist. The Fix 5F fixture has not been rebuilt or executed against Fix 6B and is not certified as an integration test for that binary.

## Safety contract

`run_fix5f_plugin_test.py` does not launch the GPU pipeline without its explicit latch. Its preflight reads GPU/Docker state and includes a bounded CPU-only `--network none`, `--pull=never` container check of the pinned tracker config. Execution requires the explicit `--operator-authorized-gpu-test` latch, exact candidate and test-artifact hashes/Build IDs, the pinned local DeepStream image, at least 8192 MiB free on GPU 0, and no active GPU compute processes. It rejects pre-existing containers with its Fix 5F owner label. The test uses `--network none`, `--pull=never`, one GPU, a 120-second total limit, and exact run-label/container-ID checks for cleanup. It never stops unrelated containers or GPU processes.

The native fixture and generated ELF test binary are distinct: the C source and Makefile are included; the generated test binary is excluded and must be rebuilt in the pinned SDK environment before any separately authorized test. Model files, TensorRT engines, camera configuration, credentials, logs, screenshots, recordings, and runtime artifacts are not included. The root `LICENSE` and per-file SPDX/copyright notices are retained.
