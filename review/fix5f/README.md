# Fix 5F source review snapshot

Source-only export of the existing guarded synthetic plugin fixture, its safety tests, and the precise Fix 5E source inputs needed to review/rebuild it. It is additive under `review/fix5f/`; it does not alter production files or the existing `review/fix6b/` snapshot. `SOURCE_PATH_MANIFEST.tsv` maps exported files to their original project or isolated SDK-source paths and records each source SHA256.

No candidate executable, generated test ELF, model, TensorRT engine, camera configuration, RTSP URL, log, image, video, virtualenv, or cache is included. Artifact hashes and Build IDs are recorded separately in `ARTIFACT_IDENTITIES.md`. The tested local development image was DeepStream 9.1 / CUDA 13.2 / GStreamer 1.24.2, pinned at `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`.

For review, begin with `FIX5F.md`, then inspect the guarded runner, CPU safety tests, native fixture and Makefile. The runner refers to local candidate/model paths that are intentionally not exported. The separate Fix 6B snapshot retains its own native source, integration launcher, wrapper and checksums.
