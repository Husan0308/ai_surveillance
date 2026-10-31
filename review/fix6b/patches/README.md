# Fix 6B source review bundle

This bundle contains the source-only native Fix 5E plus Fix 6 patch and the
isolated launcher/18-minute wrapper integration sources. It contains no native
executable, object file, model, camera image, log, credential or RTSP URL.

## Native patch

`fix5e-plus-fix6-native.patch` applies the existing generation-ownership and
preview-lineage changes from Fix 5E, followed by the Fix 6 upstream progress
probes. Its base is the DeepStream source tree represented by:

`.runtime/deepstream-alarm800-fix-offline-20261031/sources`

Apply it with:

```sh
patch -p1 -d /path/to/deepstream-sources < fix5e-plus-fix6-native.patch
```

The separately retained `fix5e.patch` and `fix6-after-fix5e.patch` are the two
layers used to form the combined native patch. The new native source is picked
up by the existing `Makefile` `*.c` wildcard; no build-file change is needed.

## Isolated host integration sources

`project-integration-sources.patch` adds these source-only review inputs:

- Fix 6 isolated host launcher source (at its candidate path under `.runtime`)
- `scripts/six_camera_live/validate_18m.py`
- `tests/test_validate_18m.py`
- `tests/test_fix6_upstream_diagnostics.py`
- `tests/native/fix6_upstream_probe_test.c`

Apply this add-only snapshot from a clean project root where those paths do not
already exist:

```sh
patch -p1 -d /path/to/project < project-integration-sources.patch
```

The candidate executable is intentionally excluded and must be rebuilt using
the pinned DeepStream 9.1 development image. The verified candidate executable
for this review is SHA256
`3cea6475a2474f903400825dba8f6f5f705db2fc93d9f49d7b6a5202f582f3b4` with
Build ID `eee53522bf57a62ddba9e94e5783b56ea275accb`. The isolated host
launcher source is SHA256
`611130f49ba5e0e0e54d4dc654376777638aa0fba278a9e0f0118ab1d010a581`.

The build image is pinned to
`sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`.
The existing physical-validation runtime image remains the separate digest
already pinned in `config/deepstream-platform.json`.

Verify the patch checksums against `SHA256SUMS`. These sources and patches are
for offline review only; no physical validation is initiated by applying them.
