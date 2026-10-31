# Verified artifact identities

All identities below were checked locally with SHA256 and `readelf -n`. Executable files are intentionally not included in this source snapshot.

| Artifact | SHA256 | Build ID | Scope |
|---|---|---|---|
| Original Fix 5E DeepStream viewer | `87a22a8ddd380676c37229df5791c53b62557356fd28996598561ddf3879ee35` | `8fb26e4f5f5c791e0a4259a887f8ba36eb7af3c6` | Fix 5F runner's pinned candidate only |
| Fix 5F synthetic plugin test executable | `3b06b5eda20fb3bb21bd31c7bae3b199b3b56aa312fd4d5a835c3361798dacf5` | `b0d942aa444a2bfaa7626ae6ab793b789dd8fac8` | Generated harness binary; omitted |
| Newer Fix 6 viewer | `3cea6475a2474f903400825dba8f6f5f705db2fc93d9f49d7b6a5202f582f3b4` | `eee53522bf57a62ddba9e94e5783b56ea275accb` | Fix 6B candidate; not tested by Fix 5F runner |

The Fix 5E source inputs pinned in the runner and verified locally are listed in `SOURCE_PATH_MANIFEST.tsv`. The Fix 6B snapshot at `../fix6b/` remains intact on this branch.
