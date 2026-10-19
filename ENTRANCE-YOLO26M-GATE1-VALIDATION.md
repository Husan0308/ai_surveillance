# Entrance Room YOLO26m — Gate 1 Validation

Verdict: **GATE1_PASS**

Gate 1 is the offline/static contract only. It does not establish TensorRT
batch-2 engine compatibility, inference quality, GPU performance, or live
camera readiness. Gate 2 was not run.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| `GATE1_STATIC_CONFIG` | PASS | Frozen PGIE contract validated; generated candidate differs only at `batch-size=6` → `batch-size=2`. |
| `YOLO_ASSET_HASH_CONTRACT` | PASS | The four frozen model/parser/label files in the preserved r12 artifact set matched the repository's `EXPECTED` SHA-256 manifest. The PGIE config also matched. |
| `BATCH2_CONFIG_GENERATION` | PASS | Unit tests and a direct CPU check confirmed a single batch-size substitution; threshold, parser, model, input, interval and preprocessing fields remain byte-for-byte unchanged. |
| `DOUBLE_RECTIFICATION_PREVENTION` | PASS | Existing AMC inputs default to `--input-geometry rectified`; the rectification bridge is only loaded/applied for explicit `raw` input. |
| `CODE_REVIEW` | PASS | YOLO uses its dedicated pinned image, config and read-only artifact mount; PeopleNet remains the default and retains its prior config/image/model mount. No production configuration is referenced for mutation. |
| `REGRESSION_TESTS` | PASS | Focused Gate 1 and live-gate suite: 15 passed, 0 failed. |
| `PRODUCTION_UNCHANGED` | PASS | Only this experimental worktree was edited. No Docker, GPU, RTSP, calibration or service-management command was run. |
| `GATE2_GPU_INFERENCE` | NOT_RUN | Explicitly out of scope for this authorization. |

## Verified static details

- Experimental branch base/fetched commit: `2dd542b7ceca2e38451854bd6ab743a362f99c4c`.
- Frozen primary config SHA-256: `fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113`.
- Frozen assets, checked from the existing r12 artifact store:
  - `yolo26m_raw_otm.onnx`: `d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33`
  - `yolo26m_raw_otm_b6_fp16.engine`: `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16`
  - `libyolo26rawperson.so`: `d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19`
  - `labels.txt`: `bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a`
- Labels contain 80 entries; `person` is class index 0. The frozen detector uses the `NvDsInferParseYolo26RawPerson` parser, `interval=0`, pre-cluster threshold `0.25`, NMS IoU `0.45`, and 640×640 input. These remain unchanged in the staged batch-2 config.
- The parser is an x86-64 ELF with `libstdc++.so.6` and `libc.so.6` as its direct dynamic dependencies. Its bytes match the frozen manifest; actual loading inside DeepStream was not tested.
- YOLO's container image is pinned to `nvcr.io/nvidia/deepstream@sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b`. The constructed YOLO command uses read-only mounts for `/models` and `/workspace/pgie-yolo26m-b2.txt`, `--network none`, and `--pull=never`.
- The YOLO command does not mount or select PeopleNet assets. The default detector remains PeopleNet, and a regression test verifies its existing image, PGIE config and read-only model mount without the YOLO-only `--pull=never` flag.
- Rectified inputs are passed through unchanged; only explicitly raw input constructs per-camera rectification transforms. Camera/frame handling and the class-0 report path were reviewed statically.

## Fixes made in this worktree

1. Corrected the unused `DEFAULT_MODEL_DIR` from the stale r5 directory to the documented and hash-verified r12 profile; added a regression test.
2. Scoped `--pull=never` to YOLO inference so the default PeopleNet command retains its prior pull behavior; added command-contract assertions for both detector paths and ensured the YOLO command has no PeopleNet mount/config dependency.

No production files, model files, calibration artifacts, or camera settings were changed.

## Tests and limits

Command run from `/home/apsidal/ai_surveillance-yolo-gate1`:

```text
/home/apsidal/ai_surveillance-gate3j/.runtime/full-stack-venv/bin/python -B -m pytest -q tests/test_entrance_yolo26m_gate1.py tests/test_entrance_live_tracking_gate.py
...............                                                          [100%]
15 passed in 0.11s
```

Python AST parsing passed for the candidate module, bridge, both focused test
files, and frozen asset-manifest builder. A direct batch-2 render check passed.
`git diff --check` passed. Host C++ syntax compilation was unavailable because
DeepStream/GStreamer development headers are not installed on the host; no
container was started to obtain them.

The fetched artifacts live in the existing checkout's ignored `.runtime`
store, not in this isolated worktree. The isolated worktree's `.runtime` is
currently absent. Before a later authorized Gate 2 run, stage verified copies
under this worktree's `.runtime/sixcam-approved-bbox-20261006-r12-profile/models`
so the candidate's containment/hash validation succeeds. No copy was made in
Gate 1. The filename `*_b6_fp16.engine` does **not** prove batch-2 runtime
compatibility; only a real nvinfer initialization/inference test can establish
that.

No RTSP session, Docker container, GPU inference, calibration, or Gate 2 A/B
comparison was performed. Gate 1 does not authorize proceeding automatically.

## Commit

The implementation and this report were tested in the isolated experimental
worktree. The final branch commit SHA is recorded in the delivery message.
