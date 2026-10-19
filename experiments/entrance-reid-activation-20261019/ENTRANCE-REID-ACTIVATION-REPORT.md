# Entrance Room — Appearance Re-ID Activation Report

Date: 2026-10-19  
Verdict: **PARTIAL**

## Result

Appearance-based Re-ID was **not activated or validated**. NVIDIA's DeepStream 9.1 TAO ReidentificationNet artifact and NvDCF configuration are available, and a test-only configuration was prepared. Every attempted DeepStream run stopped while constructing the source mux, before opening either recording, running detection/tracking, building/loading the Re-ID engine, or producing feature vectors. The configuration's presence is not evidence of model inference.

No return-person match or false-merge outcome can therefore be claimed. The existing fail-closed `PersonIdentityMemory` was not given embeddings and continues to preserve separate provisional identities rather than merge uncertain observations.

## Model and integration status

| Item | Result |
|---|---|
| Re-ID model selected for isolated validation | NVIDIA TAO ReidentificationNet, `resnet50_market1501.etlt` |
| Artifact | `/opt/nvidia/deepstream/deepstream-7.1/samples/models/Tracker/resnet50_market1501.etlt`; 96,377,716 bytes; SHA-256 `0e5b7f702ce7e3734e45f27b819866f15f6b083048481d1afae2089136e20201` |
| Model actually loaded/inferred | **No** |
| DeepStream test image | `nvcr.io/nvidia/deepstream:9.1-triton-multiarch`; image ID `sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b` |
| Runtime | DeepStream 9.1.0, TensorRT 10.16, CUDA 13.2; RTX 3060 12 GB visible to the isolated test environment |
| TensorRT engine | Not built; inference did not start |
| OSNet alternative | Available at `/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/experiments/deepstream/dev-room-pn263-global-identity-20260921/models/reid/osnet_ain_x1_0_msmt17.pth`, 17,293,009 bytes, SHA-256 `8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f`; standalone PyTorch checkpoint, not verified as an NvTracker TensorRT model |
| OSNet similarity threshold | Not set. Earlier exploratory similarity values are uncalibrated and were not reused. |

The NVIDIA TAO model is the chosen candidate because DeepStream 9.1's bundled `tracker_ReID/README` and NvDCF accuracy configuration describe this model family and the Re-ID-based reassociation path. This is a compatibility rationale, not a successful RTX 3060 inference result.

The isolated experiment files are under:

- `experiments/entrance-reid-activation-20261019/config-tracker-reid.yml`
- `experiments/entrance-reid-activation-20261019/deepstream-app.ini`
- `experiments/entrance-reid-activation-20261019/config-infer-primary.ini`

The tracker file is a test-only copy of the frozen B configuration with a Re-ID block (`reidType: 2`, `reidFeatureSize: 256`, `reidHistorySize: 100`, `inferDims: [3,256,128]`, `outputReidTensor: 1`) and a model path to the isolated read-only mount. The existing appearance matching weight remains zero in this diagnostic candidate, so it was not intended to make identity decisions. None of the three files changes production or the preserved offline event data.

## Why validation stopped

Six bounded, offline-only DeepStream app attempts were run in disposable, network-isolated containers. The logs consistently show a source/mux configuration failure before the pipeline reached PAUSED:

- `Output width not set` from `src_bin_muxer` in four attempts.
- `Batch size not set` from `src_bin_muxer` in two attempts using the new mux/source-list variants.
- Then `Failed to set pipeline to PAUSED` and process exit 255.

The official DeepStream sample syntax was checked. The local config includes source-list inputs, `max-batch-size`, and streammux width/height, but the app's source-bin mux still did not receive those values. This points to an unresolved app/source-bin configuration integration issue, not a demonstrated decoder, Re-ID model, or TensorRT failure. Since no source frames were consumed, the logs cannot establish which decoder would have been selected in this run.

Disposable log directories:

- `/tmp/entrance-reid-activation-20261019-wjArvr`
- `/tmp/entrance-reid-activation-20261019-retry-QAVNQd`
- `/tmp/entrance-reid-activation-20261019-run-w6Mndb`
- `/tmp/entrance-reid-activation-20261019-verifiedrun-Lng2G0`
- `/tmp/entrance-reid-activation-20261019-finaltry-2xZ7ej`
- `/tmp/entrance-reid-activation-20261019-source-list-1c7Mf3`

No container remains running. No feature output directory was populated. Re-ID embedding extraction is **BLOCKED / not validated**; there are zero newly produced vectors, not zero-valued vectors.

NVIDIA documents that NvTracker can emit per-object Re-ID feature vectors using `outputReidTensor: 1` and the DeepStream app's `reid-track-output-dir`, provided the selected tracker path supports it and inference is configured at the needed interval. NVIDIA also exposes Re-ID metadata through tracker APIs. Those APIs are a supported integration direction, but they were not exercised by the failed app launch.

## Video, identities, and evaluation

The preserved B run remains the only available 180-second event set. Its read-only SQLite audit found 34,539 observations and no appearance embeddings. Its camera videos and event database were not changed. The original B configuration has no Re-ID section; its Re-ID similarity weight and minimum are zero. The prior report's `PersonIdentityMemory` implementation is fail-closed and has no calibrated similarity threshold.

Potential return-person annotations, **not confirmed matches**:

| Candidate | CAM-05 (camera_1) | CAM-02 (camera_0) | What must be labeled |
|---|---|---|---|
| raw GID 7 → raw GID 8 | 7 observed at frames 1969–2521 (65.63–84.03 s); 8 starts at 2633 (87.77 s) | 7 at frames 2140–2416 (71.33–80.53 s); 8 starts at 2672 (89.07 s) | Same physical person, different people, or cannot tell? |
| raw GID 12 → raw GID 13 | 12 at frames 3652–3704 (121.73–123.47 s); 13 starts at 3787 (126.23 s) | 12 at frames 3655–3671 (121.83–122.37 s); 13 starts at 3907 (130.23 s) | Same physical person, different people, or cannot tell? |

Potential hard negatives that must be reviewed before setting thresholds:

- CAM-05 raw GID 8 overlaps raw GID 10 at frames 2788–2843 (92.93–94.77 s).
- CAM-05 raw GID 8 overlaps raw GID 11 at frames 2941–3088 (98.03–102.93 s).

For each overlap, label whether the boxes are the same physical person duplicated, visually distinct people, or uncertain. Do not merge based on crop similarity alone. Existing candidate contact sheets are `/tmp/entrance-persist-id/b-candidate-gaps.png` and `/tmp/entrance-persist-id/osnet-candidate-crops.png`; they are useful for selecting annotation intervals but do not settle identity. The prior exploratory OSNet similarities (including 0.626, 0.673, and a high 0.792 candidate) are explicitly not calibrated acceptance thresholds.

Ground-truth return examples: **none verified**.  
Correct re-identifications: **not measured**.  
False merges: **not measured** (not zero by evidence).  
ID switches, fragmentation, and duplicate tracks under a Re-ID-enabled configuration: **not measured**.  
Threshold calibration and A/B comparison: **not run**.  
GPU inference throughput/latency: **not measured**. A pre-test device snapshot showed 0% utilization and about 2.6 GiB used, but this is not Re-ID performance data.

The existing review video remains unchanged at `/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/full/entrance-three-panel-full-v2.mp4`. It is a raw-ID baseline without persistent-ID annotations. No new three-panel persistent-ID video or UI was generated because no embeddings or verified identity labels exist.

## Regression tests and preservation

Focused source-free tests:

```text
30 passed, 1 warning in 1.46s
tests/test_mv3dt_person_identity_memory.py
tests/test_mv3dt_staging.py
```

The warning is a Starlette/httpx test-client deprecation notice. The isolated experiment directory passes `git diff --check`. The B videos, B SQLite evidence, historical tracker configurations, calibration, production services, and camera configuration were left unchanged.

## Required next step

Before any threshold or persistent-identity claim, please label the two candidate return transitions and the two same-camera overlap windows above as `same person`, `different people`, or `uncertain`. If the available views cannot support those labels, mark them uncertain and provide a few additional independently verified examples; do not force a match.

After annotation, the remaining technical prerequisite is to correct the isolated DeepStream source/mux configuration so the exact 9.1 app starts and decodes these files. Then verify real 256-D feature output from the NVIDIA model, join vectors to raw camera/frame/track records, calibrate match and ambiguity thresholds on the labeled positives/negatives, run one bounded B-vs-Re-ID A/B test, and only then create a review UI/video that shows raw and persistent IDs separately.

## NVIDIA references

- [DeepStream 9.1 Re-ID metadata API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/struct__NvMOTQuery.html)
- [DeepStream 9.1 FAQ](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_FAQ.html)
- [DeepStream 9.1 config-file parser API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/9_81_2sources_2apps_2apps-common_2includes_2deepstream__config__file__parser_8h.html)
- [NVIDIA TAO ReidentificationNet model catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet)
