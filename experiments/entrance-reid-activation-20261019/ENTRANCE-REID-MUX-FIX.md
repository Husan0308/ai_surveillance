# Entrance Room — DeepStream mux and Re-ID feature validation

Date: 2026-10-19  
Verdict: **REID_FEATURE_EXPORT_PASS** (feature-export path validated; persistent identity matching remains unvalidated)

## Outcome

The isolated DeepStream 9.1 pipeline now starts with both Entrance Room video sources, runs PeopleNetTransformer and NvDCF, builds and loads NVIDIA TAO ReIdentificationNet, and exports real per-object 256-element normalized features from both cameras. The exported features were not used to merge IDs, and no PersonIdentityMemory behavior or match threshold was changed.

The previous startup errors were a mux/configuration integration failure, not a Re-ID model failure. The logs identify `src_bin_muxer` as the failing element. In the first six attempts using the generic `deepstream-app` source-bin path, four logged `Output width not set` and two source-list attempts logged `Batch size not set`. Two later generic-app trials also logged `Output width not set`. The source-bin's internal legacy mux did not receive valid output dimensions; setting the top-level `[streammux]` values alone did not initialize that internal `src_bin_muxer`. The exact original shell command line and environment dump were not retained, so they cannot be reproduced verbatim; the preserved logs do establish the executable-specific mux errors.

The retained first-round mux-failure logs are `/tmp/entrance-reid-activation-20261019-{wjArvr,run-w6Mndb,verifiedrun-Lng2G0,finaltry-2xZ7ej}/deepstream-reid.log` (`Output width not set`) and `/tmp/entrance-reid-activation-20261019-{source-list-1c7Mf3,retry-QAVNQd}/deepstream-reid.log` (`Batch size not set`). Two later generic-app failure logs are `/tmp/entrance-reid-mux-fix-2bJ7d9c9/{attempt-uri2,outputs}/deepstream-reid.log`, also `Output width not set`. A later `deepstream-test5-app` trial reached inference and built NvDCF's engine, but `/tmp/entrance-reid-mux-fix-2bJ7d9c9/test5-app/deepstream-reid.log` records the separate missing PeopleNetTransformer parser error. The final run's log records the corrected path.

The correction was to use the official MV3DT-compatible `deepstream-test5-app -c ...` path with ordinary `[source0]`/`[source1]` file sources and a configured legacy `[streammux]` in the application configuration. This produced the intended top-level `nvstreammux` with `width=1920`, `height=1080`, `batch-size=2`, `live-source=0`. `USE_NEW_NVSTREAMMUX=no` was scoped to the disposable process; no unsupported new-mux width/height settings were used. NVIDIA's custom-dataset skill itself launches `deepstream-test5-app` with a generated DeepStream application config. See [NVIDIA's DeepStream 9.1 nvstreammux documentation](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvstreammux2.html) and the locally installed `deepstream-run-mv3dt/references/custom-dataset.md`.

One intermediate `deepstream-test5-app` attempt got past mux startup and built the Re-ID TensorRT plan but failed at the detector parser (`Could not find output coverage layer` / `Failed to parse bboxes`), because the PeopleNetTransformer parser function was omitted. The final application config specifies NVIDIA's `NvDsInferParseCustomDDETRTAO` parser and completed inference without those parser errors.

## Mux and detector configuration

The successful experiment configuration is in:

- `experiments/entrance-reid-activation-20261019/deepstream-app.ini`
- `experiments/entrance-reid-activation-20261019/config-infer-primary.ini`
- `experiments/entrance-reid-activation-20261019/config-tracker-nvdcf-reid.yml`

The app config uses source 0=`cam_00.mp4` (CAM-02) and source 1=`cam_01.mp4` (CAM-05), with a two-source legacy mux at 1920×1080. The detector config loads the existing NVIDIA PeopleNetTransformer ONNX/engine and parser. Its engine and the MV3DT camera-order mapping were not modified.

Config SHA-256: `deepstream-app.ini` `0a83cebf8600aa46fa342c34480868cc34015029dbd6870031971b611159b01c`; `config-infer-primary.ini` `9a39947c2efa45fdc337d9e84d5cac40f1e7d83e1629aaad4f6f0a870e888dfe`; `config-tracker-nvdcf-reid.yml` `dc32f4226f87c0b9818777b53bba4e94d4aae2e3393e3e95277441df5dd64332`.

The successful run was a disposable, offline container using image:

```text
nvcr.io/nvidia/deepstream:9.1-triton-multiarch
ID sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b
DeepStream 9.1.0 / TensorRT 10.16 / CUDA 13.2
```

It ran with `--network=none --gpus all --cpus=4 --memory=8g --read-only`, a temporary `/tmp` tmpfs, read-only video/config/PeopleNet mounts, and a disposable writable Re-ID-model directory (for the TensorRT cache) and output directory. It did not use privileged mode or host networking. The entry point was:

```bash
USE_NEW_NVSTREAMMUX=no GST_REGISTRY=/tmp/gst-registry.bin \
GST_DEBUG=2 deepstream-test5-app -c /config/deepstream-app.ini
```

The actual container invocation bounded this command with `timeout --signal=INT --kill-after=10s 300s`, captured logs to `final-run/deepstream-reid.log`, and sampled device GPU utilization/memory once per second. The final app handled the interrupt, sent EOS for source IDs 0 and 1, and logged `App run successful`. The outer `timeout` returned 124 because its five-minute bound elapsed; this is not evidence that the 180-second files completed.

The reusable Re-ID configuration is based on NVIDIA's DeepStream 9.1 NvDCF accuracy configuration. It sets `reidType: 2`, `reidFeatureSize: 256`, `inferDims: [3,256,128]`, `batchSize: 8`, `reidHistorySize: 100`, `reidExtractionInterval: 0`, `addFeatureNormalization: 1`, and `outputReidTensor: 1`. The Re-ID matching weight remains **0.0**. Therefore this run extracts appearance features but deliberately does not let appearance change the track association decisions. `reid-track-output-dir=/outputs/reid-features` is set in the application config. NVIDIA describes the tracker feature-metadata/API path in its [DeepStream 9.1 NvMOT API](https://docs.nvidia.com/metropolis/deepstream/9.1/sdk-api/struct__NvMOTQuery.html).

## Re-ID inference evidence

The model was the installed NVIDIA TAO ReidentificationNet artifact `resnet50_market1501.etlt`, SHA-256:

```text
0e5b7f702ce7e3734e45f27b819866f15f6b083048481d1afae2089136e20201
```

It was copied into the disposable `/tmp/entrance-reid-mux-fix-2bJ7d9c9/reid-model/` only. NvDCF generated and loaded `resnet50_market1501.etlt_b8_gpu0_fp16.engine` (48,462,668 bytes) there; the TensorRT plan was not installed into the DeepStream image or production model directories. The log records the engine serialization and `[NvMultiObjectTracker] Initialized`.

`outputReidTensor: 1` plus the app's `reid-track-output-dir` produced per-frame text files. DeepStream app output naming is `app-index / stream-pad-index / frame-number`; therefore `00_000_*` maps to configured source 0/CAM-02 and `00_001_*` to source 1/CAM-05. Each nonempty line contains a tracker object ID and 256 feature values. Validation found:

| Stream | Files / frame indices | Feature rows | Malformed rows | Dimensions | L2 norm range | Distinct local tracker IDs |
|---|---:|---:|---:|---:|---:|---:|
| CAM-02 / source 0 | 3,693 / 0–3,692 | 11,920 | 0 | 256 | 0.999998–1.000002 | 7 |
| CAM-05 / source 1 | 3,693 / 0–3,692 | 12,105 | 0 | 256 | 0.999999–1.000002 | 9 |

Frames 0 and 1 had empty feature files on each stream; the remaining files contained finite 256D vectors. The files cover approximately 123.1 seconds of the 180-second clips (about 68.4%), not the complete recordings. Thus this confirms genuine ReidentificationNet inference and feature export for both sources, but does not provide feature evidence for the later GID-13 observations (CAM-05 frame 3787 onward; CAM-02 frame 3907 onward). The exported tracker IDs belong to this isolated NvDCF run; they are not the raw MV3DT global IDs from the preserved A/B database and must not be joined by numeric ID alone.

The `gpu-samples.csv` device-wide sample covers 05:23:28–05:28:30 local host time (287 samples): mean GPU utilization 87.66%, peak 100%; device memory was 2,622–3,653 MiB. These are device-total measurements, not per-process attribution. The final app-reported average was about 14 FPS per source; this was an offline partial pass and not a real-time performance claim.

The two normalized video inputs were mounted read-only:

```text
CAM-02 cam_00.mp4  SHA-256 01f316f1146bfa7bc5dbc760012aaa562f15daca70e10a2cd7e408e117b32160
CAM-05 cam_01.mp4  SHA-256 dd9e851ba25013c88ad1059f44c3ba41e1060d018ed778f45f373152066ed813
```

Source-frame numbers in the preserved event database are used for the candidate sheets. They provide nominal clip times at 30 FPS only; prior cross-camera synchronization uncertainty remains approximately ±150 ms. The image sheets are snapshots for human labeling, not synchronized exposure proof.

## Candidate review images — labels requested

All four are owner-only PNGs (`0600`) under `/tmp/entrance-reid-mux-fix-2bJ7d9c9/contact-sheets/`:

1. `candidate-gid7-to-gid8.png` — prior/new raw-ID examples across CAM-02 and CAM-05.
2. `candidate-gid12-to-gid13.png` — prior/new raw-ID examples across CAM-02 and CAM-05.
3. `candidate-gid8-versus-gid10.png` — CAM-05 GID 8 at frame 2637 and GID 10 at frame 2816, 179 frames (about 5.97 nominal seconds) apart. The stored observations do **not** show simultaneous same-frame GID 8/GID 10 observations in this interval; this is a sequential appearance comparison, not a verified hard negative.
4. `candidate-gid8-versus-gid11.png` — CAM-05 frame 3087, where both IDs are recorded and their boxes substantially overlap (IoU about 0.61). This is a possible duplicate/fragmentation case, not a known different-person example.

Please label each candidate `SAME_PERSON`, `DIFFERENT_PEOPLE`, or `UNCERTAIN`. In particular, GID 8 vs 10 requires a human judgment despite being sequential, while GID 8 vs 11 should be judged as same/duplicate, different people, or uncertain based on the single-frame visual evidence. No labels have been presumed.

| Image | SHA-256 |
|---|---|
| `candidate-gid7-to-gid8.png` | `3b53a072a97a3b8a332bf76db184e654c31e69a8748b5d6695bf1b7bfec33320` |
| `candidate-gid12-to-gid13.png` | `d5fbdfdc0b94523f05755c2f7436208285585ba160eef420e14624dabf9400a6` |
| `candidate-gid8-versus-gid10.png` | `6f7cca7b05136a77a84b5e4aa40a5ca3ef0b296ce8a4762daeb1f9dae33c71a2` |
| `candidate-gid8-versus-gid11.png` | `5e39d31e98b6f512fd186f2a7c8f8c694077cc4bd2de519a4f8fca7af2035428` |

## Explicit status and remaining work

- **Mux startup:** fixed for this isolated `deepstream-test5-app` config; legacy mux selected explicitly and both sources reached `Active sources: 2`.
- **PeopleNetTransformer:** inference path ran with the custom parser; no missing-coverage/parser failure in the final log.
- **NvDCF / NVIDIA Re-ID:** model plan built/loaded and valid vectors exported for both streams.
- **Persistent identities:** **not validated or fixed**. No visual labels were received; no thresholds were calibrated; no new feature-based identity match or A/B evaluation was performed; no raw global IDs were merged.
- **Full recording coverage:** incomplete due the explicit 300-second bound; later GID-13 events lack new Re-ID vectors.
- **Geometry/time:** inherited Entrance Room calibration and approximately ±150 ms sync uncertainty remain experimental and unchanged.
- **Production and source data:** no production services, tracker configuration, original videos, calibration, or raw event database were modified. The experiment did not connect to RTSP, run AMC/VGGT, or write to production databases. The only writable model artifact and run outputs were in the disposable `/tmp` tree; experiment configuration/helper files are under this repository's isolated `experiments/entrance-reid-activation-20261019/` directory.

After receiving the four human labels, the next safe step is to map those labeled visual crops to the per-frame NvDCF feature vectors using camera + frame + bounding-box evidence, examine positive/hard-negative similarity distributions, and only then consider a fail-closed IdentityMemory change. GID 12→13 additionally needs feature extraction over the later interval before it can be evaluated with this model.

## Evidence files

- DeepStream log: `/tmp/entrance-reid-mux-fix-2bJ7d9c9/final-run/deepstream-reid.log` (SHA-256 `9b87eb0189016ba72cc977a92512493bd2962e61ca96124744499060d9fbe8e9`)
- Device GPU samples: `/tmp/entrance-reid-mux-fix-2bJ7d9c9/final-run/gpu-samples.csv` (SHA-256 `1e8a4785987b5cc5ba5b9147cced75d476cfb59ea6d8b72009e8dc27bb0f29cc`)
- Feature output directory: `/tmp/entrance-reid-mux-fix-2bJ7d9c9/final-run/reid-features/`
- Contact-sheet generator: `experiments/entrance-reid-activation-20261019/make_reid_candidate_sheets.py`

