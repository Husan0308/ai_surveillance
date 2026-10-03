# F5R — six-camera YOLO26m resolution A/B

**Result: PASS with the existing 640×640 profile retained.** The isolated 960×960 profile is rejected: it exceeded the required UI-latency limit on all six cameras and saturated the RTX 3060 at peak. Nothing was promoted or changed in production.

## Test topology and provenance

Both 330-second trials used one DeepStream process in the pinned DeepStream 9.1.0 image, one RTSP/NVDEC source per camera, `nvstreammux` batch 6, YOLO26m `nvinfer`, the custom raw-output parser, and NvDCF. Each decoder tees to the unchanged bounded/non-leaky analytics input and a one-buffer latest-only shared-memory preview branch. The PySide wall consumes shared memory; it does not open RTSP. YOLO and NvDCF execute inside the DeepStream process. The UI preview branch is before inference by design; the test UI does not paint detector boxes.

The host has DeepStream 7.1 installed. An earlier host-linked test binary was detected and explicitly excluded. Valid results below came only from the 9.1 container build and 9.1 runtime. Container image: `nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f` (DeepStream 9.1.0, CUDA runtime 13.2, TensorRT 10.16.1).

The fixed detector semantics were FP16, person class 0, confidence 0.25, NMS IoU 0.45, interval 0, batch 6, `maintain-aspect-ratio=1`, and `symmetric-padding=1`. The tracker was the six-camera `NvDCF_stable_person` profile; V13 was not used or changed. All six camera-specific RTSP latency/drop/decoder settings were copied from the existing runtime configuration without logging credentials.

| Artifact | SHA256 |
| --- | --- |
| YOLO26m 640 ONNX | `d3aa1ab159dba902eea9983df297cd83a56e39f62138082de7f0642958f13f33` |
| YOLO26m 640 FP16 engine, batch 6 | `e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16` |
| YOLO26m 960 ONNX | `5ab90ca575146336ae9f02c479c6f21220cf0997e7cab19eb62d48b71e5a3c8d` |
| YOLO26m 960 FP16 engine, batch 6 | `2f4741f109db637546c7abfec105fc28f5f3508649e8b9b41fa340476f32ada8` |
| DeepStream 9.1 parser, used in both trials | `32766d0cc5855d243f9ec9c46ea959a3f88f250b6ca024b293f1b6b7cd7a949f` |
| staged native validation executable | `2bc945ac82d808e126c70101ab8d1559447f8c027d52f615cea56d9574486b44` |
| staged 640 PGIE config | `fbfa196338b6781b7d65c3058d4f02069e9116963804d0c3b23d1bf2301e4113` |
| staged 960 PGIE config | `b518043d21b1a206cc73369cc61e3e7f6614eb59894d2abdfe74b254bd6eaf4b` |
| six-camera NvDCF profile | `36d4c1e10afcf04adc0a482665a707b512f65788e145f24975512e9271397459` |

The 960 ONNX is a fresh Ultralytics 8.4.155 export from the same weights (SHA256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`), opset 17, raw NMS-free output `84×18900`, dynamic batch min 1 / opt 6 / max 6. TensorRT 10.16.1 built a true 960×960 engine; DeepStream deserialized the exact engine path and the parser accepted the 960 output shape with zero runtime parser errors. It was not a 640 engine with a changed `infer-dims` value.

## Six-camera live results

Each cell is `source / PGIE / NvDCF FPS`; `props` is raw person-class proposals, `track rows / IDs` is actual NvDCF output, `overlap` is a same-frame bbox-overlap estimate described below, `gap` is its longest raw-proposal-only sequence, and `queue` is sampled per-source analytics queue high-water.

| Camera | 640 FPS | 640 props; track rows / IDs; overlap; gap; queue | 960 FPS | 960 props; track rows / IDs; overlap; gap; queue |
| --- | ---: | --- | ---: | --- |
| CAM-01 | 19.82 / 19.82 / 19.82 | 1; 0 / 0; 0%; 1; 0 | 19.85 / 19.85 / 19.85 | 0; 0 / 0; 0%; 0; 0 |
| CAM-02 | 19.79 / 19.79 / 19.79 | 857; 76 / 9; 8.87%; 25; 0 | 19.85 / 19.85 / 19.84 | 102; 0 / 0; 0%; 20; 2 |
| CAM-03 | 19.81 / 19.80 / 19.80 | 0; 0 / 0; n/a; n/a; 0 | 19.84 / 19.84 / 19.83 | 0; 0 / 0; n/a; n/a; 1 |
| CAM-04 | 19.97 / 19.97 / 19.97 | 1,839; 96 / 11; 5.22%; 26; 0 | 19.96 / 19.95 / 19.95 | 852; 2 / 1; 0.23%; 18; 1 |
| CAM-05 | 19.96 / 19.96 / 19.96 | 1,069; 15 / 7; 1.40%; 5; 0 | 19.95 / 19.95 / 19.94 | 84; 0 / 0; 0%; 2; 0 |
| CAM-06 | 19.96 / 19.95 / 19.95 | 0; 0 / 0; n/a; n/a; 0 | 19.95 / 19.95 / 19.94 | 0; 0 / 0; n/a; n/a; 0 |

The `overlap` estimate greedily matches detector and tracker boxes in the same source/frame at IoU ≥0.10. It is not an NvDCF association trace and must not be read as person retention. Both representative wall captures showed no visible occupants; there was no person-positive ground truth in these live windows. Therefore person recall, small/partial-person improvement, and person-level tracker gaps are **not established** by this run. Raw proposal totals were 3,766 at 640 and 1,038 at 960; that 72% decrease may represent fewer background proposals, missed people, or both. It is not valid evidence of improved recall. Same-frame duplicate-box pairs at IoU ≥0.85 were zero in both trials.

### Decoder-reference → actual UI paint

P50 / P95 / P99 / max, milliseconds. Each camera had more than 5,600 valid matched-frame samples; malformed and invalid timing rows were zero.

| Camera | 640×640 | 960×960 |
| --- | ---: | ---: |
| CAM-01 | 15.65 / **25.95** / 34.88 / 222.48 | 18.29 / **40.61** / 63.51 / 379.14 |
| CAM-02 | 17.03 / **28.85** / 35.99 / 207.99 | 19.46 / **40.58** / 59.71 / 204.00 |
| CAM-03 | 16.80 / **29.55** / 37.07 / 55.23 | 19.75 / **43.25** / 64.38 / 186.75 |
| CAM-04 | 14.91 / **28.61** / 36.42 / 56.76 | 17.44 / **40.09** / 58.61 / 179.24 |
| CAM-05 | 14.14 / **28.52** / 38.51 / 201.38 | 17.19 / **43.40** / 71.75 / 448.70 |
| CAM-06 | 14.91 / **30.68** / 39.82 / 232.40 | 17.26 / **45.38** / 77.22 / 746.88 |

All 640 P95 values pass `<40 ms`; all six 960 P95 values fail. The 640 UI measured median preview rates were 19.87–20.03 FPS; 960 was 19.90–20.04 FPS. UI paint cadence remained near 20 FPS in both, but 960 added decoder-stage tail latency under load.

## Resource, ownership, and safety

| Measurement | 640×640 | 960×960 |
| --- | ---: | ---: |
| TRT/nvinfer element mean / max latency | 29.79 / 114.28 ms | 69.77 / 234.89 ms |
| PGIE input batches/sec (batch size 6) | 22.24 | 22.38 |
| GPU utilization mean / peak | 46.7% / 76% | 84.6% / 100% |
| GPU memory mean / peak | 2,854 / 2,921 MiB | 3,323 / 3,391 MiB |
| Native process CPU mean / peak | 71.4% / 86.7% | 96.2% / 102.3% |
| RTSP connections during steady run | exactly 6 | exactly 6 |
| API / ML healthy UI samples | 660 / 660 | 660 / 660 |
| UI samples showing 6/6 live | 660 | 660 |
| UI network readers | 0 | 0 |
| candidate RTSP sockets / containers after shutdown | 0 / 0 | 0 / 0 |

All six source decoder profiles were logged as one hardware decoder per camera with the existing low-latency/drop/extra-surface settings. During both runs each source, PGIE source frame count, and tracker batch frame count stayed approximately 20 FPS. Source error, warning, backward-PTS, and duplicate-PTS counters were zero. Analytics queues were bounded; sampled high-water was zero for 640 and at most two buffers for 960, with no sustained growth. End-of-run input-vs-PGIE/tracker counter deficits were 1–2 frames per camera at 640 and 2–4 at 960; they were retained as shutdown-boundary counter deltas, not excluded from the report. No reconnect was observed. API/ML returned to truthful degraded/offline health after candidate shutdown because fresh preview producers stopped.

## Decision and disposition

**Winner: 640×640 retained.** It is the only tested resolution that satisfies the six-camera `<40 ms` UI P95 requirement with substantial GPU headroom. The 960 candidate materially increases inference latency and reaches 100% GPU utilization; its proposal reduction cannot override those failures or establish recall. A person-present, labeled matched-frame comparison would be needed before asserting that 960 improves small/far/partial-person detection.

This is a staging decision only. Production YOLO config/engine, V13, production manifest/binary, calibration, and frozen F5 evidence are unchanged. No external Python/OpenCV detector or second camera owner was used. The initial DeepStream 7.1 host-linked candidate is not part of these results.

Machine-readable metrics, both raw run directories, exact source/config/model hashes, parser test output, and protected-file before/after hashes are under `.runtime/freeze/F5R-six-camera-yolo-resolution-20261003/`. The resolution test does not replace the separately frozen F5 validation or authorize production promotion.
