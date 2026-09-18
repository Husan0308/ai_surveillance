# YOLO26m export gate

This stage starts **after** the six-camera transport/decode/mux/preview baseline
has passed. Do not modify the validated camera graph in this gate.

The goal is only to create and validate the detector artifact that the next
DeepStream stage will consume.

## Export choice

Use Ultralytics YOLO26m detection at 640x640 with the native NMS-free one-to-one
head:

- model: `yolo26m.pt`
- task: detect
- image size: 640
- export: ONNX
- `nms=False`
- dynamic batch with requested max batch 6
- output rows: `[x1, y1, x2, y2, confidence, class_id]`
- expected output layout: normally `(N, 300, 6)`

We intentionally export ONNX rather than committing a prebuilt TensorRT engine.
The TensorRT engine will be built and validated on the deployment RTX 3060 /
DeepStream 9.1 environment in the next gate.

## Run

```bash
cd ~/ai_surveillance

python3 -m pip install -U ultralytics onnx

python3 scripts/yolo26m/export_yolo26m.py
```

Expected artifact:

```text
.runtime/models/yolo26m/yolo26m_nmsfree_b6_dynamic.onnx
```

Expected report:

```text
.runtime/models/yolo26m/export_report.json
```

The command must finish with `"status": "PASS"` and the output tensor's last
dimension must be `6`.

Stop after this export gate. Do not add `nvinfer`, tracker, ReID, face, pose,
heatmap or UI changes until this artifact is verified.
