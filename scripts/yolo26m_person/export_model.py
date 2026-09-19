#!/usr/bin/env python3
"""Export the COCO YOLO26m one-to-one head; this is model preparation, not live inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
import onnx
import ultralytics
from ultralytics import YOLO


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('.runtime/models/yolo26m'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out.resolve() / 'yolo26m.pt'
    if args.weights.resolve() != target:
        shutil.copy2(args.weights, target)
    torch.set_num_threads(4)
    model = YOLO(str(target), task='detect')
    assert model.task == 'detect' and len(model.names) == 80 and model.names[0] == 'person'
    assert model.model.yaml['scale'] == 'm' and '26' in str(model.model.yaml.get('yaml_file', ''))
    assert model.model.model[-1].end2end, 'One-to-one head is required'
    opts = dict(format='onnx', imgsz=640, batch=6, dynamic=True, simplify=True,
                opset=17, nms=False, agnostic_nms=True, end2end=True,
                max_det=300, half=False, device='cpu')
    result = Path(model.export(**opts))
    graph = onnx.load(result)
    # Only batch varies in deployment; preserve symbolic graph computations and fix H/W.
    dims = graph.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_value = 640
    dims[3].dim_value = 640
    output = graph.graph.output[0].type.tensor_type.shape.dim
    assert len(output) == 3 and output[1].dim_value == 300 and output[2].dim_value == 6
    assert not any(n.op_type == 'NonMaxSuppression' for n in graph.graph.node)
    onnx.checker.check_model(graph)
    onnx.save(graph, result)
    (args.out / 'labels.txt').write_text('\n'.join(model.names[i] for i in range(80)) + '\n')
    report = dict(model='yolo26m.pt', task='detect', weights_sha256=sha(target),
                  onnx_sha256=sha(result), ultralytics=ultralytics.__version__,
                  torch=torch.__version__, onnx=onnx.__version__, export=opts,
                  input_shape=['batch', 3, 640, 640], output_shape=['batch', 300, 6],
                  nms_nodes=0, argv=sys.argv)
    (args.out / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
