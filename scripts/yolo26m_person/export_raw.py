#!/usr/bin/env python3
"""Export raw YOLO26m one-to-many predictions, leaving NMS to DeepStream."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.runtime/exporter-8.4.155'))
import onnx
import torch
import ultralytics
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.runtime/models/yolo26m'

def main():
    weights=OUT/'yolo26m.pt'
    target=OUT/'yolo26m_raw_otm.pt'
    if (OUT/'yolo26m_raw_otm.onnx').exists():raise RuntimeError('Refusing to overwrite raw export evidence')
    shutil.copy2(weights,target)
    torch.set_num_threads(4)
    model=YOLO(str(target),task='detect')
    assert model.task=='detect' and len(model.names)==80 and model.names[0]=='person'
    assert model.model.yaml['scale']=='m' and '26' in str(model.model.yaml['yaml_file'])
    assert ultralytics.__version__=='8.4.155', 'Use the pinned isolated exporter'
    # Inspect the actual unfused one-to-many tensor before export.
    model.model.end2end=False
    model.model.eval().to('cuda:0')
    with torch.no_grad():
        sample=model.model(torch.zeros(1,3,640,640,device='cuda:0'))
        raw=sample[0] if isinstance(sample,tuple) else sample
        assert tuple(raw.shape)==(1,84,8400),tuple(raw.shape)
    opts=dict(format='onnx',imgsz=640,batch=6,dynamic=True,simplify=True,
              opset=17,device=0,nms=None,half=False)
    result=Path(model.export(**opts))
    graph=onnx.load(result)
    assert len(graph.graph.input)==1 and len(graph.graph.output)==1
    dims=graph.graph.input[0].type.tensor_type.shape.dim
    assert len(dims)==4 and dims[1].dim_value==3
    dims[2].dim_value=640;dims[3].dim_value=640
    output=graph.graph.output[0]
    od=output.type.tensor_type.shape.dim
    assert len(od)==3 and (od[1].dim_value==84 or od[1].dim_param)
    assert not any(n.op_type=='NonMaxSuppression' for n in graph.graph.node)
    onnx.checker.check_model(graph);onnx.save(graph,result)
    # This is an offline synthetic tensor export check, never camera inference.
    import onnxruntime as ort
    import numpy as np
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(result),sess_options=options,providers=['CPUExecutionProvider'])
    for batch in (1,2):
        outputs=session.run(None,{graph.graph.input[0].name:np.zeros((batch,3,640,640),dtype=np.float32)})
        assert len(outputs)==1 and outputs[0].shape==(batch,84,8400)
        assert np.isfinite(outputs[0]).all()
    od[1].dim_value=84;od[2].dim_value=8400
    onnx.checker.check_model(graph);onnx.save(graph,result)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=dict(model='yolo26m.pt',weights_sha256=sha(weights),onnx_sha256=sha(result),
                ultralytics=ultralytics.__version__,torch=torch.__version__,onnx=onnx.__version__,
                export=opts,input_name=graph.graph.input[0].name,input_shape=['batch',3,640,640],
                output_name=output.name,output_shape=['batch',84,8400],bbox_format='cxcywh',
                scores='80 sigmoid class probabilities, no objectness channel',nms_nodes=0,
                onnx_runtime_shape_checks=[1,2],argv=sys.argv)
    (OUT/'export_raw_otm.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
