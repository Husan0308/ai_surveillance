#!/usr/bin/env python3
"""Build/test the parser and build the FP16 engine on the deployment GPU/container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parser-only', action='store_true')
    ap.add_argument('--raw-otm',action='store_true')
    args=ap.parse_args()
    out=ROOT/'.runtime/models/yolo26m';out.mkdir(parents=True,exist_ok=True)
    deps=ROOT/'.runtime/build-deps/cuda13.2'
    deps.mkdir(parents=True,exist_ok=True)
    for package in ('cuda-cudart-dev', 'cuda-crt'):
        deb=deps/f'{package}.deb'
        if not deb.exists():
            url=f'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/{package}-13-2_13.2.51-1_amd64.deb'
            urllib.request.urlretrieve(url, deb)
        subprocess.run(['dpkg-deb','-x',str(deb),str(deps)],check=True)
    platform=json.loads((ROOT/'config/deepstream-platform.json').read_text())
    base=['docker','run','--rm','--network=none','--user',f'{os.getuid()}:{os.getgid()}',
          '-v',f'{out}:/models','-v',f'{ROOT}/.runtime/build-deps/cuda13.2:/cuda-headers:ro','-v',f'{ROOT}/scripts/yolo26m_person:/src:ro']
    script='''set -eu
inc=/opt/nvidia/deepstream/deepstream/sources/includes
g++ -O2 -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -fPIC -shared -isystem "$inc" -I/cuda-headers/usr/local/cuda-13.2/targets/x86_64-linux/include /src/parser.cpp -o /models/libyolo26person.so
g++ -O2 -std=c++17 -Wall -Wextra -Werror -Wno-unused-parameter -isystem "$inc" -I/cuda-headers/usr/local/cuda-13.2/targets/x86_64-linux/include /src/test_parser.cpp -L/models -Wl,-rpath,/models -lyolo26person -o /models/test-parser
/models/test-parser
'''
    if args.raw_otm:
        script=script.replace('/src/parser.cpp','/src/parser_raw.cpp').replace('/src/test_parser.cpp','/src/test_parser_raw.cpp').replace('libyolo26person.so','libyolo26rawperson.so').replace('-lyolo26person','-lyolo26rawperson').replace('/models/test-parser','/models/test-parser-raw')
    subprocess.run(base+['--entrypoint','bash',platform['image'],'-c',script],check=True)
    if args.parser_only:return
    # No camera is opened during engine construction. Inference gate starts after this succeeds.
    cmd=base+['--gpus','device=0','-e','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
        '--entrypoint','/usr/src/tensorrt/bin/trtexec',platform['image'],
        '--onnx=/models/yolo26m.onnx','--saveEngine=/models/yolo26m_b6_gpu0_fp16.engine',
        '--fp16','--minShapes=images:1x3x640x640','--optShapes=images:6x3x640x640',
        '--maxShapes=images:6x3x640x640','--memPoolSize=workspace:2048',
        '--profilingVerbosity=detailed','--skipInference']
    if args.raw_otm:
        export=json.loads((out/'export_raw_otm.json').read_text())
        if export['output_shape']!=['batch',84,8400] or export['nms_nodes']!=0:
            raise RuntimeError('Unexpected raw ONNX structure')
        cmd=[a.replace('yolo26m.onnx','yolo26m_raw_otm.onnx').replace('yolo26m_b6_gpu0_fp16.engine','yolo26m_raw_otm_b6_fp16.engine').replace('images:',export['input_name']+':') for a in cmd]
    with (out/('engine_raw_otm-build.log' if args.raw_otm else 'engine-build.log')).open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    sha=lambda p:hashlib.sha256((out/p).read_bytes()).hexdigest()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    report=dict(platform=platform,gpu=gpu,batch_size=6,min_batch=1,precision='FP16',
                onnx_sha256=sha('yolo26m.onnx'),engine_sha256=sha('yolo26m_b6_gpu0_fp16.engine'),
                parser_sha256=sha('libyolo26person.so'),build_command=cmd)
    if args.raw_otm:
        report.update(onnx_sha256=sha('yolo26m_raw_otm.onnx'),engine_sha256=sha('yolo26m_raw_otm_b6_fp16.engine'),parser_sha256=sha('libyolo26rawperson.so'),head='one-to-many',cluster_mode=2)
    (out/('engine_raw_otm.json' if args.raw_otm else 'engine.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
