from __future__ import annotations
import os,time
from pathlib import Path
import yaml
from fastapi import FastAPI,HTTPException
from fastapi.responses import StreamingResponse
from shared.config import camera_config
from .manager import CameraManager
from .jpeg_publisher import LatestJpegPublisher

ROOT=Path(__file__).resolve().parents[3]

def _expand(value):
    if isinstance(value,str):return os.path.expandvars(value)
    if isinstance(value,list):return [_expand(v) for v in value]
    if isinstance(value,dict):return {k:_expand(v) for k,v in value.items()}
    return value

def _load_yaml(path):
    with open(path,'r',encoding='utf-8') as f:return _expand(yaml.safe_load(f) or {})

# IMPORTANT: use the same canonical camera configuration layer as the normal
# services. It expands ${ENV} placeholders and, crucially, overlays the local
# untracked config/cameras.local.yaml file. The first core-v1 version read
# cameras.yaml directly, so local RTSP credentials/source overrides were lost
# and every rtspsrc connection returned no samples.
camera_cfg=camera_config().get('cameras',[])
core_cfg=_load_yaml(ROOT/'config/core_v1.yaml').get('core_v1',{})
manager=CameraManager(camera_cfg)
publishers={cid:LatestJpegPublisher(cid,store,core_cfg.get('display_fps',12),core_cfg.get('jpeg_quality',82),core_cfg.get('max_display_width',960),core_cfg.get('max_display_height',540)) for cid,store in manager.stores.items()}
app=FastAPI(title='AI Surveillance ML Core v1',version='1.0')

@app.on_event('startup')
def startup():
    manager.start()
    for publisher in publishers.values():publisher.start()

@app.on_event('shutdown')
def shutdown():
    for publisher in publishers.values():publisher.stop()
    for publisher in publishers.values():publisher.join()
    manager.stop()

@app.get('/health')
def health():
    metrics=manager.metrics();return {'status':'ok','mode':'camera-only','cameras':metrics,'online':sum(bool(v.get('online')) for v in metrics.values()),'total':len(metrics)}

@app.get('/cameras')
def cameras():
    metrics=manager.metrics();return [{'id':cid,**metrics[cid]} for cid in sorted(metrics)]

def _mjpeg(camera_id):
    publisher=publishers[camera_id];last=-1
    while True:
        jpeg,version=publisher.latest()
        if jpeg is None or version<=last:
            time.sleep(.005);continue
        last=version
        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'

@app.get('/video/{camera_id}')
def video(camera_id:str):
    if camera_id not in publishers:raise HTTPException(404,'camera not found')
    return StreamingResponse(_mjpeg(camera_id),media_type='multipart/x-mixed-replace; boundary=frame')
