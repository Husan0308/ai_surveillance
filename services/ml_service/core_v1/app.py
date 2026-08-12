from __future__ import annotations
import os,time
from pathlib import Path
import yaml
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import StreamingResponse,Response
from shared.config import camera_config
from .manager import CameraManager
from .jpeg_publisher import LatestJpegPublisher
from .detector import YoloDetectorWorker
from .runtime_metrics import process_metrics

ROOT=Path(__file__).resolve().parents[3]

def _expand(value):
    if isinstance(value,str):return os.path.expandvars(value)
    if isinstance(value,list):return [_expand(v) for v in value]
    if isinstance(value,dict):return {k:_expand(v) for k,v in value.items()}
    return value

def _load_yaml(path):
    with open(path,'r',encoding='utf-8') as f:return _expand(yaml.safe_load(f) or {})

camera_cfg=camera_config().get('cameras',[])
core_cfg=_load_yaml(ROOT/'config/core_v1.yaml').get('core_v1',{})
manager=CameraManager(camera_cfg,core_cfg)
detector_cfg=dict(core_cfg.get('detector') or {})
visual_cfg=dict(core_cfg.get('visual_tracker') or {})
detector=YoloDetectorWorker(manager.stores,detector_cfg,ROOT) if bool(detector_cfg.get('enabled',False)) else None
publishers={
    cid:LatestJpegPublisher(
        cid,store,
        core_cfg.get('display_fps',12),
        core_cfg.get('jpeg_quality',82),
        core_cfg.get('max_display_width',960),
        core_cfg.get('max_display_height',540),
        detections=(detector.results if detector else None),
        overlay_max_age_ms=detector_cfg.get('overlay_max_age_ms',350),
        tracker_config=visual_cfg,
    )
    for cid,store in manager.stores.items()
}
app=FastAPI(title='AI Surveillance ML Core v1',version='1.3')

@app.on_event('startup')
def startup():
    manager.start()
    for publisher in publishers.values():publisher.start()
    if detector:detector.start()

@app.on_event('shutdown')
def shutdown():
    if detector:
        detector.stop();detector.join(10)
    for publisher in publishers.values():publisher.stop()
    for publisher in publishers.values():publisher.join()
    manager.stop()

@app.get('/health')
def health():
    metrics=manager.metrics()
    return {
        'status':'ok','mode':'camera+yolo' if detector else 'camera-only','cameras':metrics,
        'online':sum(bool(v.get('online')) for v in metrics.values()),'total':len(metrics),
        'detector':detector.metrics() if detector else None,
        'publishers':{cid:publisher.metrics() for cid,publisher in publishers.items()},
        'service_resources':process_metrics(),
    }

@app.get('/cameras')
def cameras():
    metrics=manager.metrics();return [{'id':cid,**metrics[cid]} for cid in sorted(metrics)]

@app.get('/detections')
def detections():
    if detector is None:return {'enabled':False,'cameras':{}}
    now=time.monotonic();results={}
    for cid,result in detector.results.snapshot().items():
        results[cid]={
            'frame_id':result.frame_id,
            'result_age_ms':max(0.0,(now-result.produced_monotonic)*1000.0),
            'capture_age_ms':max(0.0,(now-result.frame_captured_monotonic)*1000.0),
            'boxes':[{'bbox':[b.x1,b.y1,b.x2,b.y2],'confidence':b.confidence} for b in result.boxes],
        }
    return {'enabled':True,'cameras':results,'metrics':detector.metrics()}

@app.get('/frame/{camera_id}')
def latest_frame(camera_id:str,after:int=Query(-1),wait_ms:int=Query(200,ge=0,le=500)):
    """Return exactly one newest JPEG, optionally long-polling for a newer one."""
    if camera_id not in publishers:raise HTTPException(404,'camera not found')
    publisher=publishers[camera_id]
    jpeg,version,published,source_frame_id=publisher.wait_newer(after,wait_ms/1000.0)
    if jpeg is None:raise HTTPException(503,'frame not ready')
    headers={
        'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma':'no-cache','X-Frame-Version':str(version),'X-Source-Frame-Id':str(source_frame_id),
        'X-Published-Monotonic':f'{published:.6f}',
    }
    return Response(content=jpeg,media_type='image/jpeg',headers=headers)

def _mjpeg(camera_id):
    publisher=publishers[camera_id];last=-1
    while True:
        jpeg,version,_,_=publisher.wait_newer(last,0.5)
        if jpeg is None or version<=last:continue
        last=version
        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'

@app.get('/video/{camera_id}')
def video(camera_id:str):
    if camera_id not in publishers:raise HTTPException(404,'camera not found')
    return StreamingResponse(_mjpeg(camera_id),media_type='multipart/x-mixed-replace; boundary=frame')
