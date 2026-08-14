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
from .reid_service import ReIDCoordinator
from .room_sessions import RoomVisitSessionManager
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
reid_cfg=dict(core_cfg.get('reid') or {})
room_sessions_cfg=dict(core_cfg.get('room_sessions') or {})
detector=YoloDetectorWorker(manager.stores,detector_cfg,ROOT) if bool(detector_cfg.get('enabled',False)) else None
reid=ReIDCoordinator(manager.stores,detector.results,reid_cfg) if detector is not None and bool(reid_cfg.get('enabled',False)) else None
camera_rooms=dict((reid_cfg.get('identity') or {}).get('camera_rooms') or {})
room_sessions=RoomVisitSessionManager(room_sessions_cfg,camera_rooms=camera_rooms) if reid is not None else None
publishers={cid:LatestJpegPublisher(cid,store,core_cfg.get('display_fps',12),core_cfg.get('jpeg_quality',82),core_cfg.get('max_display_width',960),core_cfg.get('max_display_height',540),detections=(detector.results if detector else None),overlay_max_age_ms=detector_cfg.get('overlay_max_age_ms',350),tracker_config=visual_cfg,identity_provider=reid) for cid,store in manager.stores.items()}
app=FastAPI(title='AI Surveillance ML Core v1',version='1.6')

@app.on_event('startup')
def startup():
    manager.start();stagger=max(0.0,float(core_cfg.get('publisher_start_stagger_ms',0.0))/1000.0)
    for index,publisher in enumerate(publishers.values()):
        publisher.start()
        if stagger and index+1<len(publishers):time.sleep(stagger)
    if detector:detector.start()
    if reid:reid.start()

@app.on_event('shutdown')
def shutdown():
    if reid:reid.stop();reid.join(6)
    if detector:detector.stop();detector.join(10)
    for publisher in publishers.values():publisher.stop()
    for publisher in publishers.values():publisher.join()
    manager.stop()

@app.get('/health')
def health():
    metrics=manager.metrics()
    return {'status':'ok','mode':'camera+yolo+reid' if reid else ('camera+yolo' if detector else 'camera-only'),'cameras':metrics,'online':sum(bool(v.get('online')) for v in metrics.values()),'total':len(metrics),'detector':detector.metrics() if detector else None,'reid':reid.metrics() if reid else None,'room_sessions':(room_sessions.snapshot().get('metrics') if room_sessions else None),'publishers':{cid:publisher.metrics() for cid,publisher in publishers.items()},'frame_history':{cid:store.history_metrics() for cid,store in manager.stores.items() if hasattr(store,'history_metrics')},'service_resources':process_metrics()}

@app.get('/cameras')
def cameras():
    metrics=manager.metrics();return [{'id':cid,**metrics[cid]} for cid in sorted(metrics)]

@app.get('/detections')
def detections():
    if detector is None:return {'enabled':False,'cameras':{}}
    now=time.monotonic();results={}
    for cid,result in detector.results.snapshot().items():results[cid]={'frame_id':result.frame_id,'result_age_ms':max(0.0,(now-result.produced_monotonic)*1000.0),'capture_age_ms':max(0.0,(now-result.frame_captured_monotonic)*1000.0),'boxes':[{'bbox':[b.x1,b.y1,b.x2,b.y2],'confidence':b.confidence} for b in result.boxes]}
    return {'enabled':True,'cameras':results,'metrics':detector.metrics()}

@app.get('/reid')
def reid_state():
    if reid is None:return {'enabled':False}
    state=reid.snapshot()
    if room_sessions:room_sessions.update(state)
    return {'enabled':True,'state':state,'metrics':reid.metrics()}

@app.get('/room-sessions')
def room_session_state():
    if reid is None or room_sessions is None:return {'enabled':False,'active_sessions':[],'recent_sessions':[],'events':[]}
    room_sessions.update(reid.snapshot())
    return room_sessions.snapshot()

@app.get('/frame/{camera_id}')
def latest_frame(camera_id:str,after:int=Query(-1),wait_ms:int=Query(200,ge=0,le=500)):
    if camera_id not in publishers:raise HTTPException(404,'camera not found')
    publisher=publishers[camera_id];jpeg,version,published,source_frame_id=publisher.wait_newer(after,wait_ms/1000.0)
    if jpeg is None:raise HTTPException(503,'frame not ready')
    headers={'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0','Pragma':'no-cache','X-Frame-Version':str(version),'X-Source-Frame-Id':str(source_frame_id),'X-Published-Monotonic':f'{published:.6f}'}
    return Response(content=jpeg,media_type='image/jpeg',headers=headers)

def _mjpeg(camera_id):
    publisher=publishers[camera_id];last=-1
    while True:
        jpeg,version,_,_=publisher.wait_newer(last,0.5)
        if jpeg is None or version<=last:continue
        last=version;yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'

@app.get('/video/{camera_id}')
def video(camera_id:str):
    if camera_id not in publishers:raise HTTPException(404,'camera not found')
    return StreamingResponse(_mjpeg(camera_id),media_type='multipart/x-mixed-replace; boundary=frame')
