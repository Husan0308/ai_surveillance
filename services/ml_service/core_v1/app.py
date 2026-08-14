from __future__ import annotations
import os,time
from pathlib import Path
import yaml
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import StreamingResponse,Response
from shared.config import camera_config
from services.ml_service.heatmap import FloorHeatmapCoordinator
from .manager import CameraManager
from .jpeg_publisher import LatestJpegPublisher
from .unified_detector import UnifiedYoloDetectorWorker
from .reid_service import ReIDCoordinator
from .runtime_metrics import process_metrics
from .spatial_calibration import RoomSpatialMapper

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
pose_cfg=dict(core_cfg.get('pose') or {})
heatmap_cfg=dict(core_cfg.get('heatmap') or {})
reid_cfg=dict(core_cfg.get('reid') or {})
spatial_mapper=RoomSpatialMapper(ROOT/'config/room_mapping.yaml')
detector=UnifiedYoloDetectorWorker(manager.stores,detector_cfg,ROOT,pose_cfg) if bool(detector_cfg.get('enabled',False)) else None
pose=detector.pose if detector is not None and bool(pose_cfg.get('enabled',False)) else None
heatmap=FloorHeatmapCoordinator(pose,manager.stores,spatial_mapper,heatmap_cfg) if pose is not None and bool(heatmap_cfg.get('enabled',False)) else None
reid=ReIDCoordinator(manager.stores,detector.results,reid_cfg,spatial_mapper=spatial_mapper) if detector is not None and bool(reid_cfg.get('enabled',False)) else None
publishers={cid:LatestJpegPublisher(cid,store,core_cfg.get('display_fps',12),core_cfg.get('jpeg_quality',82),core_cfg.get('max_display_width',960),core_cfg.get('max_display_height',540),detections=(detector.results if detector else None),overlay_max_age_ms=detector_cfg.get('overlay_max_age_ms',350),tracker_config=visual_cfg,identity_provider=reid) for cid,store in manager.stores.items()}
app=FastAPI(title='AI Surveillance ML Core v1',version='1.8')

@app.on_event('startup')
def startup():
    manager.start();stagger=max(0.0,float(core_cfg.get('publisher_start_stagger_ms',0.0))/1000.0)
    for index,publisher in enumerate(publishers.values()):
        publisher.start()
        if stagger and index+1<len(publishers):time.sleep(stagger)
    if detector:detector.start()
    if heatmap:heatmap.start()
    if reid:reid.start()

@app.on_event('shutdown')
def shutdown():
    if reid:reid.stop();reid.join(6)
    if heatmap:heatmap.stop();heatmap.join(6)
    if detector:detector.stop();detector.join(10)
    for publisher in publishers.values():publisher.stop()
    for publisher in publishers.values():publisher.join()
    manager.stop()

@app.get('/health')
def health():
    metrics=manager.metrics()
    return {'status':'ok','mode':'camera+yolo+pose+reid' if reid and pose else ('camera+yolo+pose' if detector and pose else ('camera+yolo+reid' if reid else ('camera+yolo' if detector else 'camera-only'))),'cameras':metrics,'online':sum(bool(v.get('online')) for v in metrics.values()),'total':len(metrics),'detector':detector.metrics() if detector else None,'pose':pose.metrics() if pose else {'enabled':False},'heatmap':heatmap.snapshot() if heatmap else {'enabled':False},'reid':reid.metrics() if reid else None,'publishers':{cid:publisher.metrics() for cid,publisher in publishers.items()},'frame_history':{cid:store.history_metrics() for cid,store in manager.stores.items() if hasattr(store,'history_metrics')},'service_resources':process_metrics()}

@app.get('/cameras')
def cameras():
    metrics=manager.metrics();return [{'id':cid,**metrics[cid]} for cid in sorted(metrics)]

@app.get('/detections')
def detections():
    if detector is None:return {'enabled':False,'cameras':{}}
    now=time.monotonic();results={}
    for cid,result in detector.results.snapshot().items():results[cid]={'frame_id':result.frame_id,'result_age_ms':max(0.0,(now-result.produced_monotonic)*1000.0),'capture_age_ms':max(0.0,(now-result.frame_captured_monotonic)*1000.0),'boxes':[{'bbox':[b.x1,b.y1,b.x2,b.y2],'confidence':b.confidence} for b in result.boxes]}
    return {'enabled':True,'cameras':results,'metrics':detector.metrics()}

@app.get('/poses')
def poses_state():
    if pose is None:return {'enabled':False,'cameras':{}}
    now=time.monotonic();results={}
    for cid,result in pose.snapshot().items():
        results[cid]={
            'frame_id':result.frame_id,
            'result_age_ms':max(0.0,(now-result.produced_monotonic)*1000.0),
            'capture_age_ms':max(0.0,(now-result.frame_captured_monotonic)*1000.0),
            'people':[
                {
                    'bbox':list(person.bbox),
                    'confidence':person.confidence,
                    'keypoints':[{'x':point.x,'y':point.y,'confidence':point.confidence} for point in person.keypoints],
                }
                for person in result.people
            ],
        }
    return {'enabled':True,'cameras':results,'metrics':pose.metrics()}

@app.get('/heatmap')
def heatmap_state():
    if heatmap is None:return {'enabled':False,'rooms':{}}
    return heatmap.snapshot()

@app.get('/heatmap/{room_id}.png')
def heatmap_png(room_id:str):
    if heatmap is None:raise HTTPException(503,'heatmap is disabled')
    payload=heatmap.render_png(room_id)
    if payload is None:raise HTTPException(404,'room not found')
    return Response(content=payload,media_type='image/png',headers={'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0','Pragma':'no-cache'})

@app.post('/heatmap/reset/{room_id}')
def reset_heatmap(room_id:str):
    if heatmap is None:raise HTTPException(503,'heatmap is disabled')
    try:heatmap.reset(room_id)
    except ValueError as exc:raise HTTPException(404,str(exc)) from exc
    return {'ok':True,'room_id':room_id}

@app.get('/reid')
def reid_state():
    if reid is None:return {'enabled':False}
    return {'enabled':True,'state':reid.snapshot(),'metrics':reid.metrics()}

@app.get('/room-mapping')
def room_mapping():
    payload=spatial_mapper.snapshot()
    payload['people']=reid.room_people() if reid is not None else []
    payload['heatmap']=heatmap.snapshot().get('rooms',{}) if heatmap is not None else {}
    return payload

@app.post('/room-mapping/calibrate')
def calibrate_room_camera(payload:dict):
    try:
        return {'ok':True,'calibration':spatial_mapper.calibrate(
            payload.get('camera_id'),
            payload.get('image_points') or [],
            payload.get('room_points') or [],
            payload.get('image_size'),
            method='assisted',
        )}
    except (TypeError,ValueError) as exc:
        raise HTTPException(400,str(exc)) from exc

@app.post('/room-mapping/reset/{camera_id}')
def reset_room_camera(camera_id:str):
    try:return {'ok':True,'calibration':spatial_mapper.clear_calibration(camera_id)}
    except ValueError as exc:raise HTTPException(404,str(exc)) from exc

@app.post('/room-mapping/auto-discovery')
def automatic_room_pair(payload:dict):
    left=str(payload.get('left_camera') or '');right=str(payload.get('right_camera') or '')
    if (left,right) not in spatial_mapper.camera_pairs() and (right,left) not in spatial_mapper.camera_pairs():
        raise HTTPException(400,'camera pair is not a verified same-room pair')
    left_frame=manager.stores[left].get()[0] if left in manager.stores else None
    right_frame=manager.stores[right].get()[0] if right in manager.stores else None
    evidence=spatial_mapper.automatic_pair_evidence(
        getattr(left_frame,'image',None),getattr(right_frame,'image',None)
    )
    evidence.update({'left_camera':left,'right_camera':right,'persisted_as_floor_calibration':False})
    return evidence

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
