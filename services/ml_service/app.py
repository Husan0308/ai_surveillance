"""Camera ingestion, batched person detection, and local per-camera tracking."""
import argparse, signal, threading, time,uuid
import numpy as np
from services.ml_service.cameras.config import load_camera_configs
from services.ml_service.cameras.manager import CameraManager
from services.ml_service.pipeline.metrics import format_metrics
from services.ml_service.pipeline.scheduler import BatchScheduler
from services.ml_service.tracking.tracker_manager import TrackerManager
from services.ml_service.tracking.appearance import AppearanceExtractor
from services.ml_service.identity.reid_model import OSNetReIDModel
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.schemas import IdentityTrackObservation
from services.ml_service.identity.identity_resolver import IdentityResolver
from services.ml_service.face.insightface_engine import InsightFaceEngine
from services.ml_service.face.face_detector import FaceDetector
from services.ml_service.face.face_manager import FaceManager
from services.ml_service.face.gallery import KnownPersonGallery
from services.ml_service.face.matcher import KnownPersonMatcher
from services.ml_service.face.quality import FaceQualityScorer
from services.ml_service.face.schemas import FaceCandidate
from services.ml_service.heatmap import HeatmapManager
from shared.config import project_config
from shared.logging import configure_logging, get_logger
from shared.settings import ServiceSettings
from services.ml_service.messaging import MLMessageBridge
from services.ml_service.control import runtime as control_runtime
CHANNELS={"enrollment":"enrollment","heatmaps":"heatmaps","status":"status","metrics":"metrics","events":"events"}

def _project_config():
    return project_config()

def run(diagnostic=False, synthetic=False, duration=0.0):
    configure_logging(service="ml-service"); log = get_logger(__name__)
    config = _project_config(); max_age = float(config.get("ai", {}).get("max_frame_age_ms", 250))
    detector, state = None, {"detections": None, "tracks": None, "cycle_metrics":{}}
    reid_model=None;appearance=None;tracker_config=config
    reid_cfg=config.get("ai",{}).get("reid",{})
    if not diagnostic and bool(reid_cfg.get("enabled",True)):
        reid_model=OSNetReIDModel(config)
        appearance=AppearanceExtractor(reid_model,reid_cfg.get("device","cpu"),reid_cfg.get("batch_size",32))
        tracker_config=dict(config);tracker_config["tracking"]={**config.get("ai",{}).get("tracker",{}),"appearance_enabled":True,
            "appearance_device":reid_cfg.get("device","cpu"),"appearance_batch_size":reid_cfg.get("batch_size",32)}
    tracker_manager = TrackerManager(tracker_config,appearance)
    identity_manager = GlobalIdentityManager(config)
    heatmap_manager = HeatmapManager(config)
    bridge=MLMessageBridge();bridge.start();enrollments={};runtime_cameras={};last_heatmap_publish={}
    face_manager=None;face_engine=None
    if not diagnostic and bool(config.get("face",{}).get("enabled",True)):
        face_engine=InsightFaceEngine(config);face_engine.warmup()
        gallery=KnownPersonGallery(max_embeddings=int(config.get("face",{}).get("max_face_embeddings_per_person",20)))
        face_cfg=config.get("face",{})
        matcher=KnownPersonMatcher(gallery,float(face_cfg.get("match_threshold",.55)),float(face_cfg.get("strong_match_threshold",.75)),float(face_cfg.get("ambiguity_margin",.05)))
        resolver=IdentityResolver(identity_manager.store)
        face_manager=FaceManager(FaceDetector(face_engine),FaceQualityScorer(float(face_cfg.get("min_quality",.55))),matcher,resolver)
    if not diagnostic:
        from services.ml_service.detection.person_detector import PersonDetector
        detector = PersonDetector(config, max_frame_age_ms=max_age)
    def consume(batch):
        if detector is None: return
        cycle_started=time.perf_counter()
        for packet in batch.frames:control_runtime.frame(packet)
        for command in bridge.poll():
            kind=command.get("type")
            if kind=="enrollment.start":
                enrollments[command["session_id"]]={**command,"person_id":str(uuid.uuid4()),"samples":[],"last":0}
                bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.started","session_id":command["session_id"],"captured":0,"required":6})
            elif kind=="enrollment.cancel":
                enrollments.pop(command.get("session_id"),None);bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.cancelled","session_id":command.get("session_id")})
            elif kind=="settings.ml.updated" and detector is not None:
                settings=command.get("settings",{})
                if settings.get("detection_confidence") is not None:detector.backend.conf=float(settings["detection_confidence"])
                if settings.get("heatmap_enabled") is not None:heatmap_manager.enabled=bool(settings["heatmap_enabled"])
            elif kind=="camera.config.changed":
                camera_id=command.get("camera_id");action=command.get("action");old_ids=set(runtime_cameras)
                if action=="deleted":runtime_cameras.pop(camera_id,None)
                else:
                    item=dict(command.get("config",{}));item["online"]=item.pop("enabled",item.get("online",True));runtime_cameras[camera_id]=item
                manager.configure(runtime_cameras.values());new_buffers=manager.buffers()
                for removed in old_ids-set(new_buffers):scheduler.unregister_camera(removed)
                for added in set(new_buffers)-old_ids:scheduler.register_camera(added,new_buffers[added])
        phase=time.perf_counter();detections = detector.process_batch(batch);detector_ms=(time.perf_counter()-phase)*1000
        frames = {packet.camera_id: packet.frame for packet in batch.frames}
        state["detections"] = detections
        phase=time.perf_counter();state["tracks"] = tracker_manager.update_batch(detections, frames);tracking_ms=(time.perf_counter()-phase)*1000
        identity_manager.metrics.values.reid_batch_size=tracker_manager.reid_batch_size
        identity_manager.metrics.values.reid_extract_ms=tracker_manager.reid_extract_ms
        phase=time.perf_counter();identities=[]
        for camera in state["tracks"].results:
            for track in camera.tracks:
                if track.misses or track.state.value!="CONFIRMED": continue
                embedding=tracker_manager.embedding_for(camera.camera_id,track.track_id)
                observation=IdentityTrackObservation(camera.camera_id,camera.frame_id,track.track_id,
                    track.bbox,track.confidence,camera.receive_timestamp,embedding,track.confidence)
                identities.append(identity_manager.update(observation))
        state["identities"]=tuple(identities)
        identity_ms=(time.perf_counter()-phase)*1000;phase=time.perf_counter()
        for result in identities:
            packet=next((p for p in batch.frames if p.camera_id==result.camera_id),None)
            bridge.publish(CHANNELS["events"],{"type":"frame.metadata","camera_id":result.camera_id,"frame_id":result.frame_id,"timestamp":packet.receive_timestamp if packet else time.time(),"tracks":[{"bbox":list(item.bbox),"confidence":item.confidence,"local_track_id":item.local_track_id,"global_id":item.global_id,"person_id":getattr(item,"person_id",None),"display_name":getattr(item,"display_name",None)} for item in result.tracks]})
        if face_manager is not None:
            face_results=[]
            for identity_result in identities:
                item=identity_result.tracks[0]
                if item.global_id is None:continue
                frame=frames.get(identity_result.camera_id)
                if frame is None:continue
                candidate=FaceCandidate(identity_result.camera_id,identity_result.frame_id,item.local_track_id,item.global_id,item.bbox,item.confidence,time.time())
                result=face_manager.process(candidate,frame)
                if result is not None:face_results.append(result)
            state["faces"]=tuple(face_results)
        face_ms=(time.perf_counter()-phase)*1000;phase=time.perf_counter()
        by_camera={}
        for identity_result in identities:
            by_camera.setdefault(identity_result.camera_id,[]).extend(identity_result.tracks)
        for packet in batch.frames:
            heatmap_manager.update(packet.camera_id,packet.frame_id,by_camera.get(packet.camera_id,()),packet.width,packet.height,packet.receive_timestamp)
        state["heatmaps"]={camera_id:heatmap_manager.snapshot(camera_id) for camera_id in frames}
        heatmap_ms=(time.perf_counter()-phase)*1000;phase=time.perf_counter()
        for camera_id,snapshot in state["heatmaps"].items():
            if time.time()-last_heatmap_publish.get(camera_id,0)<1:continue
            last_heatmap_publish[camera_id]=time.time();bridge.publish(CHANNELS["heatmaps"],{"type":"heatmap.updated","camera_id":camera_id,"mode":snapshot.mode.value,"timestamp":snapshot.timestamp,"grid_width":snapshot.grid_width,"grid_height":snapshot.grid_height,"max_value":snapshot.max_value,"values":snapshot.values.tolist()})
        if face_engine is not None:
            for session_id,session in list(enrollments.items()):
                frame=frames.get(session["camera_id"]);now=time.time()
                if frame is None or now-session["last"]<.5:continue
                session["last"]=now;faces=face_engine.detect(frame,need_embedding=True)
                if not faces:
                    bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.progress","session_id":session_id,"captured":len(session["samples"]),"required":6,"quality":0,"message":"No face detected"});continue
                face=max(faces,key=lambda item:float(item.get("det_score",0)));embedding=face.get("embedding");quality=float(face.get("det_score",0))
                if embedding is None or quality<.55:
                    bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.progress","session_id":session_id,"captured":len(session["samples"]),"required":6,"quality":quality,"message":"Face quality too low"});continue
                value=np.asarray(embedding,np.float32);value/=max(float(np.linalg.norm(value)),1e-12);session["samples"].append(value)
                bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.progress","session_id":session_id,"captured":len(session["samples"]),"required":6,"quality":quality})
                if len(session["samples"])>=6:
                    gallery.add(session["person_id"],session["name"],session["samples"])
                    centroid=np.mean(session["samples"],axis=0);centroid/=max(float(np.linalg.norm(centroid)),1e-12)
                    bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.completed","session_id":session_id,"person_id":session["person_id"],"name":session["name"],"department":session.get("department"),"quality":quality,"embedding":centroid.tolist()});enrollments.pop(session_id,None)
        state["cycle_metrics"]={"detector_ms":detector_ms,"tracking_ms":tracking_ms,"reid_ms":tracker_manager.reid_extract_ms,"identity_ms":identity_ms,"face_ms":face_ms,"heatmap_ms":heatmap_ms,"publishing_ms":(time.perf_counter()-phase)*1000,"batch_cycle_ms":(time.perf_counter()-cycle_started)*1000,"total_latency_ms":max((time.time()-packet.capture_timestamp)*1000 for packet in batch.frames)}
    scheduler = BatchScheduler(max_age, on_batch=consume)
    capture_factory = None
    if synthetic:
        from services.ml_service.diagnostic import synthetic_capture_factory
        capture_factory = synthetic_capture_factory
    manager = CameraManager(scheduler.notify_frame_available, capture_factory)
    cameras = load_camera_configs();runtime_cameras.update({item["id"]:dict(item) for item in cameras});manager.configure(cameras)
    for camera_id, buffer in manager.buffers().items(): scheduler.register_camera(camera_id, buffer)
    stop = threading.Event()
    def request_stop(*_args): stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    log.info("ML Service starting: cameras=%d max_frame_age_ms=%.0f diagnostic=%s", len(cameras), max_age, diagnostic)
    control_runtime.status={"status":"running","cameras":len(cameras)};bridge.publish(CHANNELS["status"],{"type":"system.status","status":"started","cameras":len(cameras)})
    scheduler.start(); manager.start(); started = time.monotonic(); next_report = started + 2
    try:
        while not stop.wait(.25):
            now = time.monotonic()
            if now >= next_report:
                pipeline_metrics=scheduler.snapshot_metrics(manager.metrics());pipeline_metrics["cycle"]=state["cycle_metrics"];control_runtime.metrics=pipeline_metrics;log.info("\n%s\nCYCLE %s", format_metrics(pipeline_metrics),state["cycle_metrics"]);bridge.publish(CHANNELS["metrics"],{"type":"system.metrics",**pipeline_metrics})
                if detector is not None:
                    log.info("\n%s\n%s\n%s\nHEATMAP %s", detector.metrics.format_compact(), tracker_manager.metrics.format_compact(),identity_manager.metrics.format_compact(),heatmap_manager.metrics.snapshot())
                next_report = now + 2
            if duration and now - started >= duration: break
    finally:
        log.info("Stopping cameras, scheduler, detector and trackers")
        manager.shutdown(); scheduler.stop(); scheduler.join(6)
        if detector is not None: detector.close()
        if face_engine is not None:face_engine.close()
        if reid_model is not None:reid_model.close()
        control_runtime.status={"status":"stopped"};bridge.publish(CHANNELS["status"],{"type":"system.status","status":"stopped"});bridge.close()
        tracker_manager.close();identity_manager.close();log.info("ML Service stopped cleanly")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true", help="run camera/batching without YOLO")
    parser.add_argument("--synthetic", action="store_true", help="use configured IDs with fake sources")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args(); run(args.diagnostic, args.synthetic, args.duration)

if __name__ == "__main__": main()
