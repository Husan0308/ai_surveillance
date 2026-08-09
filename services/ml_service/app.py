"""Camera ingestion, batched person detection, and local per-camera tracking."""
import argparse,json,signal,threading,time,uuid
from pathlib import Path
import numpy as np
from services.ml_service.cameras.config import fetch_camera_configs,load_camera_configs
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
from services.ml_service.face.gallery import KnownPersonGallery,SQLiteGalleryRepository
from services.ml_service.face.matcher import KnownPersonMatcher
from services.ml_service.face.quality import FaceQualityScorer
from services.ml_service.face.image_enrollment import validate_enrollment_image
from services.ml_service.face.schemas import FaceCandidate
from services.ml_service.heatmap import HeatmapManager
from shared.config import project_config,camera_config,topology_config
from shared.topology import compile_topology
from shared.logging import configure_logging, get_logger
from shared.settings import ServiceSettings
from services.ml_service.messaging import MLMessageBridge
from services.ml_service.control import runtime as control_runtime
from services.ml_service.events import frame_metadata_messages
from services.ml_service.runtime_control import RuntimeCommandLoop
from services.ml_service.secondary import SecondaryAIScheduler,SecondaryTask,SecondaryTaskType
from services.ml_service.tracking.appearance import crop_detection
from services.ml_service.metrics.timing import TimingProfile
from services.ml_service.metrics.system_sampler import SystemMetricsSampler
from services.ml_service.snapshots import UnknownSnapshotManager
CHANNELS={"enrollment":"enrollment","heatmaps":"heatmaps","status":"status","metrics":"metrics","events":"events"}

def _project_config():
    config=dict(project_config());identity=dict(config.get("identity",{}));ids=[str(item["id"]) for item in camera_config().get("cameras",[]) if item.get("id")]
    identity["topology"]=compile_topology(topology_config(),ids);config["identity"]=identity;return config

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
    snapshot_cfg=config.get("identity",{}).get("unknown_snapshots",{});snapshot_manager=UnknownSnapshotManager(Path(__file__).resolve().parents[2]/"data"/"snapshots",snapshot_cfg.get("max_identities",500),snapshot_cfg.get("retention_days",7),snapshot_cfg.get("min_improvement",.10))
    bridge=MLMessageBridge();bridge.start();enrollments={};enrollment_lock=threading.RLock();runtime_cameras={};last_heatmap_publish={};camera_event_states={};person_assignments={};conflict_keys=set();event_state_lock=threading.RLock()
    reid_requested={};face_requested={};secondary=None;fast_profile=TimingProfile();system_sampler=SystemMetricsSampler()
    face_manager=None;face_engine=None;gallery=None;quality_scorer=None
    if not diagnostic and bool(config.get("face",{}).get("enabled",True)):
        face_engine=InsightFaceEngine(config);face_engine.warmup()
        gallery=KnownPersonGallery(SQLiteGalleryRepository(ServiceSettings.from_env().database_path),max_embeddings=int(config.get("face",{}).get("max_face_embeddings_per_person",20)))
        face_cfg=config.get("face",{});quality_scorer=FaceQualityScorer(float(face_cfg.get("min_quality",.55)))
        matcher=KnownPersonMatcher(gallery,float(face_cfg.get("match_threshold",.55)),float(face_cfg.get("strong_match_threshold",.75)),float(face_cfg.get("ambiguity_margin",.05)))
        resolver=IdentityResolver(identity_manager.store)
        face_manager=FaceManager(FaceDetector(face_engine),quality_scorer,matcher,resolver)
    if not diagnostic:
        from services.ml_service.detection.person_detector import PersonDetector
        detector = PersonDetector(config, max_frame_age_ms=max_age)
    def secondary_result(task,result):
        if task.task_type == SecondaryTaskType.REID:
            tracker_manager.set_embedding(task.camera_id,task.local_track_id,result)
        elif task.task_type == SecondaryTaskType.FACE:
            state.setdefault("faces",{})[(task.camera_id,task.local_track_id)]=result
            if result is None:return
            with event_state_lock:
                previous=person_assignments.get(result.global_id)
                if result.person_id and previous!=result.person_id:
                    person_assignments[result.global_id]=result.person_id;bridge.publish(CHANNELS["events"],{"type":"person.identified","camera_id":task.camera_id,"track_id":task.local_track_id,"global_id":result.global_id,"person_id":result.person_id,"name":result.display_name,"confidence":result.identity_confidence,"timestamp":time.time()})
                if result.identity_conflict:
                    key=(result.global_id,result.person_id,round(float(result.face_similarity),3))
                    if key not in conflict_keys:
                        if len(conflict_keys)>=1024:conflict_keys.clear()
                        conflict_keys.add(key);bridge.publish(CHANNELS["events"],{"type":"identity.conflict","camera_id":task.camera_id,"track_id":task.local_track_id,"global_id":result.global_id,"person_id":result.person_id,"similarity":result.face_similarity,"timestamp":time.time()})

    def reid_processor(tasks):
        embeddings,timing=appearance.extract_batch([task.crop for task in tasks])
        tracker_manager.reid_batch_size=len(tasks);tracker_manager.reid_extract_ms=float(timing.get("total_ms",timing.get("gpu_ms",0)))
        return embeddings

    enrollment_target=int(config.get("enrollment",{}).get("images_count",10))
    def face_processor(task):
        if isinstance(task.context,dict) and task.context.get("kind")=="enrollment":
            session_id=task.context["session_id"]
            with enrollment_lock:session=enrollments.get(session_id)
            if session is None:return None
            result=validate_enrollment_image(task.context["path"],face_engine,quality_scorer,int(config.get("enrollment",{}).get("min_face_size_px",30)),float(config.get("enrollment",{}).get("min_blur_variance",40)))
            completed=None
            with enrollment_lock:
                session=enrollments.get(session_id)
                if session is None:return None
                session["processed"]+=1
                if result["accepted"]:session["samples"].append(result)
                else:session["rejections"].append(result["reason"])
                valid=len(session["samples"]);processed=session["processed"];total=session["total"]
                if valid>=enrollment_target:
                    best=sorted(session["samples"],key=lambda item:item["quality"],reverse=True)[:enrollment_target]
                    gallery.add(session["person_id"],session["name"],[item["embedding"] for item in best])
                    completed={"type":"enrollment.completed","session_id":session_id,"person_id":session["person_id"],"name":session["name"],"department":session.get("department"),"quality":sum(item["quality"] for item in best)/len(best),"dimension":512,"model_version":"buffalo_l:w600k_r50","embeddings":[{"embedding":np.asarray(item["embedding"],np.float32).tolist(),"quality":item["quality"],"source_metadata":{"filename":__import__("pathlib").Path(item["source"]).name,"blur":item["blur"]}} for item in best]}
                    enrollments.pop(session_id,None)
                elif processed>=total:
                    completed={"type":"enrollment.failed","session_id":session_id,"captured":valid,"required":enrollment_target,"message":f"Only {valid}/{enrollment_target} valid images; rejected: {len(session["rejections"])}","rejections":session["rejections"]}
                    enrollments.pop(session_id,None)
            if completed:bridge.publish(CHANNELS["enrollment"],completed)
            else:bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.progress","session_id":session_id,"captured":valid,"required":enrollment_target,"quality":float(result.get("quality",0)),"message":None if result["accepted"] else result["reason"],"rejected":len(session["rejections"])})
            return None
        return face_manager.process(task.context,task.crop)

    processors={}
    if appearance is not None and appearance.available:processors[SecondaryTaskType.REID]=reid_processor
    if face_manager is not None:processors[SecondaryTaskType.FACE]=face_processor
    secondary_cfg=config.get("ai",{}).get("secondary",{})
    secondary=SecondaryAIScheduler(processors,secondary_result,secondary_cfg.get("queue_size",36),
        secondary_cfg.get("max_task_age_ms",1000),reid_cfg.get("batch_size",6),reid_cfg.get("batch_wait_ms",5))

    def consume(batch):
        if detector is None: return
        cycle_started=time.perf_counter();inference_started=time.time()
        phase=time.perf_counter()
        video_handoff_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();detections = detector.process_batch(batch);detector_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();frames = {packet.camera_id: packet.frame for packet in batch.frames}
        packets = {packet.camera_id: packet for packet in batch.frames}
        tracker_input_ms=(time.perf_counter()-phase)*1000
        state["detections"] = detections
        state["detector_inputs"]={item.camera_id:1 for item in detections.results};state["detection_counts"]={item.camera_id:len(item.detections) for item in detections.results}
        phase=time.perf_counter();state["tracks"] = tracker_manager.update_batch(detections);tracking_ms=(time.perf_counter()-phase)*1000
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
        for identity_result in identities:
            frame=frames.get(identity_result.camera_id)
            if frame is None:continue
            for item in identity_result.tracks:
                if item.global_id and not item.person_id:snapshot_manager.submit(item.global_id,identity_result.camera_id,identity_result.frame_id,packets[identity_result.camera_id].capture_timestamp,frame,item.bbox)
        identity_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();metadata_messages=frame_metadata_messages(batch.frames,identities);metadata_prepare_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        if metadata_messages:bridge.publish(CHANNELS["events"],{"type":"frame.metadata.batch","messages":metadata_messages})
        metadata_publish_ms=(time.perf_counter()-phase)*1000
        metadata_published=time.time()
        phase=time.perf_counter();reid_crop_ms=face_crop_ms=enrollment_copy_ms=0.0
        now=time.time();refresh=float(reid_cfg.get("embedding_refresh_seconds",10));cooldown=float(reid_cfg.get("event_cooldown_seconds",5))
        if SecondaryTaskType.REID in processors:
            for camera in state["tracks"].results:
                frame=frames.get(camera.camera_id);packet=packets.get(camera.camera_id)
                if frame is None or packet is None:continue
                for track in camera.tracks:
                    if track.misses or track.state.value!="CONFIRMED":continue
                    embedding=tracker_manager.embedding_for(camera.camera_id,track.track_id);last=reid_requested.get((camera.camera_id,track.track_id),0)
                    due=embedding is None and now-last>=cooldown or embedding is not None and now-last>=refresh
                    if not due:
                        if embedding is not None:secondary.cache_hit()
                        continue
                    crop_started=time.perf_counter();crop=crop_detection(frame,track.bbox)
                    if crop is None or not crop.size:continue
                    owned=crop.copy();reid_crop_ms+=(time.perf_counter()-crop_started)*1000
                    if secondary.submit(SecondaryTask(SecondaryTaskType.REID,camera.camera_id,track.track_id,None,camera.frame_id,packet.capture_timestamp,track.bbox,owned)):reid_requested[(camera.camera_id,track.track_id)]=now
        if SecondaryTaskType.FACE in processors:
            face_refresh=float(config.get("face",{}).get("refresh_seconds",60));face_retry=float(config.get("face",{}).get("weak_retry_seconds",3))
            for identity_result in identities:
                item=identity_result.tracks[0]
                if item.global_id is None:continue
                frame=frames.get(identity_result.camera_id);packet=packets.get(identity_result.camera_id);face_key=(identity_result.camera_id,item.local_track_id);last=face_requested.get(face_key,0)
                if frame is None or packet is None or now-last<(face_refresh if face_key in state.get("faces",{}) else face_retry):continue
                crop_started=time.perf_counter();crop=crop_detection(frame,item.bbox)
                if crop is None or crop.shape[0]<int(config.get("face",{}).get("min_face_size",30)):continue
                owned=crop.copy();face_crop_ms+=(time.perf_counter()-crop_started)*1000
                h,w=owned.shape[:2];candidate=FaceCandidate(identity_result.camera_id,identity_result.frame_id,item.local_track_id,item.global_id,(0,0,w,h),item.confidence,packet.capture_timestamp)
                if secondary.submit(SecondaryTask(SecondaryTaskType.FACE,identity_result.camera_id,item.local_track_id,item.global_id,identity_result.frame_id,packet.capture_timestamp,item.bbox,owned,context=candidate)):face_requested[(identity_result.camera_id,item.local_track_id)]=now
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
        completed=time.time();publishing_ms=(time.perf_counter()-phase)*1000
        profile_values={"capture_to_scheduler":max((packet.scheduler_selected_timestamp-packet.capture_timestamp)*1000 for packet in batch.frames),"scheduler_wait":max((inference_started-packet.scheduler_selected_timestamp)*1000 for packet in batch.frames),"video_handoff":video_handoff_ms,"pure_detector":detector_ms,"tracker_input_construction":tracker_input_ms,"tracker":tracking_ms,"identity":identity_ms,"metadata_preparation":metadata_prepare_ms,"metadata_publication":metadata_publish_ms,"reid_crop_preparation":reid_crop_ms,"face_crop_preparation":face_crop_ms,"enrollment_frame_copy":enrollment_copy_ms,"secondary_submission":face_ms,"heatmap":heatmap_ms,"message_publishing":publishing_ms,"full_fast_path":(time.perf_counter()-cycle_started)*1000,"capture_to_metadata":max((metadata_published-packet.capture_timestamp)*1000 for packet in batch.frames)};fast_profile.record(profile_values)
        state["cycle_metrics"]={**profile_values,"inference_start":inference_started,"inference_end":detections.completed_at,"metadata_publish_timestamp":metadata_published}
    scheduler = BatchScheduler(max_age, on_batch=consume)
    capture_factory = None
    if synthetic:
        from services.ml_service.diagnostic import synthetic_capture_factory
        capture_factory = synthetic_capture_factory
    manager = CameraManager(scheduler.notify_frame_available, capture_factory, control_runtime.frame)
    camera_authority_ready=True
    try:cameras=fetch_camera_configs()
    except Exception:
        cameras=load_camera_configs();camera_authority_ready=False
    runtime_cameras.update({item["id"]:dict(item) for item in cameras});manager.configure(cameras)
    for camera_id, buffer in manager.buffers().items(): scheduler.register_camera(camera_id, buffer)
    def handle_command(command):
        kind=command.get("type")
        if kind=="enrollment.start":
            paths=list(command.get("sample_paths",()))
            session={**command,"person_id":str(uuid.uuid4()),"samples":[],"rejections":[],"processed":0,"total":len(paths)}
            with enrollment_lock:enrollments[command["session_id"]]=session
            bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.started","session_id":command["session_id"],"captured":0,"required":enrollment_target})
            for index,path in enumerate(paths):
                accepted=secondary.submit(SecondaryTask(SecondaryTaskType.FACE,"FILE",f"ENROLL-{index}",None,index,time.time(),(0,0,0,0),path,priority=10,context={"kind":"enrollment","session_id":command["session_id"],"path":path}))
                if not accepted:
                    with enrollment_lock:session["processed"]+=1;session["rejections"].append("queue_full")
        elif kind=="enrollment.cancel":
            with enrollment_lock:enrollments.pop(command.get("session_id"),None)
            bridge.publish(CHANNELS["enrollment"],{"type":"enrollment.cancelled","session_id":command.get("session_id")})
        elif kind=="gallery.person.updated" and gallery is not None:
            gallery.update_name(command.get("person_id"),command.get("name","Unknown"))
        elif kind=="gallery.person.deleted" and gallery is not None:
            gallery.remove(command.get("person_id"))
            identity_manager.person_deleted(command.get("person_id")) if hasattr(identity_manager,"person_deleted") else None
        elif kind=="settings.ml.updated" and detector is not None:
            settings=command.get("settings",{})
            if settings.get("detection_confidence") is not None:detector.backend.conf=float(settings["detection_confidence"])
            if settings.get("heatmap_enabled") is not None:heatmap_manager.enabled=bool(settings["heatmap_enabled"])
        elif kind=="camera.config.changed":
            camera_id=command.get("camera_id");action=command.get("action");old_buffers=manager.buffers();old_ids=set(old_buffers)
            if action=="deleted":runtime_cameras.pop(camera_id,None)
            else:
                previous=runtime_cameras.get(camera_id,{});item={**previous,**dict(command.get("config",{}))}
                item["id"]=camera_id;item["source"]=item.get("rtsp_url",item.get("source"));item["online"]=item.get("enabled",item.get("online",True));runtime_cameras[camera_id]=item
            manager.configure(runtime_cameras.values());new_buffers=manager.buffers()
            for removed in old_ids-set(new_buffers):scheduler.unregister_camera(removed);tracker_manager.remove_camera(removed);identity_manager.camera_failed(removed)
            for added in set(new_buffers)-old_ids:scheduler.register_camera(added,new_buffers[added])
            for changed in old_ids & set(new_buffers):
                if old_buffers[changed] is not new_buffers[changed]:scheduler.unregister_camera(changed);scheduler.register_camera(changed,new_buffers[changed]);tracker_manager.remove_camera(changed);identity_manager.camera_failed(changed)
    command_loop=RuntimeCommandLoop(bridge.poll,handle_command)
    stop = threading.Event()
    def request_stop(*_args): stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    log.info("ML Service starting: cameras=%d max_frame_age_ms=%.0f diagnostic=%s", len(cameras), max_age, diagnostic)
    control_runtime.status={"status":"running","cameras":len(cameras),"detector_ready":detector is not None,"reid_ready":SecondaryTaskType.REID in processors,"face_ready":SecondaryTaskType.FACE in processors,"secondary_ready":True};bridge.publish(CHANNELS["status"],{"type":"system.status","status":"started","cameras":len(cameras)})
    system_sampler.start();secondary.start();command_loop.start();scheduler.start(); manager.start(); started = time.monotonic(); next_report = started + 2
    try:
        while not stop.wait(.25):
            now = time.monotonic()
            if not camera_authority_ready and bridge.publisher.available:
                try:
                    canonical=fetch_camera_configs();old_buffers=manager.buffers();old_ids=set(old_buffers);runtime_cameras.clear();runtime_cameras.update({item["id"]:dict(item) for item in canonical});manager.configure(canonical);new_buffers=manager.buffers()
                    for removed in old_ids-set(new_buffers):scheduler.unregister_camera(removed);tracker_manager.remove_camera(removed);identity_manager.camera_failed(removed)
                    for added in set(new_buffers)-old_ids:scheduler.register_camera(added,new_buffers[added])
                    for changed in old_ids & set(new_buffers):
                        if old_buffers[changed] is not new_buffers[changed]:scheduler.unregister_camera(changed);scheduler.register_camera(changed,new_buffers[changed]);tracker_manager.remove_camera(changed);identity_manager.camera_failed(changed)
                    camera_authority_ready=True;control_runtime.status["cameras"]=len(canonical);log.info("Camera authority reconciled to API/SQLite after API recovery")
                except Exception as exc:log.warning("Camera authority reconciliation deferred: %s",exc)
            if now >= next_report:
                pipeline_metrics=scheduler.snapshot_metrics(manager.metrics())
                for camera_id,item in pipeline_metrics["cameras"].items():
                    item["detector_inputs"]=state.get("detector_inputs",{}).get(camera_id,0);item["detections"]=state.get("detection_counts",{}).get(camera_id,0);online=bool(item.get("online",False));previous=camera_event_states.get(camera_id)
                    if previous is None or previous!=online:camera_event_states[camera_id]=online;bridge.publish(CHANNELS["events"],{"type":"camera.online" if online else "camera.offline","camera_id":camera_id,"timestamp":time.time(),"details":{"backend":item.get("backend","unknown")}})
                control_runtime.status["event_delivery"]="available" if bridge.publisher.available else "unavailable";pipeline_metrics["cycle"]=state["cycle_metrics"];pipeline_metrics["unknown_snapshots"]=snapshot_manager.metrics();pipeline_metrics["secondary"]=secondary.snapshot();pipeline_metrics["tracking"]=tracker_manager.metrics.snapshot();pipeline_metrics["identity"]=identity_manager.metrics.snapshot();pipeline_metrics["detector_profile"]=detector.metrics.profile_snapshot() if detector is not None else {};pipeline_metrics["fast_path_profile"]=fast_profile.snapshot();pipeline_metrics["system"]=system_sampler.snapshot();pipeline_metrics["video"]=control_runtime.video_metrics();pipeline_metrics["event_delivery"]=bridge.publisher.snapshot();pipeline_metrics["model_forward_calls"]=pipeline_metrics.get("detector_profile",{}).get("model_forward",{}).get("count",0);control_runtime.metrics=pipeline_metrics;log.info("\n%s\nCYCLE %s\nSECONDARY %s", format_metrics(pipeline_metrics),state["cycle_metrics"],pipeline_metrics["secondary"]);bridge.publish(CHANNELS["metrics"],{"type":"system.metrics",**pipeline_metrics})
                if detector is not None:
                    log.info("\n%s\n%s\n%s\nHEATMAP %s", detector.metrics.format_compact(), tracker_manager.metrics.format_compact(),identity_manager.metrics.format_compact(),heatmap_manager.metrics.snapshot())
                next_report = now + 2
            if duration and now - started >= duration: break
    finally:
        log.info("Stopping command loop, scheduler, cameras and secondary workers")
        command_loop.stop();command_loop.join(2);scheduler.stop();manager.shutdown();scheduler.join(6);secondary.shutdown(6)
        if detector is not None:
            log.info("DETECTOR_PROFILE %s",json.dumps(detector.metrics.profile_snapshot(),sort_keys=True));log.info("FAST_PATH_PROFILE %s",json.dumps(fast_profile.snapshot(),sort_keys=True));detector.close()
        system_sampler.stop();snapshot_manager.close()
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
