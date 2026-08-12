"""Camera ingestion, batched person detection, and local per-camera tracking."""
import argparse,json,signal,threading,time,uuid
from urllib.error import URLError
from dataclasses import replace
from pathlib import Path
import numpy as np
from services.ml_service.cameras.config import fetch_camera_configs,load_camera_configs
from services.ml_service.cameras.manager import CameraManager
from services.ml_service.cameras.display_manager import OnDemandDisplayManager,reuses_ai_reader
from services.ml_service.pipeline.metrics import format_metrics
from services.ml_service.pipeline.scheduler import BatchScheduler
from services.ml_service.tracking.tracker_manager import TrackerManager
from services.ml_service.tracking.schemas import TrackingBatchResult
from services.ml_service.tracking.appearance import AppearanceExtractor
from services.ml_service.identity.reid_model import OSNetReIDModel
from services.ml_service.identity.global_identity_manager import GlobalIdentityManager
from services.ml_service.identity.worker import IdentityAssociationWorker
from services.ml_service.identity.coverage import ReIDTaskCoverage
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
from services.ml_service.events import frame_metadata_messages,merge_visual_identity_results
from services.ml_service.runtime_control import RuntimeCommandLoop
from services.ml_service.secondary import SecondaryAIScheduler,SecondaryTask,SecondaryTaskType
from services.ml_service.tracking.appearance import crop_detection,reid_crop_quality
from services.ml_service.metrics.timing import TimingProfile
from services.ml_service.metrics.system_sampler import SystemMetricsSampler
from services.ml_service.pipeline.gpu_coordinator import gpu_coordinator
from services.ml_service.snapshots import UnknownSnapshotManager
CHANNELS={"enrollment":"enrollment","heatmaps":"heatmaps","status":"status","metrics":"metrics","events":"events"}

def _project_config():
    config=dict(project_config());identity=dict(config.get("identity",{}));ids=[str(item["id"]) for item in camera_config().get("cameras",[]) if item.get("id")]
    identity["topology"]=compile_topology(topology_config(),ids);config["identity"]=identity;return config

def run(diagnostic=False,synthetic=False,duration=0.0,metrics_output=None,batch_size_override=None,scheduler_mode_override=None):
    settings=ServiceSettings.from_env();configure_logging(settings.log_level,"ml-service");log=get_logger(__name__)
    config=_project_config()
    if batch_size_override:
        config={**config,"ai":{**config.get("ai",{}),"batch_size":int(batch_size_override)}}
    max_age=float(config.get("ai",{}).get("max_frame_age_ms",250))
    perf=config.get("performance",{})
    import cv2
    cv2.setNumThreads(max(1,int(perf.get("opencv_threads",2))))
    try:
        import torch
        torch.set_num_threads(max(1,int(perf.get("torch_threads",4))))
        torch.set_num_interop_threads(max(1,int(perf.get("torch_interop_threads",1))))
    except RuntimeError:pass
    runtime_epoch=str(uuid.uuid4());detector, state = None, {"detections": None, "tracks": None, "cycle_metrics":{}}
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
    reid_requested={};reid_retry_after={};reid_completed_counts={};reid_coverage=ReIDTaskCoverage();face_requested={};identity_visual_cache={};identity_visual_lock=threading.RLock();last_snapshot_submit={};last_prediction_publish={};latest_prediction_packets={};prediction_lock=threading.Lock();prediction_stop=threading.Event();secondary=None;fast_profile=TimingProfile();system_sampler=SystemMetricsSampler()


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
        detector = PersonDetector(config,max_frame_age_ms=max_age,max_batch_size=int(config.get("ai",{}).get("batch_size",6)))
    def secondary_result(task,result):
        if task.task_type == SecondaryTaskType.REID:
            accepted=tracker_manager.set_embedding(task.camera_id,task.local_track_id,result,float((task.context or {}).get("reid_quality",0.0)),task.frame_id,task.capture_timestamp)
            context=task.context or {};quality=float(context.get("reid_quality",0.0));quality_reason=str(context.get("quality_reason","unknown"));usable=accepted and quality_reason=="quality_ok" and quality>=identity_manager.min_quality
            if usable:
                key=(task.camera_id,task.local_track_id);reid_completed_counts[key]=reid_completed_counts.get(key,0)+1
            retry_delay=float(identity_cfg.get("initial_evidence_interval_seconds",.75)) if usable else float(identity_cfg.get("low_quality_retry_seconds",3.0));retry_at=time.time()+retry_delay;reid_retry_after[(task.camera_id,task.local_track_id)]=retry_at
            reid_coverage.completed(task.camera_id,task.local_track_id,quality,quality_reason,usable,accepted,context.get("crop_width",0),context.get("crop_height",0),retry_at)
            identity_manager.record_reid_completed(orphaned=not accepted)
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
        eligible=[];eligible_indices=[];outputs=[None]*len(tasks)
        for index,task in enumerate(tasks):
            quality=reid_crop_quality(task.crop,float((task.context or {}).get("detection_confidence",0.0)));task.context.update(reid_quality=quality["score"],crop_width=quality["width"],crop_height=quality["height"],blur_variance=quality["blur_variance"],quality_reason=quality["reason"])
            if quality["reason"]=="quality_ok" and quality["score"]>=identity_manager.min_quality:eligible_indices.append(index);eligible.append(task)
        embeddings,timing=appearance.extract_batch([task.crop for task in eligible]) if eligible else ([],{"total_ms":0.0})
        for index,embedding in zip(eligible_indices,embeddings):outputs[index]=embedding
        tracker_manager.reid_batch_size=len(eligible);tracker_manager.reid_extract_ms=float(timing.get("total_ms",timing.get("gpu_ms",0)))
        return outputs

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

    def identity_worker_result(_result,remaps):
        if not remaps:return
        with identity_visual_lock:
            for old_id,canonical_id in remaps:
                snapshot_manager.remap(old_id,canonical_id)
                for cache_key,cached in list(identity_visual_cache.items()):
                    if cached.global_id==old_id:identity_visual_cache[cache_key]=replace(cached,global_id=canonical_id,display_name=canonical_id)
                bridge.publish(CHANNELS["events"],{"type":"identity.merged","old_global_id":old_id,"global_id":canonical_id,"identity_version":identity_manager.identity_version,"identity_runtime_epoch":runtime_epoch,"timestamp":time.time()})
    identity_cfg=config.get("identity",{})
    identity_worker=IdentityAssociationWorker(identity_manager,identity_worker_result,identity_cfg.get("worker_queue_size",64),identity_cfg.get("worker_max_task_age_ms",2000))
    if detector is not None:
        detector.diagnostic_context_provider=lambda:{"system":system_sampler.snapshot(),"secondary":secondary.snapshot(),"roi":detector.roi_snapshot(),"video":control_runtime.video_metrics(),"gpu":gpu_coordinator.snapshot(False)}

    def consume(batch):
        if detector is None:return
        # Fetch newest snapshot immediately before GPU launch
        refreshed_frames=[]
        for packet in batch.frames:
            latest=scheduler.take_if_newer(packet.camera_id,packet.frame_id)
            refreshed_frames.append(replace(latest,scheduler_selected_timestamp=packet.scheduler_selected_timestamp) if latest else packet)
        from services.ml_service.pipeline.batch import BatchOutput
        batch=BatchOutput(batch.batch_id,batch.created_timestamp,tuple(refreshed_frames),batch.build_started_monotonic,batch.build_completed_monotonic)

        cycle_started=time.perf_counter();inference_started=time.time()
        # Record actual capture age at launch
        launch_capture_age_ms=max([0.0]+[(inference_started-packet.capture_timestamp)*1000 for packet in batch.frames])

        phase=time.perf_counter()
        video_handoff_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();detections=detector.process_batch(batch);detector_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter();frames = {packet.camera_id: packet.frame for packet in batch.frames}
        packets = {packet.camera_id: packet for packet in batch.frames}
        tracker_input_ms=(time.perf_counter()-phase)*1000
        state["detections"] = detections
        state["detector_inputs"]={item.camera_id:1 for item in detections.results};state["detection_counts"]={item.camera_id:len(item.detections) for item in detections.results}
        phase=time.perf_counter();state["tracks"] = tracker_manager.update_batch(detections,frames);detector.update_track_hints(state["tracks"]);scheduler.update_camera_risks(tracker_manager.scheduler_risks());tracking_ms=(time.perf_counter()-phase)*1000
        current_track_keys=tracker_manager.active_track_keys()
        reid_coverage.prune(current_track_keys)
        for stale_key in set(reid_requested)-current_track_keys:reid_requested.pop(stale_key,None);reid_retry_after.pop(stale_key,None);reid_completed_counts.pop(stale_key,None)
        identity_manager.metrics.values.reid_batch_size=tracker_manager.reid_batch_size
        identity_manager.metrics.values.reid_extract_ms=tracker_manager.reid_extract_ms
        phase=time.perf_counter();identities=[]
        for camera in state["tracks"].results:
            for track in camera.tracks:
                if track.misses or track.state.value!="CONFIRMED": continue
                embedding,embedding_quality,embedding_frame_id,embedding_timestamp=tracker_manager.embedding_evidence_for(camera.camera_id,track.track_id)
                packet=frames.get(camera.camera_id)
                reid_coverage.update(camera.camera_id,track.track_id,reid_eligible=True,reid_embeddings_fresh=bool(embedding is not None and embedding_timestamp is not None and camera.receive_timestamp-embedding_timestamp<=identity_manager.max_evidence_age_ms/1000),independent_evidence_count=reid_completed_counts.get((camera.camera_id,track.track_id),0))
                observation=IdentityTrackObservation(camera.camera_id,camera.frame_id,track.track_id,
                    track.bbox,track.confidence,camera.receive_timestamp,embedding,embedding_quality,
                    embedding_frame_id,embedding_timestamp,getattr(packet,"width",0),getattr(packet,"height",0),track.detection_source,track.detection_id)
                identities.append(identity_worker.observe(observation))
        with identity_visual_lock:
            identities=list(merge_visual_identity_results(state["tracks"],identities,identity_visual_cache,int(config.get("ai",{}).get("tracker",{}).get("visible_missing_frames",5))))
        state["identities"]=tuple(identities)
        # Snapshot cropping copies pixels. Admit at most one due identity per
        # detector cycle; the snapshot worker still performs quality/disk work.
        snapshot_candidate=None;snapshot_now=time.monotonic()
        for identity_result in identities:
            frame=frames.get(identity_result.camera_id)
            if frame is None:continue
            for item in identity_result.tracks:
                if item.global_id and not item.person_id and snapshot_now-last_snapshot_submit.get(item.global_id,0)>=2.0:
                    candidate=(last_snapshot_submit.get(item.global_id,0),identity_result,item,frame)
                    if snapshot_candidate is None or candidate[0]<snapshot_candidate[0]:snapshot_candidate=candidate
        if snapshot_candidate is not None:
            _,identity_result,item,frame=snapshot_candidate;packet=packets[identity_result.camera_id]
            if snapshot_manager.submit(item.global_id,identity_result.camera_id,identity_result.frame_id,packet.capture_timestamp,frame,item.bbox):last_snapshot_submit[item.global_id]=snapshot_now
        identity_ms=(time.perf_counter()-phase)*1000
        phase=time.perf_counter()
        # Single Canonical Publisher: YOLO no longer publishes metadata directly.
        # The 15fps visual_state_publisher handles all frontend updates.
        metadata_prepare_ms=(time.perf_counter()-phase)*1000
        metadata_publish_ms=0.0
        metadata_published=time.time()
        phase=time.perf_counter();reid_crop_ms=face_crop_ms=enrollment_copy_ms=0.0;secondary_eligibility_ms=secondary_crop_select_ms=secondary_crop_copy_ms=secondary_quality_ms=secondary_task_ms=secondary_queue_ms=0.0
        now=time.time();refresh=float(reid_cfg.get("embedding_refresh_seconds",10));cooldown=float(reid_cfg.get("event_cooldown_seconds",5));reid_cache_hits=0
        if SecondaryTaskType.REID in processors:
            for camera in state["tracks"].results:
                frame=frames.get(camera.camera_id);packet=packets.get(camera.camera_id)
                if frame is None or packet is None:continue
                for track in camera.tracks:
                    eligibility_started=time.perf_counter()
                    if track.misses or track.state.value!="CONFIRMED":secondary_eligibility_ms+=(time.perf_counter()-eligibility_started)*1000;continue
                    embedding=tracker_manager.embedding_for(camera.camera_id,track.track_id);key=(camera.camera_id,track.track_id);last=reid_requested.get(key,0)
                    evidence_interval=float(identity_cfg.get("initial_evidence_interval_seconds",.75)) if reid_completed_counts.get(key,0)<2 else refresh
                    due=(embedding is None and now-last>=cooldown or embedding is not None and now-last>=evidence_interval) and now>=reid_retry_after.get(key,0)
                    secondary_eligibility_ms+=(time.perf_counter()-eligibility_started)*1000
                    if not due:
                        reid_coverage.update(*key,reason="fresh_evidence_cooldown",decision="WAITING",next_retry_at=max(reid_retry_after.get(key,0),last+(cooldown if embedding is None else evidence_interval)))
                        if embedding is not None:reid_cache_hits+=1
                        continue
                    if not secondary.can_accept(SecondaryTaskType.REID):reid_coverage.update(*key,reason="queue_busy_retry_next_cycle",decision="RETRY");continue
                    crop_started=time.perf_counter();crop=crop_detection(frame,track.bbox);secondary_crop_select_ms+=(time.perf_counter()-crop_started)*1000
                    if crop is None or not crop.size:reid_coverage.update(*key,reason="invalid_crop_retry_next_cycle",decision="RETRY");continue
                    copy_started=time.perf_counter();owned=crop.copy();secondary_crop_copy_ms+=(time.perf_counter()-copy_started)*1000;reid_crop_ms+=(time.perf_counter()-crop_started)*1000
                    task_started=time.perf_counter();context={"detection_confidence":track.confidence,"reid_quality":0.0,"crop_width":int(crop.shape[1]),"crop_height":int(crop.shape[0]),"blur_variance":0.0,"quality_reason":"pending_async_quality"};task=SecondaryTask(SecondaryTaskType.REID,camera.camera_id,track.track_id,None,camera.frame_id,packet.capture_timestamp,track.bbox,owned,context=context);secondary_task_ms+=(time.perf_counter()-task_started)*1000
                    queue_started=time.perf_counter();submitted=secondary.submit(task);secondary_queue_ms+=(time.perf_counter()-queue_started)*1000
                    if submitted:reid_requested[(camera.camera_id,track.track_id)]=now;identity_manager.record_reid_submitted();reid_coverage.submitted(*key)
                    else:reid_coverage.update(*key,reason="queue_race_retry_next_cycle",decision="RETRY")
        if reid_cache_hits:secondary.cache_hit(count=reid_cache_hits)
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
        profile_values={"secondary_eligibility":secondary_eligibility_ms,"secondary_crop_selection":secondary_crop_select_ms,"secondary_crop_copy":secondary_crop_copy_ms,"secondary_quality":secondary_quality_ms,"secondary_task_construction":secondary_task_ms,"secondary_queue_submission":secondary_queue_ms,"capture_to_scheduler":max((packet.scheduler_selected_timestamp-packet.capture_timestamp)*1000 for packet in batch.frames),"scheduler_wait":max((inference_started-packet.scheduler_selected_timestamp)*1000 for packet in batch.frames),"video_handoff":video_handoff_ms,"pure_detector":detector_ms,"tracker_input_construction":tracker_input_ms,"tracker":tracking_ms,"identity":identity_ms,"metadata_preparation":metadata_prepare_ms,"metadata_publication":metadata_publish_ms,"reid_crop_preparation":reid_crop_ms,"face_crop_preparation":face_crop_ms,"enrollment_frame_copy":enrollment_copy_ms,"secondary_submission":face_ms,"heatmap":heatmap_ms,"message_publishing":publishing_ms,"full_fast_path":(time.perf_counter()-cycle_started)*1000,"capture_to_metadata":max((metadata_published-packet.capture_timestamp)*1000 for packet in batch.frames),"launch_capture_age":launch_capture_age_ms};fast_profile.record(profile_values)
        state["cycle_metrics"]={**profile_values,"inference_start":inference_started,"inference_end":detections.completed_at,"metadata_publish_timestamp":metadata_published}
        detector_batch=detector.metrics.snapshot();detector_batch.update(system=system_sampler.snapshot(),secondary=secondary.snapshot(),identity_worker=identity_worker.snapshot(),gpu_runtime=detector.runtime_snapshot(),high_resolution_frames=sum(packet.width>1280 or packet.height>720 for packet in batch.frames),display_fullscreen=getattr(control_runtime,"high_quality_camera",None))
        state["detector_batch"]=detector_batch;log.debug("DETECTOR_BATCH_RECORD %s",json.dumps(detector_batch,sort_keys=True,default=str))
    scheduler_cfg=config.get("ai",{}).get("scheduler",{});scheduler_mode=scheduler_mode_override or scheduler_cfg.get("mode","fixed")
    scheduler = BatchScheduler(max_age,on_batch=consume,max_batch_size=int(config.get("ai",{}).get("batch_size",6)),mode=scheduler_mode,
        min_batch_size=scheduler_cfg.get("min_batch_size",2),fairness_deadline_ms=scheduler_cfg.get("fairness_deadline_ms",900))
    capture_factory = None
    if synthetic:
        from services.ml_service.diagnostic import synthetic_capture_factory
        capture_factory = synthetic_capture_factory
    prediction_rate=max(1.0,float(config.get("display",{}).get("prediction_fps",15)))
    _visual_metadata_version = {}
    def publish_visual_state():
        interval=1.0/prediction_rate
        while not prediction_stop.wait(interval):
            with prediction_lock:packets=tuple(latest_prediction_packets.values())
            messages=[]
            for packet in packets:
                # Ordering guard: only publish strictly newer frames to prevent rewind races.
                if packet.frame_id<=last_prediction_publish.get(packet.camera_id,-1):continue
                mono=time.monotonic();predicted=tracker_manager.predict_visual(packet.camera_id,packet.frame_id,packet.capture_timestamp,packet.receive_timestamp,packet.capture_monotonic or mono)
                last_prediction_publish[packet.camera_id]=packet.frame_id
                if predicted is None:continue
                batch_result=TrackingBatchResult(0,packet.receive_timestamp,packet.receive_timestamp,(predicted,))
                with identity_visual_lock:visual=merge_visual_identity_results(batch_result,(),identity_visual_cache,0)
                packet_messages=frame_metadata_messages((packet,),visual,identity_manager.canonicalize,identity_manager.identity_version,runtime_epoch)
                if packet_messages:
                    msg = packet_messages[0]
                    cam = msg["camera_id"]
                    _visual_metadata_version[cam] = _visual_metadata_version.get(cam, 0) + 1
                    msg["metadata_version"] = _visual_metadata_version[cam]
                    messages.append(msg)
            if messages:bridge.publish(CHANNELS["events"],{"type":"frame.metadata.batch","messages":messages})
    prediction_thread=threading.Thread(target=publish_visual_state,name="visual-state-publisher",daemon=False)
    def camera_frame_handoff(packet):
        control_runtime.frame(packet)
        with prediction_lock:latest_prediction_packets[packet.camera_id]=packet
    manager = CameraManager(scheduler.notify_frame_available,capture_factory,camera_frame_handoff)
    display_manager=OnDemandDisplayManager(control_runtime.display_frame,capture_factory)
    camera_authority_ready=True
    try:cameras=fetch_camera_configs()
    except (URLError,TimeoutError,ConnectionError,OSError):
        cameras=load_camera_configs();camera_authority_ready=False
    runtime_cameras.update({item["id"]:dict(item) for item in cameras});detector.configure_rois(cameras) if detector is not None else None;manager.configure(cameras)
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
        elif kind=="display.source.start":
            camera_id=str(command.get("camera_id") or "");item=runtime_cameras.get(camera_id)
            if item and (reuses_ai_reader(item) or display_manager.start(camera_id,item)):
                reuse_ai=reuses_ai_reader(item);control_runtime.set_high_quality(camera_id,reuse_ai);log.info("%s fullscreen display source %s",camera_id,"reuses AI reader" if reuse_ai else "opened on demand")
            else:log.warning("%s fullscreen display source unavailable",camera_id)
        elif kind=="display.source.stop":
            camera_id=str(command.get("camera_id") or "");display_manager.stop(camera_id);control_runtime.clear_high_quality(camera_id);log.info("%s fullscreen display source closed",camera_id)
        elif kind=="camera.config.changed":
            camera_id=command.get("camera_id");action=command.get("action");old_buffers=manager.buffers();old_ids=set(old_buffers)
            if action=="deleted":runtime_cameras.pop(camera_id,None)
            else:
                previous=runtime_cameras.get(camera_id,{});item={**previous,**dict(command.get("config",{}))}
                item["id"]=camera_id;item["source"]=item.get("rtsp_url",item.get("source"));item["online"]=item.get("enabled",item.get("online",True));runtime_cameras[camera_id]=item
            detector.configure_rois(runtime_cameras.values()) if detector is not None else None;manager.configure(runtime_cameras.values());new_buffers=manager.buffers()
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
    prediction_thread.start();system_sampler.start();secondary.start();identity_worker.start();command_loop.start();scheduler.start(); manager.start(); started = time.monotonic(); next_report = started + 2
    try:
        while not stop.wait(.25):
            now = time.monotonic()
            if not camera_authority_ready and bridge.publisher.available:
                try:
                    canonical=fetch_camera_configs();old_buffers=manager.buffers();old_ids=set(old_buffers);runtime_cameras.clear();runtime_cameras.update({item["id"]:dict(item) for item in canonical});detector.configure_rois(canonical) if detector is not None else None;manager.configure(canonical);new_buffers=manager.buffers()
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
                control_runtime.status["event_delivery"]="available" if bridge.publisher.available else "unavailable";pipeline_metrics["cycle"]=state["cycle_metrics"];pipeline_metrics["unknown_snapshots"]=snapshot_manager.metrics();pipeline_metrics["secondary"]=secondary.snapshot();identity_manager.record_reid_stale(pipeline_metrics["secondary"].get("reid",{}).get("stale",0));pipeline_metrics["tracking"]=tracker_manager.metrics.snapshot();pipeline_metrics["identity"]=identity_manager.metrics.snapshot();pipeline_metrics["identity_worker"]=identity_worker.snapshot();pipeline_metrics["reid_task_coverage"]=reid_coverage.snapshot(identity_manager.decision_snapshot());pipeline_metrics["detector_batch"]=state.get("detector_batch",{});pipeline_metrics["detector_profile"]=detector.metrics.profile_snapshot() if detector is not None else {};pipeline_metrics["detector_runtime"]=detector.runtime_snapshot() if detector is not None else {};pipeline_metrics["roi_recovery"]=detector.roi_snapshot() if detector is not None else {};pipeline_metrics["stale_before_gpu"]=detector.metrics.stale_drops_before_inference if detector is not None else 0;pipeline_metrics["fast_path_profile"]=fast_profile.snapshot();pipeline_metrics["system"]=system_sampler.snapshot();pipeline_metrics["video"]=control_runtime.video_metrics();pipeline_metrics["reader_ownership"]={"ai_reader_count":manager.reader_count(),"ai_reader_ids":manager.reader_ids(),"display":display_manager.snapshot(),"high_quality_camera":control_runtime.high_quality_camera,"high_quality_reuses_ai":control_runtime.high_quality_reuses_ai};pipeline_metrics["event_delivery"]=bridge.publisher.snapshot();pipeline_metrics["gpu_coordinator"]=gpu_coordinator.snapshot();pipeline_metrics["model_forward_calls"]=pipeline_metrics.get("detector_runtime",{}).get("gpu_batches_completed",0);control_runtime.metrics=pipeline_metrics;log.info("\n%s",format_metrics(pipeline_metrics));log.debug("CYCLE %s SECONDARY %s",state["cycle_metrics"],pipeline_metrics["secondary"]);bridge.publish(CHANNELS["metrics"],{"type":"system.metrics",**pipeline_metrics})
                if detector is not None:
                    log.info("\n%s\n%s\n%s\nROOM_IDENTITY %s\nHEATMAP %s", detector.metrics.format_compact(), tracker_manager.metrics.format_compact(),identity_manager.metrics.format_compact(),json.dumps(identity_manager.metrics.snapshot().get("room_identity") or {},sort_keys=True),heatmap_manager.metrics.snapshot())
                next_report = now + 2
            if duration and now - started >= duration: break
    finally:
        clean=True;log.info("Stopping control API, command loop, scheduler, cameras and workers")
        clean=bridge.stop_server() and clean
        prediction_stop.set();prediction_thread.join(2);clean=not prediction_thread.is_alive() and clean
        command_loop.stop();clean=command_loop.join(2) and clean
        display_manager.shutdown();clean=control_runtime.shutdown(3) and clean;control_runtime.high_quality_camera=None;control_runtime.high_quality_reuses_ai=False;scheduler.stop();manager.shutdown()
        clean=scheduler.join(6) and clean;clean=secondary.shutdown(6) and clean;clean=identity_worker.shutdown(6) and clean
        if detector is not None:
            log.debug("DETECTOR_PROFILE %s",json.dumps(detector.metrics.profile_snapshot(),sort_keys=True));log.debug("FAST_PATH_PROFILE %s",json.dumps(fast_profile.snapshot(),sort_keys=True));detector.close()
        system_sampler.stop();snapshot_manager.close()
        if face_engine is not None:face_engine.close()
        if reid_model is not None:reid_model.close()
        if metrics_output and control_runtime.metrics:
            output=Path(metrics_output);output.parent.mkdir(parents=True,exist_ok=True)
            temporary=output.with_suffix(output.suffix+".tmp")
            temporary.write_text(json.dumps(control_runtime.metrics,sort_keys=True)+"\n",encoding="utf-8");temporary.replace(output)
        tracker_manager.close();identity_manager.close();control_runtime.status={"status":"stopped"};bridge.publish(CHANNELS["status"],{"type":"system.status","status":"stopped"});clean=bridge.close() and clean
        if clean:log.info("ML Service stopped cleanly")
        else:log.error("ML Service stopped with incomplete worker shutdown")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true", help="run camera/batching without YOLO")
    parser.add_argument("--synthetic", action="store_true", help="use configured IDs with fake sources")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--metrics-output",type=Path,help="atomically write the final metrics snapshot")
    parser.add_argument("--batch-size",type=int,choices=(1,2,3,6),help="process-local detector A/B override")
    parser.add_argument("--scheduler-mode",choices=("fixed","risk_aware"),help="process-local scheduler A/B override")
    args=parser.parse_args();run(args.diagnostic,args.synthetic,args.duration,args.metrics_output,args.batch_size,args.scheduler_mode)

if __name__ == "__main__": main()
