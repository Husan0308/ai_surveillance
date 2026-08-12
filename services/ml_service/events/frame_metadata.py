"""Build one complete metadata message for each source camera frame."""
from __future__ import annotations
from dataclasses import replace
from services.ml_service.identity.schemas import GlobalTrack,GlobalTrackResult,IdentityStatus
from services.ml_service.tracking.schemas import TrackState

def _track_key(item):
    """Canonical local trajectory key shared by tracker and identity layers.

    TrackedPerson exposes both ``track_id`` (camera-scoped stable string) and
    ``local_track_id`` (numeric local id), while GlobalTrack stores the stable
    string in ``local_track_id``. Always prefer ``track_id`` so the visual cache
    is addressable from both representations. The old mixed int/string keys made
    the single visual publisher miss its identity cache and emit empty metadata.
    """
    value=getattr(item,"track_id",None) or getattr(item,"local_track_id",None) or getattr(item,"global_id",None)
    return None if value is None else str(value)

def frame_metadata_messages(packets,identity_results,canonicalize=None,identity_version=0,runtime_epoch=None):
    grouped={}
    for result in identity_results:
        grouped.setdefault((result.camera_id,result.frame_id),[]).extend(result.tracks)
    canonicalize=canonicalize or (lambda value:value)
    messages=[]
    for packet in packets:
        current=grouped.get((packet.camera_id,packet.frame_id),())
        deduped={}
        for item in current:
            key=_track_key(item)
            if key is not None:deduped[key]=item
        tracks=tuple(deduped.values())
        version=max([int(identity_version),*[int(getattr(item,"identity_version",0)) for item in tracks]])
        messages.append({
            "type":"frame.metadata","camera_id":packet.camera_id,"frame_id":packet.frame_id,"identity_version":version,"identity_runtime_epoch":runtime_epoch,
            "timestamp":packet.capture_timestamp,"capture_timestamp":packet.capture_timestamp,
            "frame_width":packet.width,"frame_height":packet.height,
            "tracks":[{
                "bbox":list(item.bbox),"confidence":item.confidence,
                "local_track_id":item.local_track_id,"global_id":canonicalize(item.global_id),"identity_version":version,
                "person_id":getattr(item,"person_id",None),
                "display_name":getattr(item,"display_name",None) if getattr(item,"person_id",None) else canonicalize(item.global_id),
                "observation_type":getattr(item,"observation_type","detected"),
                "last_detection_timestamp":getattr(item,"last_detection_timestamp",0.0),
                "prediction_age_ms":getattr(item,"prediction_age_ms",0.0),
                "tracker_state":getattr(item,"tracker_state","DETECTED"),
                "detection_source":getattr(item,"detection_source","PREDICTED"),
                "detection_id":getattr(item,"detection_id",None),
                "velocity":list(getattr(item,"velocity",(0.0,0.0,0.0,0.0))),
                "state_timestamp":getattr(item,"state_timestamp",packet.capture_timestamp),
                "visual_expires_at":getattr(item,"visual_expires_at",0.0),
                "track_generation":getattr(item,"track_generation",1),
                "geometry_monotonic":getattr(item,"geometry_monotonic",0.0),
                "visual_visible":getattr(item,"visual_visible",True),
                "boundary_exit":getattr(item,"boundary_exit",False),
                "geometry_units":"source_pixels;velocity_source_pixels_per_second",
            } for item in tracks],
        })
    return messages

def _tentative_local_visual(track):
    """Expose only a fresh real tentative detection to the UI.

    Tentative tracks deliberately have no global identity and are never cached or
    predicted through a real detector miss. This lets the operator see YOLO on
    the first observation while preserving the multi-hit identity barrier.
    """
    return GlobalTrack(
        local_track_id=_track_key(track),
        global_id=None,
        bbox=getattr(track,"bbox",(0.0,0.0,0.0,0.0)),
        confidence=float(getattr(track,"confidence",0.0)),
        identity_confidence=0.0,
        identity_status=IdentityStatus.ACTIVE,
        decision_reason="tentative_local_visual",
        person_id=None,
        display_name=None,
        observation_type="detected",
        last_detection_timestamp=float(getattr(track,"last_detection_timestamp",0.0)),
        prediction_age_ms=float(getattr(track,"prediction_age_ms",0.0)),
        tracker_state=TrackState.TENTATIVE.value,
        identity_version=0,
        detection_source=getattr(track,"detection_source","FULL_FRAME"),
        detection_id=getattr(track,"detection_id",None),
        velocity=tuple(getattr(track,"velocity",(0.0,0.0))),
        state_timestamp=float(getattr(track,"state_timestamp",0.0)),
        visual_expires_at=float(getattr(track,"visual_expires_at",0.0)),
        track_generation=int(getattr(track,"track_generation",1)),
        geometry_monotonic=float(getattr(track,"geometry_monotonic",0.0)),
        visual_visible=bool(getattr(track,"visual_visible",True)),
        boundary_exit=bool(getattr(track,"boundary_exit",False)),
    )

def merge_visual_identity_results(tracking_results,live_results,cache,max_missing_frames=18):
    """Merge tracker geometry with identity state using one stable local key."""
    output=[];live_keys=set()
    tracking_by_key={(camera.camera_id,_track_key(track)):track for camera in tracking_results.results for track in camera.tracks if _track_key(track) is not None}
    for result in live_results:
        annotated=[]
        for raw in result.tracks:
            key=(result.camera_id,_track_key(raw));track=tracking_by_key.get(key)
            item=replace(raw,observation_type="detected",last_detection_timestamp=getattr(track,"last_detection_timestamp",0.0),prediction_age_ms=0.0,velocity=getattr(track,"velocity",(0.0,0.0,0.0,0.0)),state_timestamp=getattr(track,"state_timestamp",0.0),visual_expires_at=getattr(track,"visual_expires_at",0.0),track_generation=getattr(track,"track_generation",1),geometry_monotonic=getattr(track,"geometry_monotonic",0.0),visual_visible=getattr(track,"visual_visible",True),boundary_exit=getattr(track,"boundary_exit",False))
            cache[key]=item;live_keys.add(key);annotated.append(item)
        output.append(GlobalTrackResult(result.camera_id,result.frame_id,tuple(annotated)))
    for camera in tracking_results.results:
        present=set()
        for track in camera.tracks:
            local_key=_track_key(track)
            if local_key is None:continue
            key=(camera.camera_id,local_key);present.add(key)
            if key in live_keys:continue
            state=getattr(track,"state",None);misses=int(getattr(track,"misses",0) or 0)
            if state==TrackState.TENTATIVE and misses==0:
                output.append(GlobalTrackResult(camera.camera_id,camera.frame_id,(_tentative_local_visual(track),)))
                continue
            predicted=getattr(track,"observation_type","")=="predicted" or (state in (TrackState.CONFIRMED,TrackState.LOST) and not getattr(track,"visual_hidden",False) and getattr(track,"prediction_age_ms",0.0)<=2000.0)
            if predicted and key in cache:
                item=replace(cache[key],bbox=track.predicted_bbox or track.bbox,observation_type="predicted",last_detection_timestamp=getattr(track,"last_detection_timestamp",cache[key].last_detection_timestamp),prediction_age_ms=getattr(track,"prediction_age_ms",0.0),velocity=getattr(track,"velocity",cache[key].velocity),state_timestamp=getattr(track,"state_timestamp",cache[key].state_timestamp),visual_expires_at=getattr(track,"visual_expires_at",cache[key].visual_expires_at),track_generation=getattr(track,"track_generation",cache[key].track_generation),geometry_monotonic=getattr(track,"geometry_monotonic",cache[key].geometry_monotonic),visual_visible=getattr(track,"visual_visible",cache[key].visual_visible),boundary_exit=getattr(track,"boundary_exit",cache[key].boundary_exit))
                output.append(GlobalTrackResult(camera.camera_id,camera.frame_id,(item,)))
        for key in [key for key in cache if key[0]==camera.camera_id and key not in present]:cache.pop(key,None)
    return tuple(output)
