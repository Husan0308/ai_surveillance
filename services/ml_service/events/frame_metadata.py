"""Build one complete metadata message for each source camera frame."""
from __future__ import annotations
from dataclasses import replace
from services.ml_service.identity.schemas import GlobalTrackResult
from services.ml_service.tracking.schemas import TrackState

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
            key=getattr(item,"local_track_id",None) or getattr(item,"global_id",None)
            if key is not None:deduped[str(key)]=item
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

def merge_visual_identity_results(tracking_results,live_results,cache,max_missing_frames=18):
    """Keep cached identity visual for observation gaps; never admits tentative tracks."""
    output=[];live_keys=set()
    tracking_by_key={(camera.camera_id,getattr(track,"local_track_id",None) or getattr(track,"track_id",None)):track for camera in tracking_results.results for track in camera.tracks}
    for result in live_results:
        annotated=[]
        for raw in result.tracks:
            key=(result.camera_id,getattr(raw,"local_track_id",None) or getattr(raw,"track_id",None));track=tracking_by_key.get(key);item=replace(raw,observation_type="detected",last_detection_timestamp=getattr(track,"last_detection_timestamp",0.0),prediction_age_ms=0.0,velocity=getattr(track,"velocity",(0.0,0.0,0.0,0.0)),state_timestamp=getattr(track,"state_timestamp",0.0),visual_expires_at=getattr(track,"visual_expires_at",0.0),track_generation=getattr(track,"track_generation",1),geometry_monotonic=getattr(track,"geometry_monotonic",0.0),visual_visible=getattr(track,"visual_visible",True),boundary_exit=getattr(track,"boundary_exit",False))
            cache[key]=item;live_keys.add(key);annotated.append(item)
        output.append(GlobalTrackResult(result.camera_id,result.frame_id,tuple(annotated)))
    for camera in tracking_results.results:
        present=set()
        for track in camera.tracks:
            key=(camera.camera_id,getattr(track,"local_track_id",None) or getattr(track,"track_id",None));present.add(key)
            if key in live_keys:continue
            predicted=getattr(track,"observation_type","")=="predicted" or (getattr(track,"state",None) in (TrackState.CONFIRMED,TrackState.LOST) and not getattr(track,"visual_hidden",False) and getattr(track,"prediction_age_ms",0.0)<=2000.0)
            if predicted and key in cache:
                item=replace(cache[key],bbox=track.predicted_bbox or track.bbox,observation_type="predicted",last_detection_timestamp=getattr(track,"last_detection_timestamp",cache[key].last_detection_timestamp),prediction_age_ms=getattr(track,"prediction_age_ms",0.0),velocity=getattr(track,"velocity",cache[key].velocity),state_timestamp=getattr(track,"state_timestamp",cache[key].state_timestamp),visual_expires_at=getattr(track,"visual_expires_at",cache[key].visual_expires_at),track_generation=getattr(track,"track_generation",cache[key].track_generation),geometry_monotonic=getattr(track,"geometry_monotonic",cache[key].geometry_monotonic),visual_visible=getattr(track,"visual_visible",cache[key].visual_visible),boundary_exit=getattr(track,"boundary_exit",cache[key].boundary_exit))
                output.append(GlobalTrackResult(camera.camera_id,camera.frame_id,(item,)))
        for key in [key for key in cache if key[0]==camera.camera_id and key not in present]:cache.pop(key,None)
    return tuple(output)
