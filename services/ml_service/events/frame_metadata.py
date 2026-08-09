"""Build one complete metadata message for each source camera frame."""
from __future__ import annotations

def frame_metadata_messages(packets,identity_results):
    grouped={}
    for result in identity_results:
        grouped.setdefault((result.camera_id,result.frame_id),[]).extend(result.tracks)
    messages=[]
    for packet in packets:
        tracks=grouped.get((packet.camera_id,packet.frame_id),())
        messages.append({
            "type":"frame.metadata","camera_id":packet.camera_id,"frame_id":packet.frame_id,
            "timestamp":packet.capture_timestamp,
            "tracks":[{
                "bbox":list(item.bbox),"confidence":item.confidence,
                "local_track_id":item.local_track_id,"global_id":item.global_id,
                "person_id":getattr(item,"person_id",None),
                "display_name":getattr(item,"display_name",None),
            } for item in tracks],
        })
    return messages
