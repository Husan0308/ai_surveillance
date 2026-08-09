"""Canonical, manually supplied camera topology with strict validation."""
from __future__ import annotations
from pathlib import Path
import yaml

class TopologyValidationError(ValueError):pass

def load_topology(path):
    value=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value,dict):raise TopologyValidationError("topology root must be a mapping")
    return value

def compile_topology(raw,camera_ids):
    known=set(map(str,camera_ids));verified=bool(raw.get("verified",False));rooms=raw.get("rooms",{}) or {}
    if not isinstance(rooms,dict):raise TopologyValidationError("rooms must be a mapping")
    membership={};room_cameras={}
    for room_id,room in rooms.items():
        cameras=(room or {}).get("cameras",[]) if isinstance(room,dict) else room
        cameras=list(map(str,cameras or []))
        if len(cameras)!=len(set(cameras)):raise TopologyValidationError(f"duplicate camera in room {room_id}")
        unknown=set(cameras)-known
        if unknown:raise TopologyValidationError(f"room {room_id} references unknown cameras: {sorted(unknown)}")
        for camera in cameras:
            if camera in membership:raise TopologyValidationError(f"{camera} assigned to contradictory rooms")
            membership[camera]=str(room_id)
        room_cameras[str(room_id)]=cameras
    overlaps=[];seen=set()
    for pair in raw.get("overlaps",[]) or []:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise TopologyValidationError("each overlap must contain exactly two cameras")
        a,b=map(str,pair)
        if a==b:raise TopologyValidationError(f"{a} cannot overlap itself")
        unknown={a,b}-known
        if unknown:raise TopologyValidationError(f"overlap references unknown cameras: {sorted(unknown)}")
        key=frozenset((a,b))
        if key in seen:raise TopologyValidationError(f"duplicate overlap: {a}/{b}")
        seen.add(key);overlaps.append([a,b])
    adjacency=raw.get("adjacency",{}) or {};relationships={}
    if not isinstance(adjacency,dict):raise TopologyValidationError("adjacency must be a mapping")
    for room,others in adjacency.items():
        room=str(room)
        if room not in room_cameras:raise TopologyValidationError(f"adjacency references unknown room: {room}")
        if len(others or [])!=len(set(others or [])):raise TopologyValidationError(f"duplicate adjacency for {room}")
        for other in others or []:
            other=str(other)
            if other==room:raise TopologyValidationError(f"room {room} cannot be adjacent to itself")
            if other not in room_cameras:raise TopologyValidationError(f"adjacency references unknown room: {other}")
            if room not in list(map(str,adjacency.get(other,[]) or [])):raise TopologyValidationError(f"adjacency must be symmetric: {room}/{other}")
            relationships[f"{room}:{other}"]="possible_transition"
    travel={};
    for route,value in (raw.get("travel_time",{}) or {}).items():
        if "->" not in str(route):raise TopologyValidationError(f"invalid travel route: {route}")
        start,end=map(str.strip,str(route).split("->",1))
        if start not in room_cameras or end not in room_cameras:raise TopologyValidationError(f"travel route references unknown room: {route}")
        seconds=(value or {}).get("min_seconds") if isinstance(value,dict) else value
        if float(seconds)<0:raise TopologyValidationError(f"negative travel time: {route}")
        travel[f"{start}:{end}"]=float(seconds)*1000
    if verified and set(membership)!=known:raise TopologyValidationError("verified topology must assign every camera to one room")
    return {"verified":verified,"camera_rooms":membership,"overlapping_camera_pairs":overlaps,"relationships":relationships,"min_travel_time_ms":travel,"default_min_travel_ms":20000}

def validate_topology(camera_ids,records):
    """Compatibility validator for the P3 per-camera representation."""
    known=set(camera_ids);normalized={}
    for camera_id in known:
        item=dict(records.get(camera_id,{}) or {});overlap=set(item.get("overlapping_camera_ids",()));adjacent=set(item.get("adjacent_camera_ids",()));separate=set(item.get("physically_separate_camera_ids",()))
        if camera_id in overlap:raise TopologyValidationError(f"{camera_id} cannot overlap itself")
        unknown=(overlap|adjacent|separate)-known
        if unknown:raise TopologyValidationError(f"{camera_id} references unknown cameras: {sorted(unknown)}")
        normalized[camera_id]={"camera_id":camera_id,"room_id":item.get("room_id"),"overlapping_camera_ids":sorted(overlap),"adjacent_camera_ids":sorted(adjacent),"physically_separate_camera_ids":sorted(separate)}
    for camera_id,item in normalized.items():
        for other in item["overlapping_camera_ids"]:
            if camera_id not in normalized[other]["overlapping_camera_ids"]:raise TopologyValidationError(f"overlap must be symmetric: {camera_id}/{other}")
    return normalized
