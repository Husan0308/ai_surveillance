# =====================================================================
#  AI SURVEILLANCE SYSTEM — OPERATOR CONSOLE UI
#  Python + PySide6  |  ui.py — LOCKED
# =====================================================================

import os
import math
import csv
import time
import logging
from collections import deque
from datetime import datetime

import shiboken6

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from services.frontend.api_client import ApiClient
from services.frontend.async_api import AsyncApi
from services.frontend.websocket_client import WebSocketClient
from services.frontend.roi_editor import ROIEditorDialog
from shared.settings import ServiceSettings


log = logging.getLogger(__name__)


# ============================ THEME ==================================
class TH:
    BG     = "#0f1317"
    PANEL  = "#151b22"
    CARD   = "#1a212a"
    CARD2  = "#212a35"
    HOVER  = "#26303c"
    BORDER = "#2b3542"
    ACCENT = "#2f7df6"
    ACC2   = "#5b9bff"
    OK     = "#2ecc71"
    WARN   = "#f5c542"
    ERR    = "#ef5350"
    TXT    = "#e9eef5"
    DIM    = "#94a1b3"
    FAINT  = "#5d6b7e"


FRAME_W, FRAME_H = 640, 360
GW, GH = 64, 36
CAMERA_GRID_COLUMNS = 2

def camera_grid_position(index):
    return index // CAMERA_GRID_COLUMNS, index % CAMERA_GRID_COLUMNS

def aspect_fit_rect(container_width, container_height, frame_width, frame_height):
    if frame_width <= 0 or frame_height <= 0:frame_width, frame_height = FRAME_W, FRAME_H
    scale = min(container_width / frame_width, container_height / frame_height)
    width, height = frame_width * scale, frame_height * scale
    return QRectF((container_width - width) / 2, (container_height - height) / 2, width, height)


def map_bbox_to_video_rect(video_rect, frame_width, frame_height, bbox):
    """Map a source-pixel bbox once into an aspect-fitted rendered image rectangle."""
    if frame_width <= 0 or frame_height <= 0 or video_rect.isEmpty():return QRectF()
    scale=min(video_rect.width()/frame_width,video_rect.height()/frame_height)
    render_w,render_h=frame_width*scale,frame_height*scale
    image_rect=QRectF(video_rect.x()+(video_rect.width()-render_w)/2,video_rect.y()+(video_rect.height()-render_h)/2,render_w,render_h)
    x1=clamp(float(bbox.left()),0.0,float(frame_width));y1=clamp(float(bbox.top()),0.0,float(frame_height))
    x2=clamp(float(bbox.right()),x1,float(frame_width));y2=clamp(float(bbox.bottom()),y1,float(frame_height))
    mapped=QRectF(image_rect.left()+x1*scale,image_rect.top()+y1*scale,(x2-x1)*scale,(y2-y1)*scale)
    return mapped.intersected(image_rect)

def clamped_label_rect(box,image_rect,label_width,label_height,gap=2.0):
    width=min(float(label_width),image_rect.width());height=min(float(label_height),image_rect.height())
    x=clamp(box.left(),image_rect.left(),image_rect.right()-width)
    y=box.top()-height-gap if box.top()-height-gap>=image_rect.top() else box.top()
    y=clamp(y,image_rect.top(),image_rect.bottom()-height)
    return QRectF(x,y,width,height)

def compact_unknown_label(value,track_id=None):
    import re
    raw=str(value or track_id or "")
    suffix=raw.rsplit(":",1)[-1];match=re.search(r"(\d+)$",suffix)
    if match:return f"UNK {match.group(1)[-5:]}"
    suffix=suffix.replace("UNKNOWN-","").replace("Unknown-","").replace("UNK-","")
    return f"UNK {suffix[-8:]}".strip()

def active_global_counts(camera_states):
    people={}
    for camera in camera_states:
        if not camera.online:continue
        for track in camera.tracks:
            key=track.global_id or f"{camera.id}:{track.track_id}"
            previous=people.get(key)
            if previous is None or track.known:people[key]=bool(track.known)
    known=sum(people.values());return known,len(people)-known

def unique_overlay_payloads(items):
    """One visual per local track; real detection wins over prediction."""
    output={}
    for raw in items:
        item=dict(raw);local=item.get("local_track_id") or item.get("global_id");key=(local,int(item.get("track_generation") or 1))
        if key is None:continue
        previous=output.get(str(key))
        if previous is None or not (previous.get("observation_type","detected")!="predicted" and item.get("observation_type","detected")=="predicted"):output[str(key)]=item
    return tuple(output.values())

def suspicious_overlay_pairs(tracks,threshold=.70):
    pairs=[]
    for index,left in enumerate(tracks):
        a=left._bbox;aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
        for right in tracks[index+1:]:
            b=right._bbox;inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));union=aa+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter;iou=inter/union if union else 0.0
            if iou>=threshold:pairs.append((left.track_id,right.track_id,iou))
    return tuple(pairs)

def heat_color(v):
    stops = [
        (0.0, (40, 110, 255)),
        (0.3, (0, 200, 200)),
        (0.55, (60, 215, 80)),
        (0.8, (250, 200, 40)),
        (1.0, (255, 70, 40)),
    ]

    v = max(0.0, min(1.0, v))

    for i in range(1, len(stops)):
        if v <= stops[i][0]:
            t0, c0 = stops[i - 1]
            t1, c1 = stops[i]
            f = (v - t0) / (t1 - t0) if t1 > t0 else 0
            return tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))

    return stops[-1][1]


def clamp(v, a, b):
    return max(a, min(b, v))




# ========================= REALTIME CAMERA STATE =====================
class RealtimeTrack:
    def __init__(self,payload,previous=None,metadata_timestamp=0.0):
        self.track_id=payload.get("local_track_id");self.track_generation=int(payload.get("track_generation") or 1);self.visual_key=(str(self.track_id),self.track_generation);self.global_id=payload.get("global_id");self.person_id=payload.get("person_id");self.name=payload.get("display_name") or self.global_id or self.track_id or "Unknown";self.conf=float(payload.get("confidence") or 0.0);self.known=bool(self.person_id);raw=tuple(float(v) for v in payload.get("bbox",(0,0,0,0)));self.metadata_timestamp=float(metadata_timestamp or time.time());self.observation_type=str(payload.get("observation_type") or "detected");self.tracker_state=str(payload.get("tracker_state") or "DETECTED");self.last_detection_timestamp=float(payload.get("last_detection_timestamp") or self.metadata_timestamp);self.prediction_age_ms=float(payload.get("prediction_age_ms") or 0.0);self.identity_version=int(payload.get("identity_version") or 0);self.detection_source=str(payload.get("detection_source") or self.observation_type.upper());self.detection_id=payload.get("detection_id");self.velocity=tuple(float(v) for v in payload.get("velocity",(0.0,0.0,0.0,0.0)));self.state_timestamp=float(payload.get("state_timestamp") or self.metadata_timestamp);self.geometry_monotonic=float(payload.get("geometry_monotonic") or 0.0);self.visual_expires_at=float(payload.get("visual_expires_at") or 0.0);self.visual_visible=bool(payload.get("visual_visible",True));self.boundary_exit=bool(payload.get("boundary_exit",False));self._bbox=raw;self.last_visual_age_before_ms=0.0;self.last_visual_time_error_ms=0.0;self.last_projection_dt_ms=0.0;self.negative_projection=False
    def visible_at(self,display_timestamp):
        return self.visual_visible and (not self.visual_expires_at or display_timestamp is None or float(display_timestamp)<=self.visual_expires_at)
    def bbox(self,width,height,display_timestamp=None):
        x1,y1,x2,y2=self._bbox;target=float(display_timestamp or self.state_timestamp);raw_dt=target-self.state_timestamp;self.last_visual_age_before_ms=max(0.0,raw_dt)*1000.0;self.last_projection_dt_ms=raw_dt*1000.0;self.negative_projection=raw_dt<0;dt=max(-.5,raw_dt)
        if self.visual_expires_at:dt=min(dt,max(0.0,self.visual_expires_at-self.state_timestamp))
        vx,vy,vw,vh=(self.velocity+(0.0,0.0,0.0,0.0))[:4];cx=(x1+x2)*.5+vx*dt;cy=(y1+y2)*.5+vy*dt;w=max(1.0,(x2-x1)+vw*dt);h=max(1.0,(y2-y1)+vh*dt);x1,y1,x2,y2=cx-w*.5,cy-h*.5,cx+w*.5,cy+h*.5
        self.last_visual_time_error_ms=abs(raw_dt-dt)*1000.0 if self.visible_at(target) else abs(target-self.state_timestamp)*1000.0
        x1=max(0,min(width,x1));x2=max(x1,min(width,x2));y1=max(0,min(height,y1));y2=max(y1,min(height,y2));return QRectF(x1,y1,x2-x1,y2-y1)

class CameraState(QObject):
    def __init__(self, cid, name, location, enabled=True, max_people=0):
        super().__init__()
        self.id, self.name, self.location = cid, name, location
        self.enabled, self.online = bool(enabled), False
        self.ai_on, self.heat_on, self.recording = True, False, False
        self.fps, self.res = 0.0, "—"
        self.latency, self.packet_loss, self.infer_ms = 0.0, 0.0, 0.0
        self.frame = None
        self.frame_id = None
        self.frame_timestamp = None
        self.frontend_width,self.frontend_height=0,0
        self.tracks = []
        self._expired_track_versions={};self._latest_generation_by_local={}
        self.camera_local_metadata_drops_total=0;self.stale_metadata_dropped_total=0;self.stale_resurrection_blocked_total=0;self.generation_mismatch_dropped_total=0
        self.negative_projection_dt_total=0;self.projection_dt_ms=deque(maxlen=2000)
        self.visual_age_before_ms=deque(maxlen=2000);self.visual_time_error_ms=deque(maxlen=2000);self.bbox_render_frames=0
        self.metadata_frame_id=-1;self.metadata_timestamp=0.0;self.metadata_frame_width=0;self.metadata_frame_height=0;self.identity_version=0;self.canonicalize_global_id=lambda value:value;self.independent_display_frame_domain=False
        self.receive_fps=0.0;self.render_fps=0.0;self.transport_latency_ms=0.0;self.jpeg_decode_ms=0.0;self.qimage_prepare_ms=0.0;self.gui_schedule_wait_ms=0.0;self.pending_gui_updates=0;self.decoded_frames=0;self.prepared_frames=0;self.replaced_before_render=0;self.qt_render_ms=0.0;self.dropped_display_frames=0;self._last_rendered_frame_id=None;self._render_started=time.monotonic();self._render_frames=0;self._render_ms_total=0.0
        self.surfaces = []
        self.heat = [[0.0] * GW for _ in range(GH)]
        self.hist = [[0.0] * GW for _ in range(GH)]

    def update_surfaces(self):
        live=[surface for surface in self.surfaces if shiboken6.isValid(surface)]
        self.surfaces=live
        for surface in live:surface.update()

    def apply_config(self,data):
        self.name=data.get("name") or self.id;self.location=data.get("location") or ""
        self.enabled=bool(data.get("enabled",data.get("online",False)))
        self.res=str(data.get("resolution") or "—");self.fps=float(data.get("fps") or 0)
        self.heat_on=bool(data.get("heatmap_enabled",False));self.recording=bool(data.get("recording_enabled",False));self.recovery_rois=[dict(item) for item in data.get("recovery_rois",())]

    @property
    def people(self):
        # The backend owns lifecycle; the frontend may only enforce the
        # backend-authorized visual expiry against the exact displayed frame.
        return [track for track in self.tracks if track.visible_at(self.frame_timestamp)]

    def note_render(self,paint_ms=0.0):
        if self.frame_id is None:return
        self._render_frames += 1;self._render_ms_total+=max(0.0,float(paint_ms))
        now = time.monotonic()
        elapsed = now - self._render_started
        if elapsed >= 1.0:
            self.render_fps = self._render_frames / elapsed
            self.qt_render_ms=self._render_ms_total/max(1,self._render_frames)
            self._render_frames = 0;self._render_ms_total=0.0
            self._render_started = now

    @property
    def conn_quality(self):
        return 4 if self.online and self.frame is not None else 0

    def set_metadata(self,message):
        frame_id=int(message.get("frame_id",-1))
        version=int(message.get("metadata_version",-1))
        if not hasattr(self, 'metadata_version'): self.metadata_version = -1
        
        # Use metadata_version for strict causal ordering if available, else fallback to frame_id
        if version > -1:
            if version <= self.metadata_version:
                self.camera_local_metadata_drops_total+=1;self.stale_metadata_dropped_total+=1;return False
            self.metadata_version = version
        else:
            if frame_id<self.metadata_frame_id:
                self.camera_local_metadata_drops_total+=1;self.stale_metadata_dropped_total+=1;return False
        
        if not self.independent_display_frame_domain and self.frame_id is not None and frame_id>int(self.frame_id):return False
        stamp=float(message.get("capture_timestamp") or message.get("timestamp") or time.time());identity_version=int(message.get("identity_version") or 0);previous={track.visual_key:track for track in self.tracks};deduped={};protected_locals=set()
        for raw in unique_overlay_payloads(message.get("tracks",())):
            item=dict(raw);canonical=self.canonicalize_global_id(item.get("global_id"));item["global_id"]=canonical;item["identity_version"]=max(identity_version,int(item.get("identity_version") or 0))
            if not item.get("person_id"):item["display_name"]=canonical
            local=item.get("local_track_id") or canonical
            if local is None:continue
            local=str(local);generation=int(item.get("track_generation") or 1);latest=self._latest_generation_by_local.get(local,0)
            if generation<latest:
                self.generation_mismatch_dropped_total+=1;protected_locals.add(local);continue
            if generation>latest:self._latest_generation_by_local[local]=generation
            key=(local,generation)
            deduped[key]=item
        incoming=set(deduped);previous_ids=set(previous)
        for visual_key in previous_ids-incoming:
            if visual_key[0] not in protected_locals:self._expired_track_versions[visual_key]=max(frame_id,self._expired_track_versions.get(visual_key,-1))
        accepted={}
        for visual_key,item in deduped.items():
            self._expired_track_versions.pop(visual_key,None)
            accepted[visual_key]=item
        updated=[RealtimeTrack(item,previous.get(visual_key),stamp) for visual_key,item in accepted.items()]
        updated_keys={track.visual_key for track in updated}
        self.tracks=updated+[track for key,track in previous.items() if key[0] in protected_locals and key not in updated_keys]
        overlaps=suspicious_overlay_pairs(self.tracks)
        if overlaps:
            by_id={track.track_id:track for track in self.tracks}
            lineage=[{"left":a,"right":b,"iou":round(iou,3),"left_type":by_id[a].observation_type,"right_type":by_id[b].observation_type,"left_source":by_id[a].detection_source,"right_source":by_id[b].detection_source,"left_detection":by_id[a].detection_id,"right_detection":by_id[b].detection_id,"left_global":by_id[a].global_id,"right_global":by_id[b].global_id} for a,b,iou in overlaps]
            log.warning("Suspicious distinct-track overlay overlap camera=%s frame=%s lineage=%s",self.id,frame_id,lineage)
        self.metadata_frame_width=max(0,int(message.get("frame_width") or 0));self.metadata_frame_height=max(0,int(message.get("frame_height") or 0));self.metadata_frame_id=max(self.metadata_frame_id, frame_id);self.metadata_timestamp=stamp;self.identity_version=max(self.identity_version,identity_version);return True

    def clear_frame(self):
        self.online = False
        self.frame = None
        self.frame_id = None
        self.frame_timestamp = None
        self.tracks = []
        self.metadata_frame_id=-1;self.metadata_timestamp=0.0;self.metadata_frame_width=0;self.metadata_frame_height=0;self.identity_version=0

    def heat_image(self):
        import numpy as np
        values=np.asarray(self.heat,dtype=np.float32)
        img=QImage(GW,GH,QImage.Format_RGBA8888);img.fill(Qt.transparent)
        maximum=float(values.max()) if values.size else 0.0
        if maximum <= 0:return img
        for y in range(GH):
            for x in range(GW):
                value=float(values[y,x])/maximum
                if value > .05:
                    r,g,b=heat_color(value);img.setPixelColor(x,y,QColor(r,g,b,min(235,int(60+value*160))))
        return img

    def apply_heatmap(self,snapshot):
        import cv2,numpy as np
        values=np.asarray(snapshot.get("values",[]),dtype=np.float32)
        if values.ndim!=2:return
        self.heat=cv2.resize(values,(GW,GH),interpolation=cv2.INTER_LINEAR).tolist()

    @property
    def known_count(self):return sum(1 for track in self.tracks if track.known)
    @property
    def unknown_count(self):return len(self.tracks)-self.known_count


# ============================ SYSTEM =================================
class PersonRecord:
    def __init__(self, name="", dept="", emp_id="", avatar=None):
        self.name, self.dept, self.emp_id = name, dept, emp_id

        self.status="active";self.last_seen=None;self.rec_count=0;self.avatar=avatar
        self.timeline=[0]*24;self.visited=[];self.stay_total=0;self.history=[]


class System(QObject):
    new_event = Signal(dict)
    heatmap_updated=Signal(str)
    cameras_changed=Signal()

    def __init__(self):
        super().__init__()
        from services.frontend.video_renderer import MetadataBuffer
        self.api=ApiClient();self.async_api=AsyncApi(self.api);self.websocket=WebSocketClient();self.metadata_buffer=MetadataBuffer()
        self.websocket.message.connect(self._on_remote_message);self.websocket.connect()

        self.settings = {
            "det_conf": 0.45,
            "face_th": 0.60,
            "model": "YOLOv11m-pose",
            "retention": 30,
            "sound": True,
        }

        self.sims = []
        self.video_clients = {}
        self.video_base_url=ServiceSettings.from_env().ml_url.rstrip("/")
        self.visitors = {}
        self.usage = {}
        self.identity_aliases = {}
        self.identity_version = 0
        self.identity_runtime_epoch = None

        self.events = []

        self.gpu, self.cpu, self.ram = None, None, None
        self.system_metrics, self.pipeline_metrics = {}, {}
        self._closing = False
        self._event_loop_interval_s = 0.05
        self._event_loop_expected = time.monotonic() + self._event_loop_interval_s
        self._event_loop_lag_ms = deque(maxlen=1200)
        self._event_loop_timer = QTimer(self)
        self._event_loop_timer.setInterval(round(self._event_loop_interval_s * 1000))
        self._event_loop_timer.timeout.connect(self._sample_event_loop)
        self._event_loop_timer.start()

        self.people = []

    def _sample_event_loop(self):
        now = time.monotonic()
        self._event_loop_lag_ms.append(max(0.0, (now - self._event_loop_expected) * 1000.0))
        self._event_loop_expected = now + self._event_loop_interval_s

    def frontend_runtime_metrics(self):
        samples = sorted(self._event_loop_lag_ms)
        def percentile(fraction):
            return samples[min(len(samples) - 1, int((len(samples) - 1) * fraction))] if samples else 0.0
        return {
            "event_loop_lag_ms": {
                "samples": len(samples), "p50": percentile(0.50),
                "p95": percentile(0.95), "max": samples[-1] if samples else 0.0,
            },
            "frontend_cross_camera_mutations_total":0,
            "cameras": {
                camera_id: {**client.runtime_metrics(),**self._camera_visual_metrics(camera_id)}
                for camera_id, client in sorted(self.video_clients.items())
            },
        }

    def _camera_visual_metrics(self,camera_id):
        camera=self.sim_by_id(camera_id);ages=sorted(camera.visual_age_before_ms) if camera else [];errors=sorted(camera.visual_time_error_ms) if camera else [];projection=sorted(camera.projection_dt_ms) if camera else []
        def pct(values,fraction):return values[min(len(values)-1,int((len(values)-1)*fraction))] if values else 0.0
        return {"render_fps":camera.render_fps if camera else 0.0,"bbox_render_frames":camera.bbox_render_frames if camera else 0,"visual_age_before_ms":{"p50":pct(ages,.5),"p95":pct(ages,.95),"max":ages[-1] if ages else 0.0},"visual_time_error_ms":{"p50":pct(errors,.5),"p95":pct(errors,.95),"max":errors[-1] if errors else 0.0},"projection_dt_ms":{"p50":pct(projection,.5),"p95":pct(projection,.95),"max":projection[-1] if projection else 0.0},"negative_projection_dt_total":camera.negative_projection_dt_total if camera else 0,"camera_local_metadata_drops_total":camera.camera_local_metadata_drops_total if camera else 0,"stale_metadata_dropped_total":camera.stale_metadata_dropped_total if camera else 0,"stale_resurrection_blocked_total":camera.stale_resurrection_blocked_total if camera else 0,"generation_mismatch_dropped_total":camera.generation_mismatch_dropped_total if camera else 0}

    def refresh_cameras(self):
        self.async_api.submit(self.api.get_cameras,self._apply_cameras,lambda error:log.warning("Camera API unavailable: %s",error))

    def _apply_cameras(self,rows):
        existing={camera.id:camera for camera in self.sims};ordered=[];wanted=set()
        from services.frontend.video_transport import MJPEGClient
        for data in sorted(rows,key=lambda item:str(item.get("id", ""))):
            camera_id=str(data["id"]);wanted.add(camera_id)
            camera=existing.get(camera_id) or CameraState(camera_id,data.get("name") or camera_id,data.get("location") or "",False)
            camera.apply_config(data);camera.canonicalize_global_id=self.canonicalize_global_id;ordered.append(camera);self.usage.setdefault(camera_id,0)
            client=self.video_clients.get(camera_id)
            if camera.enabled and client is None:
                client=MJPEGClient(camera_id,f"{self.video_base_url}/video/{camera_id}",self)
                client.frame.connect(self._on_video_frame);client.online.connect(self._on_video_status);client.start();self.video_clients[camera_id]=client
            elif not camera.enabled and client is not None:
                client.stop();self.video_clients.pop(camera_id,None);camera.clear_frame()
        for camera_id in set(existing)-wanted:
            client=self.video_clients.pop(camera_id,None)
            if client:client.stop()
            existing[camera_id].clear_frame();self.usage.pop(camera_id,None)
        self.sims=ordered;self.cameras_changed.emit()


    def canonicalize_global_id(self,global_id):
        if global_id is None:return None
        current=str(global_id);seen=set()
        while current in self.identity_aliases and current not in seen:
            seen.add(current);current=self.identity_aliases[current]
        for alias in seen:self.identity_aliases[alias]=current
        return current

    def _accept_identity_epoch(self,message):
        epoch=message.get("identity_runtime_epoch")
        if not epoch:return
        if self.identity_runtime_epoch is not None and epoch!=self.identity_runtime_epoch:
            self.identity_aliases.clear();self.identity_version=0;self.metadata_buffer.clear()
            for camera in self.sims:camera.identity_version=0
        self.identity_runtime_epoch=epoch

    def _on_remote_message(self,message):
        kind=message.get("type","")
        if kind in ("identity.merged","frame.metadata"):self._accept_identity_epoch(message)
        if kind=="identity.merged":
            version=int(message.get("identity_version") or 0)
            if version<self.identity_version:return
            old_id=str(message.get("old_global_id"));canonical=self.canonicalize_global_id(message.get("global_id"));self.identity_aliases[old_id]=canonical;self.identity_version=max(self.identity_version,version)
            for camera in self.sims:
                for track in camera.tracks:
                    resolved=self.canonicalize_global_id(track.global_id)
                    if resolved!=track.global_id:track.global_id=resolved;track.identity_version=self.identity_version
                    if not track.person_id:track.name=resolved
                # Identity text is presented with the next paced camera frame;
                # never trigger a six-card overlay-only repaint burst.
            return
        if kind=="frame.metadata":
            self.metadata_buffer.put(message)
            # Geometry and video must be presented atomically. Applying every
            # prediction message here caused a second, metadata-only repaint in
            # between paced video frames, doubling GUI paint load and making
            # motion look like burst/freeze/overlay tearing. _on_video_frame
            # selects the best camera-local metadata for that exact frame time.
        elif kind=="person.identified":
            self.new_event.emit(message)
        elif kind.startswith("camera."):
            camera_id=message.get("camera_id")
            if kind=="camera.config.changed":self.refresh_cameras();return
            for camera in self.sims:
                if camera.id==camera_id and kind=="camera.offline":camera.clear_frame()
            self.new_event.emit(message)
        elif kind.startswith("enrollment.") or kind.startswith("person."):
            self.new_event.emit(message)
        elif kind=="identity.conflict":
            self.new_event.emit(message)
        elif kind=="heatmap.updated":
            camera=self.sim_by_id(message.get("camera_id"))
            if camera:camera.apply_heatmap(message)
            self.heatmap_updated.emit(message.get("camera_id",""))

    def _on_video_frame(self,camera_id,frame_id,timestamp,image):
        for camera in self.sims:
            if camera.id==camera_id:
                camera.frame=image;camera.online=True;camera.frame_id=frame_id;camera.frame_timestamp=timestamp;camera.frontend_width=image.width();camera.frontend_height=image.height()
                metadata=self.metadata_buffer.match(camera_id,frame_id,timestamp,camera.independent_display_frame_domain)
                if metadata:camera.set_metadata(metadata)
                client=self.video_clients.get(camera_id)
                if client:camera.receive_fps=client.receive_fps;camera.transport_latency_ms=client.transport_latency_ms;camera.jpeg_decode_ms=client.decode_ms;camera.qimage_prepare_ms=client.prepare_ms;camera.gui_schedule_wait_ms=client.gui_schedule_wait_ms;camera.pending_gui_updates=client.pending_gui_updates;camera.decoded_frames=client.decoded_frames;camera.prepared_frames=client.prepared_frames;camera.replaced_before_render=client.replaced_before_render;camera.dropped_display_frames=client.dropped_display_frames
                camera.update_surfaces()
                break

    def _on_video_status(self,camera_id,online):
        for camera in self.sims:
            if camera.id==camera_id:
                if not online and (camera.online or camera.frame is not None):
                    camera.clear_frame();camera.update_surfaces()
                break

    @staticmethod
    def person_record(data):
        record=PersonRecord(data.get("name") or data.get("display_name") or "Unknown",data.get("department") or "",data.get("employee_id") or "")
        record.db_id=data.get("id") or data.get("person_id");record.status=data.get("status","active");record.metadata=data.get("metadata",{})
        return record

    def push_event(self, e, silent=False):
        # Faqat bugungi eventlarni ko'rsatish
        from datetime import datetime as _dt
        today_str = _dt.now().strftime("%Y-%m-%d")
        event_time = e.get("time")
        if event_time:
            if hasattr(event_time, 'strftime'):
                event_date = event_time.strftime("%Y-%m-%d")
            else:
                event_date = str(event_time)[:10]
            if event_date != today_str:
                return  # Bugun emas — o'tkazib yuborish
        e["time"] = datetime.now()

        self.events.insert(0, e)

        if len(self.events) > 500:
            self.events.pop()

        if not silent:
            self.new_event.emit(e)

    def sim_by_id(self, cid):
        for s in self.sims:
            if s.id == cid:
                return s

        return None

    @property
    def cams_online(self):
        return sum(1 for s in self.sims if s.online)

    def shutdown(self):
        if self._closing:return
        self._closing=True;self._event_loop_timer.stop();self.websocket.close()
        for client in self.video_clients.values():client.stop()
        self.video_clients.clear()
        self.async_api.shutdown()


# ========================= BASE WIDGETS ==============================
class GradientBar(QWidget):
    def __init__(self, h, top=True, radius=10):
        super().__init__()

        self.top, self.radius = top, radius

        self.setFixedHeight(h)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4 if top else 2, 12, 2 if top else 6)
        lay.setSpacing(8)

        self.lay = lay

    def paintEvent(self, e):
        p = QPainter(self)

        w, h, r = self.width(), self.height(), self.radius

        path = QPainterPath()

        if self.top:
            path.addRoundedRect(QRectF(0, -r, w, h + r), r, r)
        else:
            path.addRoundedRect(QRectF(0, 0, w, h + r), r, r)

        g = QLinearGradient(0, 0, 0, h)

        if self.top:
            g.setColorAt(0, QColor(6, 9, 13, 220))
            g.setColorAt(1, QColor(6, 9, 13, 0))
        else:
            g.setColorAt(0, QColor(6, 9, 13, 0))
            g.setColorAt(1, QColor(6, 9, 13, 225))

        p.fillPath(path, QBrush(g))

        p.end()


class PulsingDot(QWidget):
    def __init__(self, color="#2ecc71", size=9):
        super().__init__()

        self.base = QColor(color)
        self.sz = size

        self.setFixedSize(size + 6, size + 6)

        self.t = QTimer(self)
        self.t.setInterval(80)
        self.t.timeout.connect(self._tick)
        self.t.start()

    @Slot()
    def _tick(self):
        self.update()

    def stop(self):
        self.t.stop()

    def set_color(self, c):
        value = QColor(c)
        if value == self.base:
            return
        self.base = value
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        ms = QDateTime.currentMSecsSinceEpoch()
        pulse = (math.sin(ms / 300.0) + 1) / 2

        cx, cy = self.width() / 2, self.height() / 2

        c2 = QColor(self.base)
        c2.setAlpha(int(40 * pulse))

        p.setPen(Qt.NoPen)
        p.setBrush(c2)
        p.drawEllipse(QPointF(cx, cy), self.sz / 2 + 3, self.sz / 2 + 3)

        c = QColor(self.base)
        c.setAlpha(int(120 + 135 * pulse))

        p.setBrush(c)
        p.drawEllipse(QPointF(cx, cy), self.sz / 2 + pulse * 1.2, self.sz / 2 + pulse * 1.2)

        p.end()


class VideoSurface(QWidget):
    doubleClicked = Signal()

    def __init__(self, sim, radius=10, face_box=False, zoomable=False):
        super().__init__()

        self.sim, self.radius, self.face_box = sim, radius, face_box

        self.zoomable = zoomable

        self.zoom = 1.0
        self.offset = QPointF(0, 0)

        self.dragging = False
        self.drag_start = QPointF()

        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setMouseTracking(zoomable)

        sim.surfaces.append(self)

        self.destroyed.connect(self._detach_surface)

    @Slot()
    def _detach_surface(self, *_args):
        if self in self.sim.surfaces:self.sim.surfaces.remove(self)

    def dispose(self):
        if self in self.sim.surfaces:self.sim.surfaces.remove(self)

    def wheelEvent(self, e):
        if not self.zoomable:
            return

        f = 1.12 if e.angleDelta().y() > 0 else 1 / 1.12

        self.zoom = clamp(self.zoom * f, 1.0, 4.0)

        if self.zoom <= 1.01:
            self.zoom = 1.0
            self.offset = QPointF(0, 0)

        self.setCursor(Qt.OpenHandCursor if self.zoom > 1 else Qt.ArrowCursor)

        self.update()

    def mousePressEvent(self, e):
        if self.zoomable and self.zoom > 1:
            self.dragging = True
            self.drag_start = e.position() - self.offset
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self.dragging:
            self.offset = e.position() - self.drag_start
            self.update()

    def mouseReleaseEvent(self, e):
        self.dragging = False
        self.setCursor(Qt.OpenHandCursor if self.zoom > 1 else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if self.zoomable:
            self.zoom = 1.0
            self.offset = QPointF(0, 0)
            self.update()
        else:
            self.doubleClicked.emit()

    def video_rect(self):
        frame = self.sim.frame
        fw = frame.width() if frame is not None else FRAME_W
        fh = frame.height() if frame is not None else FRAME_H
        return aspect_fit_rect(self.width(), self.height(), fw, fh)

    def paintEvent(self, e):
        paint_started=time.perf_counter();p = QPainter(self)

        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.fillRect(self.rect(), QColor("#05070a"))

        f = self.sim.frame

        if f is None:
            p.setPen(QColor(TH.ERR));p.setFont(QFont("Segoe UI",15,QFont.Bold))
            p.drawText(self.rect().adjusted(0,-16,0,0),Qt.AlignCenter,"⚠ NO SIGNAL")
            p.setPen(QColor(TH.DIM));p.setFont(QFont("Segoe UI",9))
            p.drawText(self.rect().adjusted(0,22,0,0),Qt.AlignCenter,"OFFLINE")
            p.end()
            return

        if self.radius:
            path = QPainterPath()
            path.addRoundedRect(QRectF(self.rect()), self.radius, self.radius)
            p.setClipPath(path)

        p.save()

        if self.zoomable and self.zoom > 1.0:
            p.translate(self.width() / 2, self.height() / 2)
            p.scale(self.zoom, self.zoom)
            p.translate(
                -self.width() / 2 + self.offset.x() / self.zoom,
                -self.height() / 2 + self.offset.y() / self.zoom,
            )

        vr = self.video_rect()

        p.drawImage(vr, f)

        s = self.sim

        if s.online:
            if s.heat_on:
                pm = QPixmap.fromImage(s.heat_image()).scaled(
                    int(vr.width()),
                    int(vr.height()),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )

                p.setOpacity(0.6)
                p.drawPixmap(vr.toRect(), pm)
                p.setOpacity(1.0)

            if s.ai_on:
                source_w=s.metadata_frame_width or f.width();source_h=s.metadata_frame_height or f.height()
                self._draw_ai(p, vr, source_w, source_h)

        p.restore()

        if self.zoomable and self.zoom > 1.01:
            p.setPen(QColor(TH.TXT))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))

            p.drawText(
                self.rect().adjusted(0, 8, -12, 0),
                Qt.AlignRight,
                f"🔍 {self.zoom:.1f}×  (scroll · drag · dbl-click reset)",
            )

        p.end()
        self.sim.note_render((time.perf_counter()-paint_started)*1000)

    def _map(self, vr, fw, fh, bb):
        return map_bbox_to_video_rect(vr, fw, fh, bb)

    def _draw_ai(self, p, vr, fw, fh):
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setClipRect(vr,Qt.IntersectClip)

        font = QFont("Segoe UI", 8.5, QFont.Bold)
        p.setFont(font)



        for ps in self.sim.people:
            r=self._map(vr,fw,fh,ps.bbox(fw,fh,self.sim.frame_timestamp));self.sim.projection_dt_ms.append(ps.last_projection_dt_ms)
            if ps.negative_projection:self.sim.negative_projection_dt_total+=1
            self.sim.visual_age_before_ms.append(ps.last_visual_age_before_ms);self.sim.visual_time_error_ms.append(ps.last_visual_time_error_ms);self.sim.bbox_render_frames+=1

            col = QColor(TH.OK) if ps.known else QColor("#f59e42")

            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 2))

            L = min(r.width(), r.height()) * 0.22

            for x1, y1, x2, y2, x3, y3 in [
                (r.left(), r.top() + L, r.left(), r.top(), r.left() + L, r.top()),
                (r.right() - L, r.top(), r.right(), r.top(), r.right(), r.top() + L),
                (r.right(), r.bottom() - L, r.right(), r.bottom(), r.right() - L, r.bottom()),
                (r.left() + L, r.bottom(), r.left(), r.bottom(), r.left(), r.bottom() - L),
            ]:
                p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                p.drawLine(QPointF(x2, y2), QPointF(x3, y3))

            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), 70), 1))
            p.drawRect(r)

            if ps.known:
                display_name = str(ps.name)
            else:
                raw_id = str(ps.name or ps.track_id)
                if ":" in raw_id:
                    raw_id = raw_id.rsplit(":", 1)[-1]
                if raw_id.upper().startswith("UNK-"):
                    raw_id = raw_id[4:]
                elif raw_id.lower().startswith("unknown-"):
                    raw_id = raw_id[8:]
                display_name = compact_unknown_label(raw_id,ps.track_id)
            lbl = display_name

            fm = QFontMetrics(font)

            tw = fm.horizontalAdvance(lbl) + 10
            lh = fm.height() + 6

            label_rect=clamped_label_rect(r,vr,tw,lh)

            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 215))
            p.drawRoundedRect(label_rect,4,4)

            p.setPen(QColor("#0c1116"))
            p.drawText(label_rect.adjusted(5,0,-5,0),Qt.AlignVCenter,lbl)

        p.restore()



# ============================ HEADER =================================
class Chip(QFrame):
    def __init__(self, label, icon=""):
        super().__init__()

        self.setObjectName("chip")

        h = QHBoxLayout(self)
        h.setContentsMargins(9, 4, 9, 4)
        h.setSpacing(6)

        t = QLabel(f"{icon} {label}".strip())
        t.setStyleSheet(f"color:{TH.DIM}; font-size:9px; font-weight:700;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFixedSize(36, 4)
        self.bar.setTextVisible(False)

        self.val = QLabel("--")
        self.val.setStyleSheet("font-size:10px; font-weight:700;")

        h.addWidget(t)
        h.addWidget(self.bar)
        h.addWidget(self.val)

    def set_value(self, v):
        if v is None:
            self.val.setText("—")
            self.bar.setValue(0)
            return
        self.val.setText(f"{int(v)}%")
        self.bar.setValue(int(v))

        c = TH.ERR if v > 85 else (TH.WARN if v > 65 else TH.OK)

        self.bar.setStyleSheet(
            f"QProgressBar{{background:#232c37;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{c};border-radius:2px;}}"
        )


class NotificationDropdown(QFrame):
    def __init__(self, mw):
        super().__init__(mw)

        self.mw = mw

        self.setFixedWidth(340)
        self.setFixedHeight(310)

        self.setObjectName("notifDrop")

        self.hide()

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)

        row = QHBoxLayout()

        t = QLabel("🔔 Notifications")
        t.setStyleSheet("font-size:12px; font-weight:800; color:white;")

        clr = QPushButton("Mark all read")
        clr.setObjectName("btnGhost")
        clr.setFixedHeight(24)
        clr.setCursor(Qt.PointingHandCursor)
        clr.clicked.connect(self.clear_all)

        row.addWidget(t)
        row.addStretch(1)
        row.addWidget(clr)

        v.addLayout(row)

        self.box = QVBoxLayout()
        self.box.setSpacing(4)
        self.box.addStretch(1)

        v.addLayout(self.box, 1)

        self.items = []

    def add_alert(self, e):
        if e.get("level") not in ("warn", "err"):
            return

        w = QFrame()
        w.setObjectName("notifItem")

        h = QHBoxLayout(w)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(8)

        ic = QLabel("⚠️" if e["level"] == "warn" else "🚨")

        tx = QLabel(f"{e['cam']} — {e['person']}")
        tx.setStyleSheet(f"font-size:10px; color:{TH.TXT};")

        tm = QLabel(e["time"].strftime("%H:%M:%S"))
        tm.setStyleSheet(f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;")

        h.addWidget(ic)
        h.addWidget(tx, 1)
        h.addWidget(tm)

        w.setCursor(Qt.PointingHandCursor)

        w.mousePressEvent = lambda ev, _=w: (self.mw.navigate("events"), self.hide())

        self.box.insertWidget(0, w)
        self.items.insert(0, w)

        while len(self.items) > 8:
            old = self.items.pop()
            old.deleteLater()

    def clear_all(self):
        for w in self.items:
            w.deleteLater()

        self.items = []

        self.mw.header._unseen = 0
        self.mw.header.badge.hide()

        self.hide()

    def toggle(self):
        if self.isVisible():
            self.hide()
            return

        gp = self.mw.header.bell.mapToGlobal(QPoint(0, self.mw.header.bell.height()))
        lp = self.mw.mapFromGlobal(gp)

        self.move(lp.x() - self.width() + self.mw.header.bell.width(), lp.y() + 4)

        self.show()
        self.raise_()


class Header(QFrame):
    def __init__(self, sys_, mw):
        super().__init__()

        self.setObjectName("header")

        self.sys, self.mw = sys_, mw

        self.setFixedHeight(64)

        h = QHBoxLayout(self)
        h.setContentsMargins(14, 0, 14, 0)
        h.setSpacing(10)

        logo = QLabel("◉")
        logo.setFixedSize(34, 34)
        logo.setStyleSheet(
            f"background:{TH.ACCENT}; border-radius:9px; color:white;"
            "font-size:16px; font-weight:800;"
        )
        logo.setAlignment(Qt.AlignCenter)

        tt = QVBoxLayout()
        tt.setSpacing(0)

        t1 = QLabel("AI Surveillance System")
        t1.setStyleSheet("font-size:13.5px; font-weight:800; color:white;")

        t2 = QLabel("Operator Console")
        t2.setStyleSheet(f"font-size:9px; color:{TH.DIM};")

        tt.addWidget(t1)
        tt.addWidget(t2)

        h.addWidget(logo)
        h.addLayout(tt)
        h.addSpacing(8)
        h.addStretch(1)


        # h.addWidget(self.search)  # ✅ Search removed
        h.addStretch(1)

        self.chip_gpu = Chip("GPU")
        self.chip_cpu = Chip("CPU")
        self.chip_ram = Chip("RAM")

        for c in (self.chip_gpu, self.chip_cpu, self.chip_ram):
            h.addWidget(c)

        self.ai_chip = QLabel("● AI LOADING…")
        self.ai_chip.setStyleSheet(
            f"color:{TH.WARN}; font-size:10px; font-weight:800;"
            f"background:#24210f; border:1px solid #443d1a;"
            "border-radius:10px; padding:4px 10px;"
        )

        h.addWidget(self.ai_chip)

        self.cam_chip = QLabel("🎥 0/0")
        self.cam_chip.setStyleSheet(
            f"color:{TH.TXT}; font-size:10px; font-weight:700;"
            f"background:{TH.CARD2}; border:1px solid {TH.BORDER};"
            "border-radius:10px; padding:4px 10px;"
        )

        h.addWidget(self.cam_chip)

        self.bell = QPushButton("🔔")
        self.bell.setFixedSize(34, 34)
        self.bell.setObjectName("bellBtn")
        self.bell.setCursor(Qt.PointingHandCursor)

        self.badge = QLabel("0", self.bell)
        self.badge.setFixedSize(16, 16)
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(
            f"background:{TH.ERR}; color:white; border-radius:8px;"
            "font-size:8.5px; font-weight:800;"
        )
        self.badge.move(17, 1)
        self.badge.hide()

        self._unseen = 0

        self.bell.clicked.connect(lambda: mw.notif.toggle())

        h.addWidget(self.bell)

        self.clock = QLabel()
        self.clock.setStyleSheet(
            f"color:{TH.TXT}; font-size:11px; font-family:Consolas,monospace;"
        )

        h.addWidget(self.clock)

        av = QLabel("OP")
        av.setFixedSize(30, 30)
        av.setAlignment(Qt.AlignCenter)
        av.setStyleSheet(
            f"background:{TH.ACCENT}; color:white; border-radius:15px;"
            "font-size:10px; font-weight:800;"
        )

        h.addWidget(av)

        un = QLabel("Operator")
        un.setStyleSheet(f"font-size:10.5px; color:{TH.DIM}; font-weight:600;")

        h.addWidget(un)

        self.tick_clock()

    def set_ai_ready(self):
        self.ai_chip.setText("● AI ACTIVE")
        self.ai_chip.setStyleSheet(
            f"color:{TH.OK}; font-size:10px; font-weight:800;"
            f"background:#16241c; border:1px solid #234433;"
            "border-radius:10px; padding:4px 10px;"
        )

    def bump(self):
        self._unseen += 1

        self.badge.setText(str(min(99, self._unseen)))
        self.badge.show()

    def tick_clock(self):
        self.clock.setText(datetime.now().strftime("%a %d %b %Y   %H:%M:%S"))

    def update_stats(self):
        self.chip_gpu.set_value(self.sys.gpu)
        self.chip_cpu.set_value(self.sys.cpu)
        self.chip_ram.set_value(self.sys.ram)

        n, tot = self.sys.cams_online, len(self.sys.sims)

        self.cam_chip.setText(f"🎥 {n}/{tot}")

        self.cam_chip.setStyleSheet(
            f"color:{TH.TXT if n == tot else TH.WARN}; font-size:10px; font-weight:700;"
            f"background:{TH.CARD2}; border:1px solid {TH.BORDER};"
            "border-radius:10px; padding:4px 10px;"
        )


# ============================ SIDEBAR ================================
class SideBar(QFrame):
    changed = Signal(str)

    ITEMS = [
        ("live", "🎥", "Dashboard"),
        ("people", "👥", "Person Management"),
        ("enroll", "🪪", "Enrollment"),
        # ("analytics", "📈", "Analytics"),  # ✅ Analytics nav removed
        ("events", "⚡", "Events"),
        ("settings", "⚙️", "Settings"),
    ]

    def __init__(self, mw):
        super().__init__()

        self.setObjectName("sidebar")

        self.mw = mw

        self.buttons = {}

        self.setMinimumWidth(210)
        self.setMaximumWidth(210)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 14, 10, 10)
        v.setSpacing(4)

        brand = QLabel("CONTROL PANEL")
        brand.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-weight:800;"
            "letter-spacing:2px; padding:0 0 6px 10px;"
        )

        v.addWidget(brand)

        for key, icon, text in self.ITEMS:
            b = QPushButton(f"  {icon}   {text}")

            b.setObjectName("sideBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(42)
            b.setToolTip(text)

            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))

            self.buttons[key] = (b, icon, text)

            v.addWidget(b)

        v.addStretch(1)

        self.collapse = QPushButton("⟨  Collapse")
        self.collapse.setObjectName("sideBtn")
        self.collapse.setFixedHeight(38)
        self.collapse.setCursor(Qt.PointingHandCursor)
        self.collapse.clicked.connect(mw.toggle_sidebar)

        v.addWidget(self.collapse)

        self.collapsed = False

    def set_active(self, key):
        for k, (b, _, _) in self.buttons.items():
            b.setChecked(k == key)

    def set_collapsed(self, c):
        self.collapsed = c

        for k, (b, icon, text) in self.buttons.items():
            b.setText(icon if c else f"  {icon}   {text}")
            b.setProperty("collapsed", c)
            b.style().unpolish(b)
            b.style().polish(b)

        self.collapse.setText("⟩" if c else "⟨  Collapse")


# ========================= RIGHT PANEL ===============================
class RightPanel(QFrame):
    def __init__(self, sys_):
        super().__init__()

        self.setObjectName("rightPanel")

        self.sys = sys_

        self.setMinimumWidth(250)
        self.setMaximumWidth(310)

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 14, 12, 10)
        v.setSpacing(10)

        t = QLabel("LIVE STATUS")
        t.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-weight:800; letter-spacing:2px;"
        )

        v.addWidget(t)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.known_card = self._stat("👤 KNOWN", TH.OK)
        self.unk_card = self._stat("❓ UNKNOWN", "#f59e42")

        row.addWidget(self.known_card[0])
        row.addWidget(self.unk_card[0])

        v.addLayout(row)

        v.addWidget(self._sep("SYSTEM"))

        self.b_gpu = self._meter("GPU")
        self.b_cpu = self._meter("CPU")
        self.b_fps = self._meter("AI")

        for w in (self.b_gpu[0], self.b_cpu[0], self.b_fps[0]):
            v.addWidget(w)
        self.metric_detail=QLabel("—");self.metric_detail.setWordWrap(True);self.metric_detail.setStyleSheet(f"color:{TH.DIM};font-size:8.5px;font-family:Consolas,monospace;");v.addWidget(self.metric_detail)

        v.addWidget(self._sep("ALERTS"))

        self.alerts = QVBoxLayout()
        self.alerts.setSpacing(5)

        v.addLayout(self.alerts)

        v.addWidget(self._sep("RECENT EVENTS"))

        self.evbox = QVBoxLayout()
        self.evbox.setSpacing(2)
        self.evbox.addStretch(1)

        wrap = QWidget()
        wrap.setStyleSheet("background:transparent;")
        wrap.setLayout(self.evbox)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(wrap)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setStyleSheet("QScrollArea{border:none;background:transparent;} QScrollArea > QWidget > QWidget{background:transparent;}")

        v.addWidget(sc, 1)

        self._alert_items, self._ev_items = [], []

    def _sep(self, txt):
        l = QLabel(txt)

        l.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8px; font-weight:800;"
            "letter-spacing:1.5px; padding-top:6px;"
        )

        return l

    def _stat(self, label, color):
        f = QFrame()
        f.setObjectName("statCard")

        lay = QVBoxLayout(f)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(2)

        num = QLabel("0")
        num.setStyleSheet(f"color:{color}; font-size:20px; font-weight:800;")

        cap = QLabel(label)
        cap.setStyleSheet(f"color:{TH.DIM}; font-size:8.5px; font-weight:700;")

        lay.addWidget(num)
        lay.addWidget(cap)

        return (f, num)

    def _meter(self, label):
        w = QWidget()

        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        l = QLabel(label)
        l.setFixedWidth(28)
        l.setStyleSheet(f"color:{TH.DIM}; font-size:9px; font-weight:700;")

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)

        val = QLabel("--")
        val.setFixedWidth(44)
        val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val.setStyleSheet(f"color:{TH.TXT}; font-size:9.5px; font-family:Consolas,monospace;")

        h.addWidget(l)
        h.addWidget(bar, 1)
        h.addWidget(val)

        return (w, bar, val)

    def refresh(self):
        sims = [s for s in self.sys.sims if s.online]

        known,unknown=active_global_counts(sims)
        self.known_card[1].setText(str(known))
        self.unk_card[1].setText(str(unknown))

        for (bar, val), v in [
            ((self.b_gpu[1], self.b_gpu[2]), self.sys.gpu),
            ((self.b_cpu[1], self.b_cpu[2]), self.sys.cpu),
        ]:
            bar.setValue(int(v or 0))
            val.setText(f"{int(v)}%" if v is not None else "—")

            c = TH.ERR if v is not None and v > 85 else (TH.WARN if v is not None and v > 65 else TH.ACCENT)

            bar.setStyleSheet(
                f"QProgressBar{{background:#232c37;border-radius:3px;}}"
                f"QProgressBar::chunk{{background:{c};border-radius:3px;}}"
            )

        batch_rate = self.sys.pipeline_metrics.get("batch_rate")

        self.b_fps[1].setValue(min(100, int((batch_rate or 0) / 10 * 100)))
        self.b_fps[2].setText(f"{batch_rate:.1f}/s" if batch_rate is not None else "—")
        self.b_fps[2].setStyleSheet(
            f"color:{TH.OK}; font-size:9.5px; font-family:Consolas,monospace;"
        )
        system=self.sys.system_metrics;profile=self.sys.pipeline_metrics.get("detector_profile",{});detector=(profile.get("pure_detector_wall") or {}).get("p50");used=system.get("gpu_memory_used_mb");total=system.get("gpu_memory_total_mb");temp=system.get("gpu_temperature_c");nvdec=system.get("nvdec_utilization_percent");ram_used=system.get("ram_used_bytes");ram_total=system.get("ram_total_bytes")
        gpu_text=f"VRAM {used/1024:.1f}/{total/1024:.1f}GB  {temp:.0f}C  NVDEC {nvdec:.0f}%" if None not in (used,total,temp,nvdec) else "VRAM/TEMP/NVDEC —"
        ram_text=f"RAM {ram_used/1073741824:.1f}/{ram_total/1073741824:.1f}GB" if None not in (ram_used,ram_total) else "RAM —"
        ai_text=f"AI detector p50 {detector:.0f}ms  CAM {self.sys.cams_online}/{len(self.sys.sims)}" if detector is not None else f"AI —  CAM {self.sys.cams_online}/{len(self.sys.sims)}"
        self.metric_detail.setText("\n".join((gpu_text,ram_text,ai_text)))

    def add_event(self, e):
        e=dict(e);e.setdefault("cam",e.get("camera_id") or "SYSTEM");e.setdefault("person",e.get("person_name") or e.get("name") or e.get("global_id") or e.get("event_type") or e.get("type") or "Event")
        # Normalize timestamp - handle both datetime objects and ISO string timestamps
        ts = e.get("time")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                e["time"] = ts
            except ValueError:
                ts = None
        if ts is None or not hasattr(ts, "strftime"):
            ts = datetime.now()
            e["time"] = ts

        lvl = e.get("level", "info")

        if lvl in ("warn", "err"):
            col = TH.WARN if lvl == "warn" else TH.ERR

            f = QFrame()
            f.setObjectName("alertItem")
            f.setStyleSheet(
                f"background:{TH.CARD}; border-left:3px solid {col};"
                "border-radius:6px; padding:2px;"
            )

            h = QHBoxLayout(f)
            h.setContentsMargins(8, 6, 8, 6)
            h.setSpacing(6)

            tx = QLabel(f"{e['cam']} — {e['person']}")
            tx.setStyleSheet(f"font-size:9.5px; color:{TH.TXT};")

            tm = QLabel(ts.strftime("%H:%M:%S"))
            tm.setStyleSheet(
                f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;"
            )

            h.addWidget(tx, 1)
            h.addWidget(tm)

            self.alerts.insertWidget(0, f)
            self._alert_items.insert(0, f)

            while len(self._alert_items) > 3:
                old = self._alert_items.pop()
                old.deleteLater()

        dotc = {
            "ok": TH.OK,
            "warn": TH.WARN,
            "err": TH.ERR,
            "info": TH.ACCENT,
        }.get(lvl, TH.DIM)

        w = QWidget()

        h = QHBoxLayout(w)
        h.setContentsMargins(2, 3, 2, 3)
        h.setSpacing(6)

        tm = QLabel(ts.strftime("%H:%M:%S"))
        tm.setStyleSheet(
            f"color:{TH.FAINT}; font-size:8.5px; font-family:Consolas,monospace;"
        )

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{dotc}; font-size:7px;")

        tx = QLabel(f"{e['cam']} · {e['person']}")
        tx.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px;")

        h.addWidget(tm)
        h.addWidget(dot)
        h.addWidget(tx, 1)

        self.evbox.insertWidget(0, w)
        self._ev_items.insert(0, w)

        while len(self._ev_items) > 25:
            old = self._ev_items.pop()
            old.deleteLater()


# ========================== CAMERA CARD ==============================
class QuickInfo(QFrame):
    def __init__(self, sim):
        super().__init__()

        self.sim = sim

        self.setObjectName("quickInfo")

        self.setFixedWidth(210)

        g = QGridLayout(self)
        g.setContentsMargins(12, 10, 12, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        t = QLabel("QUICK INFO")
        t.setStyleSheet(
            f"color:{TH.ACC2}; font-size:8px; font-weight:800; letter-spacing:1.5px;"
        )

        g.addWidget(t, 0, 0, 1, 2)

        self.vals = {}

        rows = [
            ("camera", "Camera"),
            ("status", "Status"),
            ("fps", "Current FPS"),
            ("people", "Detected People"),
            ("split", "Known / Unknown"),
            ("rec", "Recording"),
            ("ai", "AI Status"),
            ("lat", "Latency"),
            ("loss", "Packet Loss"),
            ("infer", "AI Inference"),
        ]

        for i, (key, label) in enumerate(rows, start=1):
            l = QLabel(label)
            l.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px;")

            v = QLabel("--")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(f"color:{TH.TXT}; font-size:9.5px; font-weight:700;")

            g.addWidget(l, i, 0)
            g.addWidget(v, i, 1)

            self.vals[key] = v

    def refresh(self):
        s = self.sim
        on = s.online

        self.vals["camera"].setText(f"{s.id} — {s.name}")

        self.vals["status"].setStyleSheet(
            f"color:{TH.OK if on else TH.ERR}; font-size:9.5px; font-weight:700;"
        )

        self.vals["fps"].setText(f"{s.fps:.1f}" if on else "—")

        self.vals["people"].setText(str(len(s.people)) if on else "—")

        self.vals["split"].setText(f"{s.known_count} / {s.unknown_count}" if on else "—")

        self.vals["rec"].setText("🔴 REC" if s.recording and on else "OFF")

        self.vals["ai"].setText("ON" if s.ai_on else "OFF")
        self.vals["ai"].setStyleSheet(
            f"color:{TH.OK if s.ai_on else TH.FAINT}; font-size:9.5px; font-weight:700;"
        )

        self.vals["lat"].setText(f"{s.latency:.0f} ms" if on else "—")
        self.vals["loss"].setText(f"{s.packet_loss:.1f} %" if on else "—")
        self.vals["infer"].setText(f"{s.infer_ms:.0f} ms" if on else "—")


class CameraCard(QFrame):
    def __init__(self, sim, hub):
        super().__init__()

        self.setObjectName("camCard")

        self.sim, self.hub = sim, hub
        self._disposed=False
        self._last_offline=None

        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(0)
        self.glow.setColor(QColor(TH.ACCENT))
        self.glow.setOffset(0, 0)

        self.setGraphicsEffect(self.glow)

        self._glow_anim = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.surface = VideoSurface(sim)

        lay.addWidget(self.surface)

        self.top = GradientBar(32, top=True)
        self.top.setParent(self)

        self.lbl_id = QLabel(sim.id)
        self.lbl_id.setStyleSheet("color:white; font-size:11px; font-weight:800;")

        self.lbl_loc = QLabel(f"· {sim.location}")
        self.lbl_loc.setStyleSheet(f"color:{TH.DIM}; font-size:10px;")

        self.dot = PulsingDot(TH.OK)


        self.top.lay.addWidget(self.lbl_id)
        self.top.lay.addWidget(self.lbl_loc)
        self.top.lay.addStretch(1)
        self.top.lay.addWidget(self.dot)

        self.bot = GradientBar(30, top=False)
        self.bot.setParent(self)

        self.lbl_fps = QLabel()
        self.lbl_res = QLabel(sim.res)
        self.lbl_ppl = QLabel()
        self.lbl_ai = QLabel()
        self.lbl_conn = QLabel()

        for l in (self.lbl_fps, self.lbl_res, self.lbl_ppl, self.lbl_ai, self.lbl_conn):
            l.setStyleSheet(f"color:{TH.DIM}; font-size:9.5px; font-weight:600;")
            self.bot.lay.addWidget(l)

        self.bot.lay.addStretch(1)

        self.lbl_rec = QLabel("● REC")
        self.lbl_rec.setStyleSheet(f"color:{TH.ERR}; font-size:9px; font-weight:800;")

        self.bot.lay.addWidget(self.lbl_rec)

        self.rec_t = QTimer(self)
        self.rec_t.setInterval(600)
        self.rec_t.timeout.connect(self._toggle_recording)
        self.rec_t.start()

        self.tb = QFrame(self)
        self.tb.setObjectName("camToolbar")

        th = QHBoxLayout(self.tb)
        th.setContentsMargins(4, 4, 4, 4)
        th.setSpacing(4)

        def tool(txt, tip, check=False):
            b = QToolButton()

            b.setText(txt)
            b.setToolTip(tip)
            b.setObjectName("camTool")
            b.setFixedSize(30, 30)
            b.setCheckable(check)
            b.setCursor(Qt.PointingHandCursor)

            th.addWidget(b)

            return b

        self.btn_ai = tool("🤖", "AI Overlay", True)
        self.btn_heat = tool("🔥", "Heatmap", True)
        self.btn_snap = tool("📸", "Snapshot")
        self.btn_full = tool("⛶", "Fullscreen")
        self.btn_more = tool("⋮", "More")

        self.tb.hide()

        self.btn_ai.toggled.connect(lambda c: self._set_ai(c))
        self.btn_heat.toggled.connect(lambda c: self._set_heat(c))
        self.btn_snap.clicked.connect(lambda: hub.snapshot(sim))
        self.btn_full.clicked.connect(lambda: hub.open_fullscreen(sim))
        self.btn_more.clicked.connect(self._menu)
        self.hub.sys.heatmap_updated.connect(self._on_heatmap_updated)

        self.qi = QuickInfo(sim)
        self.qi.setParent(self)
        self.qi.hide()

        self.surface.doubleClicked.connect(lambda: hub.open_fullscreen(sim))

        self.refresh()

    @Slot()
    def _toggle_recording(self):
        self.lbl_rec.setVisible(not self.lbl_rec.isVisible())

    @Slot(str)
    def _on_heatmap_updated(self,camera_id):
        if not self._disposed and camera_id==self.sim.id and self.sim.heat_on:self.refresh()

    def dispose(self):
        if self._disposed:return
        self._disposed=True;self.rec_t.stop();self.dot.stop();self.surface.dispose()
        try:self.hub.sys.heatmap_updated.disconnect(self._on_heatmap_updated)
        except (RuntimeError,TypeError):pass

    def deleteLater(self):
        self.dispose();super().deleteLater()

    def closeEvent(self,event):
        self.dispose();super().closeEvent(event)

    def _set_ai(self, c):
        self.sim.ai_on = c
        self.refresh()

    def _set_heat(self, c):
        self.sim.heat_on = c
        if c:self._load_heatmap()
        self.refresh()

    def _load_heatmap(self):
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.get_heatmap(self.sim.id,"live"),lambda snapshot:(self.sim.apply_heatmap(snapshot),self.refresh()),lambda error:self.hub.toast(f"Heatmap API: {error}"),owner=self)

    def _menu(self):
        m = QMenu(self)

        a1 = m.addAction("📷   Camera details")
        a4 = m.addAction("⎘   Copy Camera ID")

        act = m.exec(QCursor.pos())

        s = self.sim

        if act == a1:
            self.qi.setVisible(not self.qi.isVisible())

        elif act == a4:
            QApplication.clipboard().setText(s.id)

    def resizeEvent(self, e):
        if hasattr(self, "top"):
            self.top.setGeometry(0, 0, self.width(), 32)
            self.bot.setGeometry(0, self.height() - 30, self.width(), 30)

            self.tb.move(self.width() - self.tb.sizeHint().width() - 8, 38)

            self.qi.move(
                self.width() - self.qi.width() - 8,
                38 + self.tb.height() + 6,
            )

        super().resizeEvent(e)

    def enterEvent(self, e):
        self.tb.show()
        self.qi.show()
        self.qi.refresh()

        a = QPropertyAnimation(self.glow, b"blurRadius")
        a.setDuration(150)
        a.setEndValue(18)
        a.start()

        self._glow_anim = a

    def leaveEvent(self, e):
        self.tb.hide()
        self.qi.hide()

        a = QPropertyAnimation(self.glow, b"blurRadius")
        a.setDuration(150)
        a.setEndValue(0)
        a.start()

        self._glow_anim = a

    def refresh(self):
        s = self.sim
        on = s.online
        self.dot.set_color(TH.OK if on else TH.ERR)
        def text_if_changed(label,value):
            if label.text()!=value:label.setText(value)
        def style_if_changed(widget,value):
            if widget.styleSheet()!=value:widget.setStyleSheet(value)
        text_if_changed(self.lbl_fps,f"{s.receive_fps:.1f} FPS" if on else "-- FPS")
        text_if_changed(self.lbl_ppl,f"👥 {len(s.people)}" if on else "👥 —")
        text_if_changed(self.lbl_ai,"🤖 AI ON" if s.ai_on else "🤖 AI OFF")
        style_if_changed(self.lbl_ai,f"color:{TH.OK if s.ai_on else TH.FAINT}; font-size:9.5px; font-weight:700;")
        q = s.conn_quality
        bars = "▂▄▆█"
        text_if_changed(self.lbl_conn,bars[:q] + "░" * (4 - q) if on else "░░░░")
        style_if_changed(self.lbl_conn,f"color:{TH.OK if q >= 3 else (TH.WARN if q >= 2 else TH.ERR)}; font-size:9px;")
        offline=not on
        if offline!=self._last_offline:
            self._last_offline=offline;self.setProperty("offline",offline);self.style().unpolish(self);self.style().polish(self)
        for b, v in ((self.btn_ai, s.ai_on), (self.btn_heat, s.heat_on)):
            if b.isChecked()!=v:
                b.blockSignals(True);b.setChecked(v);b.blockSignals(False)
        if self.qi.isVisible():self.qi.refresh()


# =========================== FULLSCREEN ==============================
class FullscreenCam(QDialog):
    def __init__(self, sim, hub):
        super().__init__(None)

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background:#000;")

        self.sim, self.hub = sim, hub
        self._disposed = False

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        top = QFrame()
        top.setFixedHeight(56)
        top.setStyleSheet(
            f"background:{TH.PANEL}; border-bottom:1px solid {TH.BORDER};"
        )

        h = QHBoxLayout(top)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(10)

        self.lbl_id = QLabel(sim.id)
        self.lbl_id.setStyleSheet("font-size:15px; font-weight:800; color:white;")

        self.lbl_loc = QLabel(f"{sim.name} — {sim.location}")
        self.lbl_loc.setStyleSheet(f"color:{TH.DIM}; font-size:11px;")

        self.lbl_st = QLabel()

        h.addWidget(self.lbl_id)
        h.addWidget(self.lbl_loc)
        h.addStretch(1)
        h.addWidget(self.lbl_st)

        back = QPushButton("← Back")
        back.setObjectName("btnGhost")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(self.accept)

        h.addWidget(back)

        v.addWidget(top)

        self.surface = VideoSurface(sim, radius=0, zoomable=True)

        v.addWidget(self.surface, 1)

        bot = QFrame()
        bot.setFixedHeight(64)
        bot.setStyleSheet(f"background:{TH.PANEL}; border-top:1px solid {TH.BORDER};")

        bh = QHBoxLayout(bot)
        bh.setSpacing(8)

        def fbtn(txt, tip, check=False):
            b = QPushButton(txt)

            b.setObjectName("btnGhost")
            b.setToolTip(tip)
            b.setCheckable(check)
            b.setFixedHeight(38)
            b.setCursor(Qt.PointingHandCursor)

            bh.addWidget(b)

            return b

        bh.addStretch(1)

        self.b_ai = fbtn("🤖  AI", "AI Overlay", True)
        self.b_heat = fbtn("🔥  Heatmap", "Heatmap", True)
        self.b_snap = fbtn("📸  Snapshot", "Snapshot")
        self.b_back = fbtn("←  Back", "Exit fullscreen")

        bh.addStretch(1)

        self.b_ai.toggled.connect(lambda c: setattr(sim, "ai_on", c))
        self.b_heat.toggled.connect(self._set_heat)
        self.b_snap.clicked.connect(lambda: hub.snapshot(sim))
        self.b_back.clicked.connect(self.accept)

        v.addWidget(bot)

        self.refresh()

    def _set_heat(self,enabled):
        self.sim.heat_on=enabled
        if enabled:self.hub.sys.async_api.submit(lambda:self.hub.sys.api.get_heatmap(self.sim.id,"live"),lambda snapshot:(self.sim.apply_heatmap(snapshot),self.refresh()),lambda error:self.hub.toast(f"Heatmap API: {error}"),owner=self)

    def refresh(self):
        s = self.sim

        self.lbl_st.setText("🟢 ONLINE" if s.online else "🔴 OFFLINE")
        self.lbl_st.setStyleSheet(
            f"color:{TH.OK if s.online else TH.ERR}; font-size:11px; font-weight:800;"
        )

        for b, val in ((self.b_ai, s.ai_on), (self.b_heat, s.heat_on)):
            b.blockSignals(True)
            b.setChecked(val)
            b.blockSignals(False)

    def done(self,result):
        if not self._disposed:
            self._disposed=True;self.surface.zoom=1.0;self.surface.offset=QPointF(0,0);self.surface.dispose()
        super().done(result)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_Back):
            self.accept()


# ============================= CHARTS ================================
class ChartCard(QFrame):
    def __init__(self, title):
        super().__init__()

        self.setObjectName("chartCard")

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        t = QLabel(title)
        t.setStyleSheet(
            f"color:{TH.DIM}; font-size:10px; font-weight:800; letter-spacing:1px;"
        )

        v.addWidget(t)

        self.body = v


class BarChart(QWidget):
    def __init__(self, horizontal=False):
        super().__init__()
        self.values, self.labels = [], []
        self.color, self.horizontal = TH.ACCENT, horizontal
        self._value_formatter = None
        self._max_value = None
        self._top_values = None
        self._show_minutes = False
        self.setMinimumHeight(120)

    def set_data(self, values, labels, color=None, formatter=None, max_value=None, top_values=None):
        self.values, self.labels = values, labels
        if color: self.color = color
        self._value_formatter = formatter
        self._max_value = max_value
        self._top_values = top_values
        self.update()

    def _label_text(self, v):
        if callable(self._value_formatter): return self._value_formatter(v)
        if self._show_minutes: return f"{int(v)}m"
        return str(int(v))

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self.values)
        if n == 0: p.end(); return
        mx = self._max_value if self._max_value else max(1, max(self.values))
        p.setFont(QFont("Segoe UI", 7))
        if self.horizontal:
            bh = max(6, (h - 4 * (n - 1)) // n)
            for i, (v, l) in enumerate(zip(self.values, self.labels)):
                y = i * (bh + 4); bw = int(v / mx * (w - 70))
                p.setPen(QColor(TH.DIM)); p.drawText(QRect(0, y, 52, bh), Qt.AlignVCenter | Qt.AlignRight, l)
                p.setPen(Qt.NoPen); p.setBrush(QColor(TH.CARD2)); p.drawRoundedRect(56, y, w - 56, bh, 3, 3)
                p.setBrush(QColor(self.color))
                if bw > 2: p.drawRoundedRect(56, y, bw, bh, 3, 3)
                if v > 0:
                    text = self._label_text(v)
                    if text:
                        p.setPen(QColor("#ffffff")); fnt = p.font(); fnt.setPixelSize(9); fnt.setBold(True); p.setFont(fnt)
                        fm = p.fontMetrics(); tw = fm.horizontalAdvance(text)
                        tx = 56 + bw - tw - 6 if bw > tw + 10 else 56 + bw + 4
                        p.drawText(tx, y + bh // 2 + fm.ascent() // 2 - 2, text)
        else:
            bw = max(4, (w - 6 * (n - 1)) // n)
            for i, (v, l) in enumerate(zip(self.values, self.labels)):
                x = i * (bw + 6); bh2 = int(v / mx * (h - 16))
                p.setPen(Qt.NoPen); p.setBrush(QColor(TH.CARD2)); p.drawRoundedRect(x, 0, bw, h - 14, 3, 3)
                p.setBrush(QColor(self.color))
                if bh2 > 2: p.drawRoundedRect(x, h - 14 - bh2, bw, bh2, 3, 3)
                if self._top_values and i < len(self._top_values) and self._top_values[i] > 0:
                    tv = self._top_values[i]
                    p.setPen(QColor("#f59e0b")); fnt = p.font(); fnt.setPixelSize(10); fnt.setBold(True); p.setFont(fnt)
                    tt = f"{int(tv)} kishi"; fm = p.fontMetrics(); tw = fm.horizontalAdvance(tt)
                    p.drawText(x + (bw - tw) // 2, max(10, (h - 14 - bh2) - 6), tt)
                if v > 0:
                    text = self._label_text(v)
                    if text:
                        p.setPen(QColor("#ffffff")); fnt = p.font(); fnt.setPixelSize(9); fnt.setBold(True); p.setFont(fnt)
                        fm = p.fontMetrics(); tw = fm.horizontalAdvance(text); tx = x + (bw - tw) // 2
                        ty = (h - 14 - bh2) + bh2 // 2 + fm.ascent() // 2 - 2 if bh2 > 18 else (h - 14 - bh2) - 4
                        p.drawText(tx, ty, text)
                p.setPen(QColor(TH.FAINT)); p.setFont(QFont("Segoe UI", 7))
                p.drawText(QRect(x - 6, h - 13, bw + 12, 12), Qt.AlignHCenter, l)
        p.end()


class Page(QWidget):
    def __init__(self):
        super().__init__()

        self.setObjectName("page")

        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(14, 12, 14, 12)
        self.v.setSpacing(10)

    def title_row(self, title, sub=""):
        row = QHBoxLayout()
        row.setSpacing(10)

        t = QLabel(title)
        t.setStyleSheet("font-size:16px; font-weight:800; color:white;")

        row.addWidget(t)

        if sub:
            s = QLabel(sub)
            s.setStyleSheet(f"color:{TH.FAINT}; font-size:10px;")
            row.addWidget(s)

        row.addStretch(1)

        self.v.addLayout(row)

        return row


class LivePage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Dashboard", "Double-click camera to enlarge")
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self.grid_content = QWidget()
        self.grid_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.grid = QGridLayout(self.grid_content)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(10)
        self.grid.setVerticalSpacing(10)
        self.grid.setAlignment(Qt.AlignTop)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.scroll.setWidget(self.grid_content)
        self.v.addWidget(self.scroll, 1)
        self.cards = []
        self.reload_cameras()

    def reload_cameras(self):
        for card in self.cards:
            self.grid.removeWidget(card);card.dispose();card.deleteLater()
        self.cards=[]
        cameras=list(self.hub.sys.sims)
        for index,sim in enumerate(cameras):
            card=CameraCard(sim,self.hub);self.cards.append(card)
            row,column=camera_grid_position(index);self.grid.addWidget(card,row,column)
        rows=(len(cameras)+CAMERA_GRID_COLUMNS-1)//CAMERA_GRID_COLUMNS
        for row in range(rows):
            self.grid.setRowMinimumHeight(row,210);self.grid.setRowStretch(row,1)
        self.grid_content.setMinimumHeight(max(0,rows*210+max(0,rows-1)*self.grid.verticalSpacing()))
        if hasattr(self.hub,"cards"):self.hub.cards=self.cards

class EventDetailDialog(QDialog):
    TYPES = {
        "recognized": ("Recognized", TH.OK),
        "unknown": ("Unknown", "#f59e42"),
        "offline": ("Offline", TH.ERR),
        "online": ("Online", TH.OK),
        "snapshot": ("Snapshot", TH.ACCENT),
        "system": ("System", TH.DIM),
        "enrolled": ("Enrolled", TH.OK),
        "intrusion": ("Zone Intrusion", TH.ERR),
        "overstay": ("Overstay", TH.WARN),
    }

    def __init__(self, e, sys_, parent=None):
        super().__init__(parent)

        self.e = e

        self.setWindowTitle(f"Event — {e['type']}")

        self.setModal(True)
        self.setFixedWidth(480)

        v = QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(12)

        name, col = self.TYPES.get(e["type"], (e["type"], TH.DIM))

        badge = QLabel(name)
        badge.setStyleSheet(
            f"background:{col}; color:#0c1116; font-weight:800;"
            "border-radius:5px; padding:4px 12px; font-size:11px;"
        )
        badge.setFixedWidth(badge.sizeHint().width() + 24)

        v.addWidget(badge)

        g = QGridLayout()
        g.setSpacing(8)

        for i, (label, val) in enumerate([
            ("Time", e["time"].strftime("%Y-%m-%d %H:%M:%S")),
            ("Camera", e["cam"]),
            ("Person", e["person"]),
            ("Confidence", f'{e["conf"] * 100:.1f}%' if e["conf"] < 1 else "—"),
            ("Level", e["level"].upper()),
        ]):
            l = QLabel(label)
            l.setStyleSheet(f"color:{TH.DIM}; font-size:10px; font-weight:700;")

            vl = QLabel(str(val))
            vl.setStyleSheet(f"color:{TH.TXT}; font-size:11px;")

            g.addWidget(l, i, 0)
            g.addWidget(vl, i, 1)

        v.addLayout(g)

                # Snapshot ko'rsatish
        sim = sys_.sim_by_id(e["cam"])
        
        snapshot_path = e.get("snapshot_path")
        
        # Avval snapshot faylni tekshirish
        if snapshot_path and os.path.exists(snapshot_path):
            pm = QPixmap(snapshot_path).scaled(
                440,
                247,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            snap = QLabel()
            snap.setPixmap(pm)
            snap.setStyleSheet(f"border:1px solid {TH.BORDER}; border-radius:8px;")

            v.addWidget(snap)
            
        # Agar fayl yo'q bo'lsa, lekin camera frame bor bo'lsa
        elif sim and sim.frame:
            pm = QPixmap.fromImage(sim.frame).scaled(
                440,
                247,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            snap = QLabel()
            snap.setPixmap(pm)
            snap.setStyleSheet(f"border:1px solid {TH.BORDER}; border-radius:8px;")

            v.addWidget(snap)

        else:
            na = QLabel("📷 Snapshot not available")
            na.setAlignment(Qt.AlignCenter)
            na.setStyleSheet(
                f"color:{TH.FAINT}; background:{TH.CARD};"
                "border-radius:8px; padding:20px; font-size:10px;"
            )

            v.addWidget(na)

        row = QHBoxLayout()

        ack = QPushButton("✓ Acknowledge")
        ack.setObjectName("btnPrimary")
        ack.setCursor(Qt.PointingHandCursor)
        ack.clicked.connect(lambda: self._ack(sys_))

        close = QPushButton("Close")
        close.setObjectName("btnGhost")
        close.setCursor(Qt.PointingHandCursor)
        close.clicked.connect(self.accept)

        row.addStretch(1)
        row.addWidget(ack)
        row.addWidget(close)

        v.addLayout(row)

    def _ack(self, sys_):
        event_id=self.e.get("id") or self.e.get("event_id")
        if not event_id:return
        def done(_):
            self.e["ack"]=True;sys_.push_event(dict(type="system",level="info",cam=self.e.get("cam","SYS"),person="Event acknowledged",conf=1.0));self.accept()
        sys_.async_api.submit(lambda:sys_.api.acknowledge_event(event_id),done,lambda error:QMessageBox.warning(self,"API",error))


class EventsPage(Page):
    TYPES = EventDetailDialog.TYPES

    def __init__(self, hub):

        super().__init__()

        self.hub = hub

        row = self.title_row("Events", f"{len(hub.sys.events)} records | 📅 Bugun")


        self.flt = QComboBox()
        self.flt.addItems([
            "All",
            "Camera",
            "Person",
            "Identity",
            "Enrollment",
            "Conflict",
        ])
        self.flt.currentTextChanged.connect(self.rebuild)

        exp = QPushButton("📄 Export CSV")
        exp.setObjectName("btnGhost")
        exp.setFixedHeight(28)
        exp.setCursor(Qt.PointingHandCursor)
        exp.clicked.connect(self.export_csv)

        # Date picker
        from PySide6.QtCore import QDate
        self.date_picker = QDateEdit()
        self.date_picker.setCalendarPopup(True)
        self.date_picker.setDate(QDate.currentDate())
        self.date_picker.setDisplayFormat("dd/MM/yyyy")
        self.date_picker.setFixedWidth(130)
        self.date_picker.setFixedHeight(28)
        self.date_picker.dateChanged.connect(self._on_date_changed)
        
        today_btn = QPushButton("📅 Bugun")
        today_btn.setObjectName("btnGhost")
        today_btn.setFixedHeight(28)
        today_btn.setCursor(Qt.PointingHandCursor)
        today_btn.clicked.connect(lambda: self.date_picker.setDate(QDate.currentDate()))
        
        row.addWidget(self.date_picker)
        row.addWidget(today_btn)
        row.addWidget(self.flt)
        row.addWidget(exp)

        self.tbl = QTableWidget(0, 6)

        self.tbl.setHorizontalHeaderLabels([
            "Time",
            "Type",
            "Camera",
            "Person / Identity",
            "Details",
            "Status",
        ])

        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.tbl.verticalHeader().setVisible(False)

        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(True)

        self.tbl.cellDoubleClicked.connect(self._detail)

        self.v.addWidget(self.tbl, 1)

        self._events = []
        self._load_events_for_date(QDate.currentDate().toString("yyyy-MM-dd"))
        self.rebuild()

    def _match(self, e):
        f = self.flt.currentText()

        ok = (
            f == "All"
            or (f == "Camera" and e["type"].startswith("camera."))
            or (f == "Person" and e["type"].startswith("person."))
            or (f == "Identity" and e["type"].startswith("identity."))
            or (f == "Enrollment" and e["type"].startswith("enrollment."))
            or (f == "Conflict" and e["type"] == "identity.conflict")
        )

        return ok

    def _load_events_for_date(self,date_str):
        def loaded(items):
            self._events=[self._normalize_event(item) for item in items];self.rebuild()
        self.hub.sys.async_api.submit(
            lambda:self.hub.sys.api.get_events(from_ts=f"{date_str}T00:00:00Z",to_ts=f"{date_str}T23:59:59Z"),
            loaded,lambda error:self.hub.toast(f"Events API: {error}"))

    @staticmethod
    def _normalize_event(e):
        n = dict(e)
        metadata = n.get("metadata") or n.get("details") or {}
        n["type"] = str(n.get("event_type") or n.get("type") or "legacy")
        n["time"] = n.get("timestamp") or n.get("time") or datetime.now()
        n["cam"] = str(n.get("camera_id") or n.get("cam") or "")
        n["person"] = str(n.get("person_name") or n.get("name") or n.get("person_id") or n.get("global_id") or n.get("person") or "")
        n["conf"] = float(n.get("confidence") or metadata.get("confidence") or 0.0)
        n["level"] = str(n.get("severity") or n.get("level") or "info")
        n["ack"] = bool(n.get("acknowledged", n.get("ack", False)))
        n["details_text"] = str(metadata.get("message") or n.get("message") or metadata or "")
        if n.get("taxonomy_status")=="legacy":n["details_text"]="LEGACY | "+n["details_text"]
        return n

    @staticmethod
    def _event_date(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%Y-%m-%d")
        if isinstance(t, str):
            return t[:10]
        return ""

    @staticmethod
    def _event_time(e):
        t = e.get("time")
        if hasattr(t, "strftime"):
            return t.strftime("%H:%M:%S")
        s = str(t)
        return s[11:19] if len(s) >= 19 else s

    def _on_date_changed(self, qdate):
        """Date picker o'zgarganda events ni qayta yuklash"""
        date_str = qdate.toString("yyyy-MM-dd")
        log.debug("Event date changed: %s",date_str)
        self._load_events_for_date(date_str)
        self.rebuild()

    def rebuild(self):
        self.tbl.setRowCount(0)
        if not hasattr(self, '_events'):
            self._events = []
        selected_date = self.date_picker.date().toString("yyyy-MM-dd") if hasattr(self, 'date_picker') else None
        for e in self._events[:300]:
            if selected_date and self._event_date(e) != selected_date:
                continue
            if self._match(e):
                self._insert(e)

    def add_event(self, e):
        e = self._normalize_event(e)
        selected_date = self.date_picker.date().toString("yyyy-MM-dd") if hasattr(self, 'date_picker') else None
        if selected_date and self._event_date(e) != selected_date:
            return
        if not hasattr(self, '_events'):
            self._events = []
        self._events.insert(0, e)
        if self._match(e):
            self._insert(e, top=True)

    def _insert(self, e, top=False):
        r = 0 if top else self.tbl.rowCount()
        self.tbl.insertRow(r)
        name, col = self.TYPES.get(e["type"], (e["type"], TH.DIM))
        vals = [self._event_time(e), name, e["cam"], e["person"], e.get("details_text", ""), "Acknowledged" if e.get("ack") else "New"]
        for c, txt in enumerate(vals):
            it = QTableWidgetItem(str(txt))
            it.setTextAlignment(Qt.AlignCenter if c in (0, 2, 5) else Qt.AlignVCenter | Qt.AlignLeft)
            if c == 1: it.setForeground(QColor(col))
            if c == 0: it.setFont(QFont("Consolas", 9))
            if e.get("ack"): it.setForeground(QColor(TH.FAINT))
            self.tbl.setItem(r, c, it)
        self.tbl.setRowHeight(r, 30)
        if self.tbl.rowCount() > 300: self.tbl.removeRow(self.tbl.rowCount() - 1)

    def _snap(self, cid):
        s = self.hub.sys.sim_by_id(cid)

        if s:
            self.hub.snapshot(s)
        else:
            self.hub.toast("⚠ Camera not available")

    def _detail(self, r, c):
        it = self.tbl.item(r, 1)
        if not it:
            return
        cam = it.text()
        time_str = self.tbl.item(r, 0).text() if self.tbl.item(r, 0) else ""
        for e in getattr(self, '_events', []):
            if e["cam"] == cam and self._event_time(e) == time_str:
                EventDetailDialog(e, self.hub.sys, self).exec()
                self.rebuild()
                return

    def export_csv(self):
        os.makedirs("exports", exist_ok=True)
        fn = f"exports/events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fn, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Time", "Type", "Camera", "Person / Identity", "Details", "Status"])
            for e in getattr(self, '_events', []):
                if self._match(e):
                    t = e.get("time")
                    tstr = t.isoformat() if hasattr(t, "isoformat") else str(t)
                    w.writerow([tstr, e["type"], e["cam"], e["person"], e.get("details_text", ""), "Acknowledged" if e.get("ack") else "New"])
        self.hub.toast(f"📄 Exported {self.tbl.rowCount()} events → {fn}")

class PersonManagementPage(Page):
    def __init__(self, hub):

        super().__init__()

        self.hub = hub

        row = self.title_row(
            "Person Management",
            f"{len(hub.sys.people)} registered",
        )

        # # Refresh tugmasi
        

        sync_btn = QPushButton("🔄 DB Sync")
        sync_btn.setObjectName("btnPrimary")
        sync_btn.setFixedHeight(28)
        sync_btn.setCursor(Qt.PointingHandCursor)
        sync_btn.clicked.connect(self.db_sync)
        
        row.addWidget(sync_btn)
        


        enr = QPushButton("＋ Enroll New")
        enr.setObjectName("btnPrimary")
        enr.setCursor(Qt.PointingHandCursor)
        enr.clicked.connect(lambda: hub.navigate("enroll"))

        row.addWidget(enr)

        self.tbl = QTableWidget(0, 6)

        self.tbl.setHorizontalHeaderLabels([
            "Photo",
            "Name",
            "Department",
            "Status",
            "Last Seen",
            "Recognitions",
        ])

        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        self.tbl.verticalHeader().setVisible(False)

        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setShowGrid(False)
        self.tbl.setAlternatingRowColors(True)

        self.tbl.cellClicked.connect(self._profile)

        self.v.addWidget(self.tbl, 1)

        self._online_persons = {}  # person_id -> {camera_id, track_id, name}
        
        self.hub.sys.websocket.message.connect(self._on_realtime_message)
        
        self.rebuild()

    def _on_realtime_message(self,message):
        if message.get("type")!="person.identified":return
        self._on_persons_online(message.get("camera_id",""),[message])

    def _on_persons_online(self, camera_id, persons_list):
        """IdentityManager dan kelgan online persons"""
        for p in persons_list:
            pid = p.get("person_id")
            if pid is not None:
                self._online_persons[pid] = {
                    "camera_id": camera_id,
                    "track_id": p.get("track_id"),
                    "name": p.get("name", "Unknown"),
                    "known": p.get("known", False),
                    "stay_sec": p.get("stay_sec", 0),
                    "total_stay": p.get("total_stay", 0),
                }
        # Offline bo'lganlarni tozalash (bu kamerada ko'rinmayotganlar)
        cam_person_ids = {p.get("person_id") for p in persons_list if p.get("person_id")}
        to_remove = [pid for pid, info in self._online_persons.items() 
                     if info["camera_id"] == camera_id and pid not in cam_person_ids]
        for pid in to_remove:
            del self._online_persons[pid]
        
        # Jadvalni yangilash (faqat status ustunini)
        self._update_online_status()

    def _update_online_status(self):
        """Jadvalda online/offline statusni yangilash"""
        for row in range(self.tbl.rowCount()):
            name_item = self.tbl.item(row, 1)
            status_item = self.tbl.item(row, 3)
            if name_item is None or status_item is None:
                continue
            
            # person_id ni topish (name orqali)
            person_name = name_item.text()
            is_online = any(
                info["name"] == person_name or 
                (info["known"] and str(info.get("person_id","")) == person_name)
                for info in self._online_persons.values()
            )
            
            if is_online:
                status_item.setForeground(QColor("#2ecc71"))
            else:
                status_item.setForeground(QColor("#95a5a6"))

    def showEvent(self, e):
        """Person Management ochilganda DB dan avto-sync."""
        super().showEvent(e)
        import time as _st
        _n = _st.time()
        if not hasattr(self, "_last_db_sync") or _n - self._last_db_sync > 3:
            self._last_db_sync = _n
            try:
                self.db_sync()
                log.debug("Person records synchronized on page entry")
            except Exception as _se:
                log.warning("Person synchronization failed: %s",_se)

    def rebuild(self):
        """Jadvalni to'liq qayta qurish — yuz + jonli status"""
        try:
            from datetime import datetime

            self.tbl.clearContents()
            self.tbl.setRowCount(0)

            people = list(self.hub.sys.people)

            visible_count = 0

            for rec in people:
                r = self.tbl.rowCount()
                self.tbl.insertRow(r)

                # ===== PHOTO =====
                ph = QTableWidgetItem()
                avatar_pm = None
                # 1) PersonRecordUI dan avatar
                if hasattr(rec, 'avatar') and rec.avatar is not None and not rec.avatar.isNull():
                    avatar_pm = rec.avatar

                if avatar_pm is not None:
                    scaled = avatar_pm.scaled(38, 38, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                    ph.setData(Qt.DecorationRole, scaled)

                ph.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 0, ph)

                # ===== NAME =====
                p_name = (getattr(rec, 'name', None) or getattr(rec, 'person_name', None) or "?")
                p_emp = (getattr(rec, 'emp_id', None) or getattr(rec, 'employee_id', None) or "")
                nm = QTableWidgetItem(f"{p_name}\n{p_emp}")
                nm.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 1, nm)

                # ===== DEPARTMENT =====
                dept_item = QTableWidgetItem(getattr(rec, 'dept', None) or getattr(rec, 'department', None) or "—")
                dept_item.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 2, dept_item)

                # ===== STATUS =====
                # Online/Offline status
                pid = getattr(rec, 'id', None) or getattr(rec, 'person_id', None)
                is_online = pid in self._online_persons if hasattr(self, '_online_persons') and pid else False
                if is_online:
                    _cam = self._online_persons.get(pid, {}).get("camera_id", "")
                    status_text, status_color = f"🟢 Online · {_cam}", "#2ecc71"
                else:
                    status_text, status_color = "", "#95a5a6"  # ✅ No offline label
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                st.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 3, st)

                # ===== LAST SEEN =====
                ls_val = getattr(rec, 'last_seen', None) or getattr(rec, 'last_seen_dt', None)
                if ls_val and hasattr(ls_val, 'strftime'):
                    ls_text = ls_val.strftime("%d/%m/%Y %H:%M")
                elif ls_val:
                    ls_text = str(ls_val)[:16]
                else:
                    ls_text = "—"
                ls_item = QTableWidgetItem(ls_text)
                ls_item.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 4, ls_item)

                # ===== RECOGNITIONS + STAY TOTAL =====
                rc_val = getattr(rec, 'rec_count', 0) or 0
                stay_val = getattr(rec, 'stay_total', 0) or 0
                stay_text = f"{int(stay_val//60)}m {int(stay_val%60)}s" if stay_val else "0m"
                rc = QTableWidgetItem(f"{rc_val}\n{stay_text}")
                rc.setTextAlignment(Qt.AlignCenter)
                rc.setData(Qt.UserRole, rec)
                self.tbl.setItem(r, 5, rc)

                self.tbl.setRowHeight(r, 50)
                visible_count += 1

            if hasattr(self, 'lbl_count'):
                self.lbl_count.setText(f"{visible_count} registered")


            # Auto-refresh timer
            if not hasattr(self, '_presence_timer'):
                from PySide6.QtCore import QTimer
                self._presence_timer = QTimer(self)
                self._presence_timer.setInterval(20000)
                self._presence_timer.timeout.connect(self._auto_refresh_presence)
                self._presence_timer.start()

        except Exception as e:
            log.exception("Person table rebuild failed")

    def _auto_refresh_presence(self):
        """Har 20s da faqat Status ustunini yangilash (tez, to'liq rebuild emas)"""
        try:
            for r in range(self.tbl.rowCount()):
                nm = self.tbl.item(r, 1)
                if not nm:
                    continue
                rec = nm.data(Qt.UserRole)
                if rec is None:
                    continue
                # Online/Offline status
                pid = getattr(rec, 'id', None) or getattr(rec, 'person_id', None)
                is_online = pid in self._online_persons if hasattr(self, '_online_persons') and pid else False
                if is_online:
                    status_text, status_color = "🟢 Online", "#2ecc71"
                else:
                    status_text, status_color = "", "#95a5a6"
                st = QTableWidgetItem(status_text)
                st.setForeground(QColor(status_color))
                self.tbl.setItem(r, 3, st)
        except (RuntimeError,TypeError,AttributeError):
            log.exception("Person presence refresh failed")

    def add_record(self, rec):
        self.hub.sys.people.append(rec)

        self.rebuild()

    def _profile(self, r, c):
        """Istalgan ustunga bossa profil ochiladi"""
        rec = None

        # 1) Bosilgan ustundan UserRole olishga urinamiz
        it = self.tbl.item(r, c)
        if it:
            rec = it.data(Qt.UserRole)

        # 2) Bo'sh bo'lsa, Name ustunidan (1) olamiz
        if rec is None:
            name_it = self.tbl.item(r, 1)
            if name_it:
                rec = name_it.data(Qt.UserRole)

        # 3) Hali bo'sh bo'lsa, Photo ustunidan (0) olamiz
        if rec is None:
            photo_it = self.tbl.item(r, 0)
            if photo_it:
                rec = photo_it.data(Qt.UserRole)

        if rec is not None:
            ProfileDialog(rec,api=self.hub.sys.api,async_api=self.hub.sys.async_api).exec()
        else:
            log.warning("No person record at row=%s column=%s",r,c)

    def force_refresh(self):
        def loaded(rows):
            if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
                self.hub.toast("Persons API returned invalid data");return
            self.hub.sys.people=[self.hub.sys.person_record(row) for row in rows]
            self.rebuild()
        self.hub.sys.async_api.submit(self.hub.sys.api.get_persons,loaded,lambda error:self.hub.toast(f"Persons API: {error}"))

    def db_sync(self):
        self.force_refresh()

class ProfileDialog(QDialog):
    def __init__(self,rec,api,async_api):
        super().__init__()
        self.rec=rec;self.api=api;self.async_api=async_api
        self.setWindowTitle(f"Profile — {rec.name}")
        self.setModal(True)
        self.resize(780, 720)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(12)

        # ===== HEADER (FIXED) =====
        top = QHBoxLayout(); top.setSpacing(14)
        av = QLabel()
        if hasattr(rec, 'avatar') and rec.avatar is not None and not rec.avatar.isNull():
            av.setPixmap(rec.avatar.scaled(84, 84, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        av.setStyleSheet("border-radius:12px;")

        info = QVBoxLayout()
        nm = QLabel(rec.name)
        nm.setStyleSheet("font-size:17px; font-weight:800; color:white;")
        from datetime import datetime
        st_text, st_color = "—", TH.FAINT  # default
        if rec.last_seen:
            try: secs = (datetime.now() - rec.last_seen).total_seconds()
            except (TypeError,AttributeError):secs=999999
            if secs < 60: st_text, st_color = "🟢 Online", TH.OK
            elif secs < 300: st_text, st_color = f"🟡 {int(secs // 60)}m ago", TH.WARN
            else: st_text, st_color = "—", TH.FAINT
        st = QLabel(st_text)
        st.setStyleSheet(f"color:{st_color}; font-size:11px; font-weight:700;")
        info.addWidget(nm); info.addWidget(st); info.addStretch(1)
        top.addWidget(av); top.addLayout(info); top.addStretch(1)
        outer.addLayout(top)

        # ===== STATS (FIXED) =====
        stats = QHBoxLayout()
        ls_text = rec.last_seen.strftime("%d %b %H:%M") if rec.last_seen else "—"
        for label, val in [("LAST SEEN", ls_text), ("RECOGNITIONS", str(rec.rec_count)), ("STAY TOTAL", f"{rec.stay_total} min")]:
            f = QFrame(); f.setObjectName("statCard")
            fl = QVBoxLayout(f); fl.setContentsMargins(10, 8, 10, 8); fl.setSpacing(2)
            a = QLabel(val); a.setStyleSheet("color:white; font-size:13px; font-weight:800;")
            b = QLabel(label); b.setStyleSheet(f"color:{TH.FAINT}; font-size:8px; font-weight:800;")
            fl.addWidget(a); fl.addWidget(b); stats.addWidget(f)
        outer.addLayout(stats)

        # ===== VISIT HISTORY (SCROLL ICHIDA) =====
        vc = ChartCard("VISIT HISTORY (last 7 days)")
        visit_scroll = QScrollArea()
        visit_scroll.setWidgetResizable(True)
        visit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        visit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        visit_scroll.setMaximumHeight(180)
        visit_scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        visit_content = QWidget()
        vv = QVBoxLayout(visit_content); vv.setContentsMargins(0,0,0,0); vv.setSpacing(4)
        l = QLabel("No visit records yet");l.setStyleSheet(f"color:{TH.FAINT}; font-size:10.5px; font-style:italic;");vv.addWidget(l)
        vv.addStretch(1)
        visit_scroll.setWidget(visit_content)
        vc.body.addWidget(visit_scroll)
        outer.addWidget(vc)

        # ===== RECENT EVENTS (SCROLL ICHIDA) =====
        ec = ChartCard("RECENT EVENTS")
        event_scroll = QScrollArea()
        event_scroll.setWidgetResizable(True)
        event_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        event_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        event_scroll.setMaximumHeight(180)
        event_scroll.setStyleSheet("QScrollArea{border:none; background:transparent;}")
        event_content = QWidget()
        ev = QVBoxLayout(event_content); ev.setContentsMargins(0,0,0,0); ev.setSpacing(4)
        events_loaded = False
        if not events_loaded:
            l = QLabel("No events recorded yet"); l.setStyleSheet(f"color:{TH.FAINT}; font-size:10.5px; font-style:italic;"); ev.addWidget(l)
        ev.addStretch(1)
        event_scroll.setWidget(event_content)
        ec.body.addWidget(event_scroll)
        outer.addWidget(ec)

        # ===== TIMELINE =====
        tl = ChartCard("TIMELINE (presence min/hour - Today, 60min=full)")
        ch = BarChart()
        timeline_data = [0] * 24
        ch.set_data(timeline_data, [f"{i:02d}" for i in range(24)], QColor("#3b82f6"),
                   formatter=lambda v: f"{int(v)}m" if v > 0 else "", max_value=60)
        ch._show_minutes = True; ch.setMinimumHeight(90)
        tl.body.addWidget(ch); outer.addWidget(tl)

        # ===== BUTTONS (FIXED) =====
        btn_row = QHBoxLayout(); btn_row.addStretch(1)
        edit_btn = QPushButton("✏ Edit"); edit_btn.setObjectName("btnGhost")
        edit_btn.setCursor(Qt.PointingHandCursor); edit_btn.clicked.connect(self._edit_person)
        del_btn = QPushButton("🗑 Delete"); del_btn.setObjectName("btnGhost")
        del_btn.setCursor(Qt.PointingHandCursor); del_btn.setStyleSheet(f"color:{TH.ERR};")
        del_btn.clicked.connect(self._delete_person)
        close_btn = QPushButton("Close"); close_btn.setObjectName("btnPrimary")
        close_btn.setCursor(Qt.PointingHandCursor); close_btn.clicked.connect(self.accept)
        btn_row.addWidget(edit_btn); btn_row.addWidget(del_btn); btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _edit_person(self):
        new_name,ok1=QInputDialog.getText(self,"Edit Name","Name:",text=self.rec.name)
        if not ok1 or not new_name.strip():return
        new_dept,ok2=QInputDialog.getText(self,"Edit Department","Department:",text=getattr(self.rec,"dept",""))
        if not ok2:return
        self.async_api.submit(lambda:self.api.update_person(self.rec.db_id,{"name":new_name.strip(),"department":new_dept.strip()}),lambda _:(setattr(self.rec,"name",new_name.strip()),setattr(self.rec,"dept",new_dept.strip()),self.accept()),lambda error:QMessageBox.warning(self,"API",error),owner=self)

    def _delete_person(self):
        if QMessageBox.question(self,"Delete Person",f"Are you sure you want to delete {self.rec.name!r}?",QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        self.async_api.submit(lambda:self.api.delete_person(self.rec.db_id),lambda _:self.accept(),lambda error:QMessageBox.warning(self,"API",error),owner=self)


class EnrollmentPage(Page):
    def __init__(self, hub):
        super().__init__()

        self.hub = hub
        self.session_id=None
        self.hub.sys.websocket.message.connect(self._on_enrollment_event)

        self.title_row(
            "Person Enrollment",
            "select at least 10 clear face images · validated asynchronously",
        )

        body = QHBoxLayout()
        body.setSpacing(12)

        # ==================== LEFT: IMAGE UPLOAD ====================
        left = QFrame()
        left.setObjectName("camCard")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(28, 28, 28, 28)
        ll.setSpacing(16)

        upload_icon = QLabel("🖼️")
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet("font-size:72px; border:none;")
        upload_title = QLabel("Upload face images")
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setStyleSheet(f"color:{TH.TXT}; font-size:22px; font-weight:800; border:none;")
        upload_hint = QLabel("Choose at least 10 clear photos. Invalid, blurry, small, or multi-face images will be rejected.")
        upload_hint.setWordWrap(True)
        upload_hint.setAlignment(Qt.AlignCenter)
        upload_hint.setStyleSheet(f"color:{TH.DIM}; font-size:12px; border:none;")
        self.face_status = QLabel("No images selected")
        self.face_status.setAlignment(Qt.AlignCenter)
        self.face_status.setStyleSheet(f"color:{TH.DIM}; font-size:11px; font-weight:700; border:none;")
        self.btn_upload_main = QPushButton("📁  Select face images")
        self.btn_upload_main.setObjectName("btnPrimary")
        self.btn_upload_main.setCursor(Qt.PointingHandCursor)
        self.btn_upload_main.clicked.connect(self.upload_images)

        ll.addStretch(1)
        ll.addWidget(upload_icon)
        ll.addWidget(upload_title)
        ll.addWidget(upload_hint)
        ll.addWidget(self.btn_upload_main, 0, Qt.AlignHCenter)
        ll.addWidget(self.face_status)
        ll.addStretch(1)
        body.addWidget(left, 1)

        # ==================== RIGHT: FORM ====================
        right = QFrame()
        right.setObjectName("chartCard")
        right.setFixedWidth(350)

        f = QVBoxLayout(right)
        f.setContentsMargins(16, 16, 16, 16)
        f.setSpacing(10)

        t = QLabel("REGISTER NEW PERSON")
        t.setStyleSheet(
            f"color:{TH.ACC2}; font-size:9px; font-weight:800; letter-spacing:1.5px;"
        )
        f.addWidget(t)

        self.name = QLineEdit()
        self.name.setPlaceholderText("Full Name *")

        self.dept = QComboBox()
        self.dept.addItems([
            "Security", "IT", "Finance", "HR",
            "Operations", "Management", "Other",
        ])

        self.dept_custom = QLineEdit()
        self.dept_custom.setPlaceholderText("Other department name...")
        self.dept_custom.hide()
        self.dept.currentTextChanged.connect(self._on_dept_changed)


        f.addWidget(self.name)
        f.addWidget(self.dept)
        f.addWidget(self.dept_custom)

        self.prog = QProgressBar()
        self.prog.setRange(0, 10)
        self.prog.setValue(0)
        self.prog.setFixedHeight(8)
        self.prog.setTextVisible(False)

        self.prog_lbl = QLabel("Uploaded 0 / 10")
        self.prog_lbl.setStyleSheet(
            f"color:{TH.DIM}; font-size:10px; font-weight:700;"
        )

        f.addWidget(self.prog_lbl)
        f.addWidget(self.prog)

        self.thumbs = QGridLayout()
        self.thumbs.setSpacing(6)
        f.addLayout(self.thumbs)

        self.emb = QLabel("Embedding: —")
        self.emb.setStyleSheet(
            f"color:{TH.FAINT}; font-size:9.5px; font-family:Consolas,monospace;"
        )
        f.addWidget(self.emb)
        f.addStretch(1)

        # Image-only enrollment; webcam capture is intentionally unavailable.
        self.btn_upload = QPushButton("📁 Select Images")
        self.btn_upload.setObjectName("btnGhost")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self.upload_images)
        self.btn_reg = QPushButton("💾  Register")
        self.btn_reg.setObjectName("btnPrimary")
        self.btn_reg.setEnabled(False)
        self.btn_reg.setCursor(Qt.PointingHandCursor)
        self.btn_reg.clicked.connect(self.register)

        f.addWidget(self.btn_upload)
        f.addWidget(self.btn_reg)

        body.addWidget(right)
        self.v.addLayout(body, 1)

        self.captures = []

    def _on_enrollment_event(self,event):
        if event.get("session_id")!=self.session_id:return
        kind=event.get("type")
        if kind=="enrollment.progress":
            captured=int(event.get("captured",0));required=int(event.get("required",10))
            self.prog.setMaximum(required);self.prog.setValue(captured);self.prog_lbl.setText(f"Valid {captured} / {required}")
            self.face_status.setText(event.get("message") or f"Quality {float(event.get('quality',0)):.0%}")
        elif kind=="enrollment.completed":
            self.hub.toast("✅ Enrollment completed");self._reset()
            if hasattr(self.hub,"pm"):self.hub.pm.force_refresh()
        elif kind in ("enrollment.failed","enrollment.cancelled"):
            self.face_status.setText(f"⚠ {event.get('message',kind)}");self.btn_reg.setEnabled(True)

    def upload_images(self):
        paths,_=QFileDialog.getOpenFileNames(self,"Select enrollment images","","Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not paths:return
        self.captures=list(dict.fromkeys(paths))[:30];count=len(self.captures)
        self.prog.setMaximum(max(10,count));self.prog.setValue(0);self.prog_lbl.setText(f"Selected {count} / 10 required")
        self.face_status.setText(f"{count} images ready for validation");self.btn_reg.setEnabled(count>=10)

    # ==================== REGISTER ====================
    def register(self):
        name=self.name.text().strip()
        if not name:self.hub.toast("⚠ Please enter the person name");self.name.setFocus();return
        dept=self.dept.currentText();dept=self.dept_custom.text().strip() or "Other" if dept=="Other" else dept
        if len(self.captures)<10:self.hub.toast("⚠ Select at least 10 images");return
        self.btn_reg.setEnabled(False);self.face_status.setText("Uploading images for validation…")
        def started(session):
            self.session_id=session["id"] if "id" in session else session["session_id"]
            self.face_status.setText("Validating face images…")
            self.hub.toast(f"Enrollment started for {name}")
        self.hub.sys.async_api.submit(lambda:self.hub.sys.api.start_enrollment(name,self.captures,dept),started,lambda error:(self.btn_reg.setEnabled(True),self.face_status.setText(f"⚠ {error}")))

    # ==================== RESET ====================
    def _reset(self):
        for i in reversed(range(self.thumbs.count())):
            w = self.thumbs.itemAt(i).widget()
            if w:
                w.deleteLater()

        self.captures = []
        self.prog.setValue(0)
        self.prog_lbl.setText("Uploaded 0 / 10")
        self.emb.setText("Embedding: —")
        self.emb.setStyleSheet(f"color:{TH.FAINT}; font-size:9.5px; font-family:Consolas,monospace;")
        self.btn_reg.setEnabled(False)
        self.name.clear()
        self.face_status.setText("🟢 Ready for next enrollment")

    def _on_dept_changed(self, text):
        if text == "Other":
            self.dept_custom.show()
        else:
            self.dept_custom.hide()

            
class SettingsPage(Page):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.title_row("Settings", "System preferences")
        self.tabs = QTabWidget()
        self.v.addWidget(self.tabs, 1)
        st = self.hub.sys.settings

        # ==== Cameras ====
        w = QWidget()
        self.camera_layout = QVBoxLayout(w)
        self.camera_layout.setSpacing(8)
        self.rebuild_cameras()
        self.tabs.addTab(w, "🎥 Cameras")

        # ==== Database ====
        w = QWidget()
        fm = QFormLayout(w)
        fm.setSpacing(12)
        fm.addRow("Storage", QLabel("SQLite · managed by API service"))
        br = QHBoxLayout()
        bb = QPushButton("Backup Now")
        bb.setObjectName("btnGhost")
        bb.clicked.connect(lambda: hub.toast("💾 Database backup started"))
        bv = QPushButton("Vacuum")
        bv.setObjectName("btnGhost")
        bv.clicked.connect(lambda: hub.toast("✅ Database optimized"))
        br.addWidget(bb)
        br.addWidget(bv)
        br.addStretch(1)
        fm.addRow("", br)
        self.tabs.addTab(w, "🗄 Database")

        # ==== Preferences ====
        w = QWidget()
        fm = QFormLayout(w)
        fm.setSpacing(12)
        fm.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.snd = QCheckBox("Sound alerts for critical events")
        self.snd.setChecked(st.get("sound", True))
        self.snd.stateChanged.connect(lambda _state:hub.sys.async_api.submit(lambda:hub.sys.api.update_settings({"sound_enabled":self.snd.isChecked()}),lambda _:hub.sys.settings.__setitem__("sound",self.snd.isChecked()),lambda error:hub.toast(f"Settings API: {error}")))
        fm.addRow("Notifications", self.snd)
        self.tabs.addTab(w, "Preferences")

    def rebuild_cameras(self):
        while self.camera_layout.count():
            item=self.camera_layout.takeAt(0)
            if item.widget():item.widget().deleteLater()
            if item.layout():
                while item.layout().count():
                    child=item.layout().takeAt(0)
                    if child.widget():child.widget().deleteLater()
        for camera in self.hub.sys.sims:
            row=QHBoxLayout();checkbox=QCheckBox(f"{camera.id}  —  {camera.name}");checkbox.setChecked(camera.enabled)
            def changed(_state,on=camera,box=checkbox):
                requested=box.isChecked();box.setEnabled(False)
                def applied(_):box.setEnabled(True);self.hub.sys.refresh_cameras()
                def failed(error):
                    box.blockSignals(True);box.setChecked(on.enabled);box.blockSignals(False);box.setEnabled(True);self.hub.toast(f"Camera API: {error}")
                self.hub.sys.async_api.submit(lambda:self.hub.sys.api.update_camera(on.id,{"enabled":requested}),applied,failed,owner=box)
            checkbox.stateChanged.connect(changed)
            resolution=QLabel(camera.res);fps=QLabel(f"{camera.fps:.0f} FPS");edit=QPushButton("Detection Recovery ROI");edit.setObjectName("btnGhost");edit.clicked.connect(lambda _checked=False,on=camera:ROIEditorDialog(self.hub,on,self).exec())
            row.addWidget(checkbox,1);row.addWidget(resolution);row.addWidget(fps);row.addWidget(edit);self.camera_layout.addLayout(row)
        self.camera_layout.addStretch(1)


SPLASH_STEPS = [
    (0, "Tizim ishga tushmoqda..."),
    (20, "Sozlamalar yuklanmoqda..."),
    (40, "Kameralar ulanmoqda..."),
    (60, "AI modellar tayyorlanmoqda..."),
    (80, "Ma'lumotlar bazasi ochilmoqda..."),
    (100, "Tayyor!"),
]


class SplashScreen(QDialog):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Dialog)

        self.setFixedSize(480, 300)
        self.setModal(True)

        scr = QApplication.primaryScreen().availableGeometry()

        self.move(scr.center() - self.rect().center())

        v = QVBoxLayout(self)
        v.setContentsMargins(40, 36, 40, 30)
        v.setSpacing(14)

        logo = QLabel("◉")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color:{TH.ACCENT}; font-size:44px; font-weight:900;")

        t = QLabel("AI SURVEILLANCE SYSTEM")
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet(
            "color:white; font-size:18px; font-weight:900; letter-spacing:2px;"
        )

        s = QLabel("Operator Console")
        s.setAlignment(Qt.AlignCenter)
        s.setStyleSheet(f"color:{TH.DIM}; font-size:10px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)

        self.lbl = QLabel(SPLASH_STEPS[0][1])
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet(f"color:{TH.FAINT}; font-size:10px;")

        for w in (logo, t, s):
            v.addWidget(w)

        v.addSpacing(10)

        v.addWidget(self.bar)
        v.addWidget(self.lbl)

        self.step = 0

        self.timer = QTimer(self)
        self.timer.setInterval(320)
        self.timer.timeout.connect(self.next_step)
        self.timer.start()

    def next_step(self):
        self.step += 1

        if self.step >= len(SPLASH_STEPS):
            self.timer.stop()

            self.accept()

            return

        pct, txt = SPLASH_STEPS[self.step]

        self.bar.setValue(pct)
        self.lbl.setText(txt)


# =========================== MAIN WINDOW =============================
class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("AI Surveillance System — Operator Console")

        self.resize(1680, 980)
        self.setMinimumSize(1180, 700)

        self.sys = System()

        self.tick_n = 0
        self.fs = None

        self._sb_anim = []
        self._nav_anim = None
        self._force_close = False

        self.header = Header(self.sys, self)

        self.stack = QStackedWidget()

        self.live = LivePage(self)
        self.pm = PersonManagementPage(self)
        self.enroll = EnrollmentPage(self)
        self.events_pg = EventsPage(self)
        self.settings_pg = SettingsPage(self)
        self.sys.cameras_changed.connect(self._cameras_changed)

        self.page_index = {}

        for key, w in [
            ("live", self.live),
            ("people", self.pm),
            ("enroll", self.enroll),
            ("events", self.events_pg),
            ("settings", self.settings_pg),
        ]:
            self.page_index[key] = self.stack.addWidget(w)

        self.cards = self.live.cards

        self.right = RightPanel(self.sys)

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(1)

        split.addWidget(self.stack)
        split.addWidget(self.right)

        split.setStretchFactor(0, 70)
        split.setStretchFactor(1, 15)

        self.sidebar = SideBar(self)

        body = QWidget()

        hb = QHBoxLayout(body)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)

        hb.addWidget(self.sidebar)
        hb.addWidget(split)

        self.central_widget = QWidget()

        vb = QVBoxLayout(self.central_widget)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)

        vb.addWidget(self.header)
        vb.addWidget(body, 1)

        self.setCentralWidget(self.central_widget)

        self.notif = NotificationDropdown(self)

        self.fade = QWidget(self.central_widget)
        self.fade.setStyleSheet(f"background:{TH.BG};")
        self.fade.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fade.hide()

        self.fade_eff = QGraphicsOpacityEffect(self.fade)
        self.fade.setGraphicsEffect(self.fade_eff)
        self.fade_eff.setOpacity(0)

        self.sidebar.changed.connect(self.navigate)
        self.sys.new_event.connect(self.on_event)

        self.fast_timer = QTimer(self)
        self.fast_timer.timeout.connect(self.tick)
        self.fast_timer.start(40)

        self.slow_timer = QTimer(self)
        self.slow_timer.timeout.connect(self.slow_tick)
        self.slow_timer.start(1000)

        for e in reversed(self.sys.events):
            self.right.add_event(e)
            self.events_pg.add_event(e)

        self.navigate("live")

        self.right.refresh()

        self.ai_ready_timer=QTimer(self);self.ai_ready_timer.setSingleShot(True);self.ai_ready_timer.timeout.connect(self.header.set_ai_ready);self.ai_ready_timer.start(3000)
        self.sys.refresh_cameras()

        pages = ("live","people","enroll","events","settings")
        for key,page in zip("12345",pages):
            QShortcut(QKeySequence(key),self,lambda target=page:self.navigate(target))

        QShortcut(QKeySequence("Ctrl+E"), self, lambda: self.navigate("events"))

        self._tray_ok = QSystemTrayIcon.isSystemTrayAvailable()

        if self._tray_ok:
            self.tray = QSystemTrayIcon(self)

            ipm = QPixmap(32, 32)
            ipm.fill(Qt.transparent)

            p = QPainter(ipm)
            p.setRenderHint(QPainter.Antialiasing)

            p.setBrush(QColor(TH.ACCENT))
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, 32, 32)

            p.setPen(QColor("white"))
            p.setFont(QFont("Segoe UI", 10, QFont.Bold))
            p.drawText(ipm.rect(), Qt.AlignCenter, "AI")

            p.end()

            self.tray.setIcon(QIcon(ipm))
            self.tray.setToolTip("AI Surveillance System")

            menu = QMenu()

            menu.addAction("Show").triggered.connect(self.showNormal)
            menu.addAction("Hide").triggered.connect(self.hide)
            menu.addAction("Exit").triggered.connect(self._tray_exit)

            self.tray.setContextMenu(menu)

            self.tray.activated.connect(
                lambda r: self.showNormal()
                if r == QSystemTrayIcon.DoubleClick
                else None
            )

            self.tray.show()

    def navigate(self, page):
        if self.stack.currentIndex() == self.page_index[page]:
            self.sidebar.set_active(page)

            return

        pos = self.stack.mapTo(self.central_widget, QPoint(0, 0))

        self.fade.setGeometry(pos.x(), pos.y(), self.stack.width(), self.stack.height())

        self.fade.show()
        self.fade.raise_()

        a = QPropertyAnimation(self.fade_eff, b"opacity")
        a.setDuration(110)
        a.setStartValue(0)
        a.setEndValue(1)
        a.finished.connect(lambda: self._finish_nav(page))
        a.start()

        self._nav_anim = a

    def _finish_nav(self, page):
        self.stack.setCurrentIndex(self.page_index[page])

        self.sidebar.set_active(page)

        a = QPropertyAnimation(self.fade_eff, b"opacity")
        a.setDuration(110)
        a.setStartValue(1)
        a.setEndValue(0)
        a.finished.connect(self.fade.hide)
        a.start()

        self._nav_anim = a

    def toggle_sidebar(self):
        collapsed = not self.sidebar.collapsed

        w = 80 if collapsed else 210

        self.sidebar.set_collapsed(collapsed)

        self._sb_anim = []

        for prop in (b"minimumWidth", b"maximumWidth"):
            a = QPropertyAnimation(self.sidebar, prop)

            a.setDuration(160)
            a.setEndValue(w)
            a.start()

            self._sb_anim.append(a)

    def on_event(self, e):
        self.right.add_event(e)
        self.events_pg.add_event(e)
        self.notif.add_alert(e)

        if e.get("level") in ("warn", "err"):
            self.header.bump()

            if self.sys.settings.get("sound", True):
                QApplication.beep()

    def _cameras_changed(self):
        self.live.reload_cameras();self.settings_pg.rebuild_cameras()
        self.cards=self.live.cards;self.header.update_stats();self.right.refresh()

    def tick(self):
        self.tick_n += 1

        if self.tick_n % 5 == 0:
            for c in self.cards:
                c.refresh()

            self.right.refresh()

            if self.fs:
                self.fs.refresh()

    def slow_tick(self):
        sy = self.sys
        def apply_metrics(metrics):
            sy.pipeline_metrics = dict(metrics or {})
            system = dict(sy.pipeline_metrics.get("system") or {})
            sy.system_metrics = system
            sy.gpu = system.get("gpu_utilization_percent")
            sy.cpu = system.get("cpu_percent")
            sy.ram = system.get("ram_percent")
            self.header.update_stats()
        sy.async_api.submit(sy.api.get_system_metrics,apply_metrics,lambda _error:None)

        self.header.tick_clock()


    def open_fullscreen(self,sim):
        client=self.sys.video_clients.get(sim.id);sim.independent_display_frame_domain=True
        if client:client.set_display_mode(True)
        dlg=FullscreenCam(sim,self);self.fs=dlg
        try:
            dlg.showFullScreen();dlg.exec()
        finally:
            sim.independent_display_frame_domain=False
            if client:client.set_display_mode(False)
            self.fs=None

    def snapshot(self, sim):
        if not sim.online or sim.frame is None:
            self.toast("⚠ Camera is offline")

            return

        os.makedirs("snapshots", exist_ok=True)

        fn = f"snapshots/{sim.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        sim.frame.save(fn)

        self.toast(f"📸 Snapshot saved — {fn}")

        self.sys.push_event(
            dict(
                type="snapshot",
                level="info",
                cam=sim.id,
                person="Snapshot captured",
                conf=1.0,
            )
        )

    def toast(self, msg):
        t = QLabel(msg, self)

        t.setStyleSheet(
            f"background:{TH.CARD2}; color:{TH.TXT};"
            f"border:1px solid {TH.BORDER}; border-radius:9px;"
            "padding:10px 16px; font-size:11.5px; font-weight:600;"
        )

        t.adjustSize()

        t.move(self.width() - t.width() - 24, self.height() - t.height() - 24)

        t.show()
        t.raise_()

        eff = QGraphicsOpacityEffect(t)
        t.setGraphicsEffect(eff)

        anim = QPropertyAnimation(eff, b"opacity", t)
        anim.setDuration(500)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)

        anim2 = QPropertyAnimation(t, b"pos", t)
        anim2.setDuration(500)
        anim2.setStartValue(t.pos())
        anim2.setEndValue(t.pos() + QPoint(0, 14))

        start_timer=QTimer(t);start_timer.setSingleShot(True);start_timer.timeout.connect(anim.start);start_timer.timeout.connect(anim2.start);start_timer.start(2100)
        cleanup_timer=QTimer(t);cleanup_timer.setSingleShot(True);cleanup_timer.timeout.connect(t.deleteLater);cleanup_timer.start(2700)

    def _tray_exit(self):
        self._force_close = True

        self.close()

    def closeEvent(self, e):
        if self._tray_ok and not self._force_close:
            self.hide()

            self.tray.showMessage(
                "AI Surveillance System",
                "Minimized to system tray",
                QSystemTrayIcon.Information,
                2000,
            )

            e.ignore()

        else:
            self.fast_timer.stop();self.slow_timer.stop();self.ai_ready_timer.stop()
            for card in tuple(self.cards):card.dispose()
            if self.fs:self.fs.close()
            self.sys.shutdown()
            e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Slash and not isinstance(self.focusWidget(), QLineEdit):
            self.header.search.setFocus()

            e.accept()

        else:
            super().keyPressEvent(e)


# ============================ STYLESHEET =============================
STYLE = """
* { font-family: "Segoe UI", "Roboto", "Ubuntu", sans-serif; }

QMainWindow, QDialog { background: #101418; }

QWidget { color: #e9eef5; font-size: 12px; }

QWidget#page { background: #0f1317; }

QFrame#header { background: #151b22; border-bottom: 1px solid #2b3542; }

QFrame#sidebar { background: #151b22; border-right: 1px solid #2b3542; }

QFrame#rightPanel { background: #151b22; }

QFrame#chip { background: #1a212a; border: 1px solid #2b3542; border-radius: 12px; }

QPushButton#bellBtn { background: #1a212a; border: 1px solid #2b3542;
    border-radius: 17px; font-size: 14px; }

QPushButton#bellBtn:hover { background: #26303c; }

QPushButton#sideBtn { text-align: left; padding: 10px 14px; border-radius: 8px;
    color: #94a1b3; font-size: 12.5px; border: none; background: transparent; }

QPushButton#sideBtn[collapsed="true"] { text-align: center; padding: 10px 0; }

QPushButton#sideBtn:hover { background: #202a35; color: #e9eef5; }

QPushButton#sideBtn:checked { background: #2f7df6; color: #fff; font-weight: 700; }

QFrame#camCard { background: #0a0d11; border: 1px solid #2b3542; border-radius: 10px; }

QFrame#camCard[offline="true"] { border: 1px solid #ef5350; }

QFrame#camToolbar { background: rgba(13,17,22,215); border: 1px solid #2b3542;
    border-radius: 9px; }

QToolButton#camTool, QPushButton#camTool { background: transparent; border: none;
    border-radius: 6px; font-size: 13px; color: #e9eef5; }

QToolButton#camTool:hover, QPushButton#camTool:hover { background: #2f7df6; }

QToolButton#camTool:checked { background: #2f7df6; }

QFrame#quickInfo { background: rgba(13,17,22,228); border: 1px solid #2b3542;
    border-radius: 9px; }

QFrame#statCard, QFrame#chartCard { background: #1a212a;
    border: 1px solid #2b3542; border-radius: 10px; }

QFrame#alertItem { background: #1a212a; }

QFrame#notifDrop { background: #1a212a; border: 1px solid #2b3542;
    border-radius: 10px; }

QFrame#notifItem { background: #212a35; border-radius: 6px; }

QFrame#notifItem:hover { background: #26303c; }

QPushButton#btnPrimary { background: #2f7df6; color: white; border: none;
    border-radius: 7px; padding: 8px 16px; font-weight: 700; }

QPushButton#btnPrimary:hover { background: #4a8ff7; }

QPushButton#btnPrimary:disabled { background: #233043; color: #5d6b7e; }

QPushButton#btnGhost { background: #232c37; border: 1px solid #2b3542;
    border-radius: 7px; padding: 7px 14px; color: #e9eef5; }

QPushButton#btnGhost:hover { background: #2a3441; }

QPushButton#btnGhost:checked { background: #2f7df6; border-color: #5b9bff;
    color: white; }

QLineEdit, QComboBox, QSpinBox { background: #232c37; border: 1px solid #2b3542;
    border-radius: 6px; padding: 6px 9px;
    selection-background-color: #2f7df6; }

QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #2f7df6; }

QComboBox QAbstractItemView { background: #202a35; border: 1px solid #2b3542;
    selection-background-color: #2f7df6; outline: none; }

QProgressBar { background: #232c37; border-radius: 3px; border: none; }

QProgressBar::chunk { background: #2f7df6; border-radius: 3px; }

QTableWidget { background: #1a212a; alternate-background-color: #1d2530;
    gridline-color: #2b3542; border: 1px solid #2b3542; border-radius: 8px;
    selection-background-color: #2f7df6; }

QTableWidget::item { padding: 4px 8px; }

QHeaderView::section { background: #202a35; color: #94a1b3; padding: 7px;
    border: none; border-bottom: 1px solid #2b3542;
    font-weight: 700; font-size: 10.5px; }

QTabWidget::pane { border: 1px solid #2b3542; border-radius: 8px;
    background: #1a212a; top: -1px; }

QTabBar::tab { padding: 9px 18px; background: transparent; color: #94a1b3;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px; margin-right: 2px; }

QTabBar::tab:selected { background: #1a212a; color: #fff;
    border-color: #2b3542; font-weight: 700; }

QTabBar::tab:hover { color: #e9eef5; }

QMenu { background: #202a35; border: 1px solid #2b3542;
    border-radius: 8px; padding: 4px; }

QMenu::item { padding: 7px 26px; border-radius: 5px; }

QMenu::item:selected { background: #2f7df6; color: white; }

QScrollArea { background: transparent; }

QScrollBar:vertical { background: transparent; width: 9px; margin: 0; }

QScrollBar::handle:vertical { background: #2b3542; border-radius: 4px;
    min-height: 24px; }

QScrollBar::handle:vertical:hover { background: #3a4757; }

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal { background: transparent; height: 9px; }

QScrollBar::handle:horizontal { background: #2b3542; border-radius: 4px;
    min-width: 24px; }

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSlider::groove:horizontal { background: #232c37; height: 5px;
    border-radius: 2px; }

QSlider::handle:horizontal { background: #2f7df6; width: 14px;
    margin: -5px 0; border-radius: 7px; }

QCheckBox { spacing: 8px; }

QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
    border: 1px solid #2b3542; background: #232c37; }

QCheckBox::indicator:checked { background: #2f7df6; border-color: #2f7df6; }

QSplitter::handle { background: #2b3542; }

QToolTip { background: #202a35; color: #e9eef5; border: 1px solid #2b3542;
    border-radius: 5px; padding: 5px 8px; }
"""
