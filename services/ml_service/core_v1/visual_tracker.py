from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class VisualBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(slots=True)
class _Track:
    track_id: int
    box: VisualBox
    raw_box: VisualBox
    last_seen: float
    last_update: float
    last_observation: float
    hits: int = 1
    vcx: float = 0.0
    vcy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0


def _width(box: VisualBox) -> float:
    return max(0.0, box.x2 - box.x1)


def _height(box: VisualBox) -> float:
    return max(0.0, box.y2 - box.y1)


def _area(box: VisualBox) -> float:
    return _width(box) * _height(box)


def _center_size(box: VisualBox):
    w=max(1.0,_width(box));h=max(1.0,_height(box))
    return (box.x1+box.x2)*0.5,(box.y1+box.y2)*0.5,w,h


def _from_center_size(cx:float,cy:float,w:float,h:float,confidence:float) -> VisualBox:
    w=max(1.0,float(w));h=max(1.0,float(h))
    return VisualBox(cx-w*0.5,cy-h*0.5,cx+w*0.5,cy+h*0.5,confidence)


def _intersection(a:VisualBox,b:VisualBox) -> float:
    x1=max(a.x1,b.x1);y1=max(a.y1,b.y1);x2=min(a.x2,b.x2);y2=min(a.y2,b.y2)
    return max(0.0,x2-x1)*max(0.0,y2-y1)


def _iou(a:VisualBox,b:VisualBox) -> float:
    inter=_intersection(a,b);union=_area(a)+_area(b)-inter
    return inter/union if union>0 else 0.0


def _containment(a:VisualBox,b:VisualBox) -> float:
    inter=_intersection(a,b);smaller=min(_area(a),_area(b))
    return inter/smaller if smaller>0 else 0.0


def _center_distance(a:VisualBox,b:VisualBox) -> float:
    acx=(a.x1+a.x2)*0.5;acy=(a.y1+a.y2)*0.5;bcx=(b.x1+b.x2)*0.5;bcy=(b.y1+b.y2)*0.5
    scale=max(20.0,math.sqrt(max(_area(a),_area(b))))
    return math.hypot(acx-bcx,acy-bcy)/scale


def _center_shift_boxes(a:VisualBox,b:VisualBox) -> float:
    acx,acy,aw,ah=_center_size(a);bcx,bcy,bw,bh=_center_size(b)
    return math.hypot(acx-bcx,acy-bcy)/max(20.0,max(aw,ah,bw,bh))


def _horizontal_overlap_ratio(a:VisualBox,b:VisualBox) -> float:
    overlap=max(0.0,min(a.x2,b.x2)-max(a.x1,b.x1));smaller=max(1.0,min(_width(a),_width(b)))
    return overlap/smaller


def _vertical_overlap_ratio(a:VisualBox,b:VisualBox) -> float:
    overlap=max(0.0,min(a.y2,b.y2)-max(a.y1,b.y1));smaller=max(1.0,min(_height(a),_height(b)))
    return overlap/smaller


def _vertical_gap_ratio(a:VisualBox,b:VisualBox) -> float:
    if a.y2<b.y1:gap=b.y1-a.y2
    elif b.y2<a.y1:gap=a.y1-b.y2
    else:gap=0.0
    return gap/max(1.0,max(_height(a),_height(b)))


def _x_center_ratio(a:VisualBox,b:VisualBox) -> float:
    acx=(a.x1+a.x2)*0.5;bcx=(b.x1+b.x2)*0.5
    return abs(acx-bcx)/max(1.0,max(_width(a),_width(b)))


def _area_ratio(a:VisualBox,b:VisualBox) -> float:
    aa=_area(a);bb=_area(b)
    return min(aa,bb)/max(1.0,max(aa,bb))


class VisualTracker:
    """Visual-only continuity and smooth presentation for one camera.

    Detections remain the source of truth. The display state keeps box center,
    size and a bounded constant-velocity estimate separately. Small detector
    jitter is smoothed; real movement gets a faster adaptive correction; short
    detector gaps are bridged with bounded prediction. This state must never be
    used as ReID/global-identity evidence.
    """

    def __init__(self, *, hold_ms=900,memory_ms=6000,prediction_ms=250,
                 match_iou=0.16,reacquire_distance=1.05,
                 duplicate_iou=0.50,duplicate_containment=0.82,
                 duplicate_center_distance=0.40,fragment_duplicate=False,
                 fragment_horizontal_overlap=0.70,fragment_x_center=0.30,
                 fragment_max_area_ratio=0.70,fragment_min_vertical_overlap=0.12,
                 fragment_max_vertical_gap=0.10,smoothing=0.96,
                 center_smoothing=None,size_smoothing=None,velocity_smoothing=0.45,
                 max_prediction_shift_boxes=0.40,max_prediction_size_ratio=0.12,
                 adaptive_center_smoothing=0.88,adaptive_error_boxes=0.28,
                 snap_distance_boxes=0.80,reversal_damping=0.30,
                 low_conf_confirm=0.12,start_conf=0.34,new_track_min_conf=0.24,
                 strong_confirm_hits=2,weak_confirm_hits=3,new_track_zones=None,
                 exclusion_zones=None,exclusion_max_box_height=0.24,
                 exclusion_overlap_threshold=0.35):
        self.hold_sec=max(0.05,float(hold_ms)/1000.0);self.memory_sec=max(self.hold_sec,float(memory_ms)/1000.0);self.prediction_sec=max(0.0,float(prediction_ms)/1000.0)
        self.match_iou=float(match_iou);self.reacquire_distance=max(0.1,float(reacquire_distance));self.duplicate_iou=float(duplicate_iou);self.duplicate_containment=float(duplicate_containment);self.duplicate_center_distance=max(0.0,float(duplicate_center_distance))
        self.fragment_duplicate=bool(fragment_duplicate);self.fragment_horizontal_overlap=float(fragment_horizontal_overlap);self.fragment_x_center=float(fragment_x_center);self.fragment_max_area_ratio=float(fragment_max_area_ratio);self.fragment_min_vertical_overlap=float(fragment_min_vertical_overlap);self.fragment_max_vertical_gap=float(fragment_max_vertical_gap)
        legacy=min(0.99,max(0.05,float(smoothing)))
        self.center_smoothing=min(0.98,max(0.05,float(center_smoothing if center_smoothing is not None else legacy)))
        self.size_smoothing=min(0.98,max(0.03,float(size_smoothing if size_smoothing is not None else min(legacy,0.60))))
        self.velocity_smoothing=min(0.95,max(0.05,float(velocity_smoothing)))
        self.max_prediction_shift_boxes=max(0.10,float(max_prediction_shift_boxes));self.max_prediction_size_ratio=max(0.0,min(0.50,float(max_prediction_size_ratio)))
        self.adaptive_center_smoothing=min(0.99,max(self.center_smoothing,float(adaptive_center_smoothing)));self.adaptive_error_boxes=max(0.05,float(adaptive_error_boxes));self.snap_distance_boxes=max(self.adaptive_error_boxes,float(snap_distance_boxes));self.reversal_damping=max(0.0,min(1.0,float(reversal_damping)))
        self.low_conf_confirm=float(low_conf_confirm);self.start_conf=float(start_conf);self.new_track_min_conf=float(new_track_min_conf);self.strong_confirm_hits=max(1,int(strong_confirm_hits));self.weak_confirm_hits=max(self.strong_confirm_hits,int(weak_confirm_hits))
        self.new_track_zones=[tuple(float(v) for v in zone[:5]) for zone in (new_track_zones or []) if len(zone)>=5];self.exclusion_zones=[tuple(float(v) for v in zone[:4]) for zone in (exclusion_zones or []) if len(zone)>=4]
        self.exclusion_max_box_height=max(0.0,float(exclusion_max_box_height));self.exclusion_overlap_threshold=max(0.0,min(1.0,float(exclusion_overlap_threshold)))
        self._tracks:dict[int,_Track]={};self._next_id=1;self._last_result_frame_id=-1;self._birth_candidates:list[tuple[VisualBox,float,int]]=[]

    def _excluded(self,box:VisualBox,source_width:float|None,source_height:float|None) -> bool:
        if not self.exclusion_zones or not source_width or not source_height:return False
        width=max(1.0,float(source_width));height=max(1.0,float(source_height));cx=((box.x1+box.x2)*0.5)/width;cy=((box.y1+box.y2)*0.5)/height;box_h=max(0.0,box.y2-box.y1)/height
        if box_h>self.exclusion_max_box_height:return False
        box_norm=VisualBox(box.x1/width,box.y1/height,box.x2/width,box.y2/height,box.confidence)
        for x1,y1,x2,y2 in self.exclusion_zones:
            zone=VisualBox(x1,y1,x2,y2,1.0)
            if x1<=cx<=x2 and y1<=cy<=y2:return True
            if _intersection(box_norm,zone)/max(1e-9,_area(box_norm))>=self.exclusion_overlap_threshold:return True
        return False

    def _birth_threshold(self,box:VisualBox,source_width:float|None,source_height:float|None) -> float:
        threshold=self.new_track_min_conf
        if not self.new_track_zones or not source_width or not source_height:return threshold
        width=max(1.0,float(source_width));height=max(1.0,float(source_height));cx=((box.x1+box.x2)*0.5)/width;cy=((box.y1+box.y2)*0.5)/height
        for x1,y1,x2,y2,zone_conf in self.new_track_zones:
            if x1<=cx<=x2 and y1<=cy<=y2:threshold=max(threshold,float(zone_conf))
        return threshold

    def _fragment_duplicate_match(self,box:VisualBox,other:VisualBox) -> bool:
        if not self.fragment_duplicate:return False
        if _area_ratio(box,other)>self.fragment_max_area_ratio:return False
        if _horizontal_overlap_ratio(box,other)<self.fragment_horizontal_overlap:return False
        if _x_center_ratio(box,other)>self.fragment_x_center:return False
        return _vertical_overlap_ratio(box,other)>=self.fragment_min_vertical_overlap or _vertical_gap_ratio(box,other)<=self.fragment_max_vertical_gap

    def _is_duplicate(self,box:VisualBox,other:VisualBox) -> bool:
        iou=_iou(box,other);containment=_containment(box,other);distance=_center_distance(box,other)
        return iou>=self.duplicate_iou or (containment>=self.duplicate_containment and distance<=self.duplicate_center_distance) or self._fragment_duplicate_match(box,other)

    def _dedupe(self,boxes,source_width=None,source_height=None) -> list[VisualBox]:
        candidates=[VisualBox(float(b.x1),float(b.y1),float(b.x2),float(b.y2),float(b.confidence)) for b in boxes];candidates=[b for b in candidates if not self._excluded(b,source_width,source_height)];candidates.sort(key=lambda b:b.confidence,reverse=True);kept=[]
        for box in candidates:
            if not any(self._is_duplicate(box,other) for other in kept):kept.append(box)
        return kept

    def _predicted_box(self,track:_Track,target_time:float) -> VisualBox:
        dt=min(max(0.0,float(target_time)-track.last_observation),self.prediction_sec);cx,cy,w,h=_center_size(track.box);dx=track.vcx*dt;dy=track.vcy*dt
        max_shift=max(12.0,max(w,h)*self.max_prediction_shift_boxes);magnitude=math.hypot(dx,dy)
        if magnitude>max_shift and magnitude>1e-6:
            factor=max_shift/magnitude;dx*=factor;dy*=factor
        max_dw=w*self.max_prediction_size_ratio;max_dh=h*self.max_prediction_size_ratio;dw=max(-max_dw,min(max_dw,track.vw*dt));dh=max(-max_dh,min(max_dh,track.vh*dt))
        return _from_center_size(cx+dx,cy+dy,w+dw,h+dh,track.box.confidence)

    def _confirm_new_track(self,det:VisualBox,now:float,required_hits:int) -> bool:
        fresh=[];matched=None
        for box,ts,hits in self._birth_candidates:
            if now-ts<=1.5:
                fresh.append((box,ts,hits))
                if matched is None and (_iou(box,det)>=0.20 or _center_distance(box,det)<=0.55):matched=(box,ts,hits)
        self._birth_candidates=fresh
        if matched is None:self._birth_candidates.append((det,now,1));return required_hits<=1
        self._birth_candidates.remove(matched);hits=matched[2]+1
        if hits>=required_hits:return True
        self._birth_candidates.append((det,now,hits));return False

    def _update_motion(self,track:_Track,det:VisualBox,observation:float,now:float) -> None:
        old_cx,old_cy,old_w,old_h=_center_size(track.raw_box);det_cx,det_cy,det_w,det_h=_center_size(det);obs_dt=observation-track.last_observation
        if obs_dt<=0.005:obs_dt=max(0.01,now-track.last_update)
        nv_cx=(det_cx-old_cx)/obs_dt;nv_cy=(det_cy-old_cy)/obs_dt;nv_w=(det_w-old_w)/obs_dt;nv_h=(det_h-old_h)/obs_dt
        if track.vcx*nv_cx<0:track.vcx*=self.reversal_damping
        if track.vcy*nv_cy<0:track.vcy*=self.reversal_damping
        if track.vw*nv_w<0:track.vw*=self.reversal_damping
        if track.vh*nv_h<0:track.vh*=self.reversal_damping
        va=self.velocity_smoothing;vb=1.0-va;track.vcx=vb*track.vcx+va*nv_cx;track.vcy=vb*track.vcy+va*nv_cy;track.vw=vb*track.vw+va*nv_w;track.vh=vb*track.vh+va*nv_h
        predicted=self._predicted_box(track,observation);pcx,pcy,pw,ph=_center_size(predicted);error_boxes=_center_shift_boxes(predicted,det);ca=self.adaptive_center_smoothing if error_boxes>=self.adaptive_error_boxes else self.center_smoothing;sa=self.size_smoothing
        if error_boxes>=self.snap_distance_boxes:
            new_cx,new_cy,new_w,new_h=det_cx,det_cy,det_w,det_h;track.vcx*=0.50;track.vcy*=0.50;track.vw*=0.35;track.vh*=0.35
        else:
            new_cx=(1.0-ca)*pcx+ca*det_cx;new_cy=(1.0-ca)*pcy+ca*det_cy;new_w=(1.0-sa)*pw+sa*det_w;new_h=(1.0-sa)*ph+sa*det_h
        track.box=_from_center_size(new_cx,new_cy,new_w,new_h,max(det.confidence,track.box.confidence*0.82));track.raw_box=det;track.last_seen=now;track.last_update=now;track.last_observation=observation;track.hits+=1

    def update(self,result,now:float,source_width=None,source_height=None) -> None:
        if result is None or int(result.frame_id)==self._last_result_frame_id:return
        self._last_result_frame_id=int(result.frame_id);detections=self._dedupe(result.boxes,source_width,source_height);observation=float(getattr(result,'frame_captured_monotonic',now) or now)
        if not math.isfinite(observation) or observation<=0:observation=now
        for tid in list(self._tracks):
            if now-self._tracks[tid].last_seen>self.memory_sec:del self._tracks[tid]
        unmatched_tracks=set(self._tracks);unmatched_dets=set(range(len(detections)));pairs=[]
        for tid,track in self._tracks.items():
            reference=self._predicted_box(track,observation);dormant=(now-track.last_seen)>self.hold_sec;max_distance=self.reacquire_distance if dormant else 0.82
            for di,det in enumerate(detections):
                iou=_iou(reference,det);dist=_center_distance(reference,det)
                if iou>=self.match_iou or dist<=max_distance:pairs.append((iou+(max(0.0,max_distance-dist)*0.30)-(0.08 if dormant else 0.0),tid,di))
        pairs.sort(reverse=True);matched_tracks=set()
        for _score,tid,di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:continue
            self._update_motion(self._tracks[tid],detections[di],observation,now);matched_tracks.add(tid);unmatched_tracks.remove(tid);unmatched_dets.remove(di)
        for di in list(unmatched_dets):
            det=detections[di]
            if any(self._is_duplicate(det,self._tracks[tid].box) for tid in matched_tracks):unmatched_dets.remove(di)
        for di in unmatched_dets:
            det=detections[di];zone_threshold=self._birth_threshold(det,source_width,source_height);strong_threshold=max(self.start_conf,zone_threshold)
            if det.confidence>=strong_threshold:confirmed=self._confirm_new_track(det,now,self.strong_confirm_hits)
            elif det.confidence>=zone_threshold:confirmed=self._confirm_new_track(det,now,self.weak_confirm_hits)
            else:confirmed=False
            if not confirmed:continue
            tid=self._next_id;self._next_id+=1;self._tracks[tid]=_Track(track_id=tid,box=det,raw_box=det,last_seen=now,last_update=now,last_observation=observation)

    def visible(self,now:float,target_time:float|None=None,max_observation_age_sec:float|None=None) -> list[VisualBox]:
        target=float(target_time if target_time is not None else now);candidates=[]
        for track in self._tracks.values():
            age=now-track.last_seen
            if age>self.hold_sec:continue
            source_age=max(0.0,target-track.last_observation)
            if max_observation_age_sec is not None and source_age>max_observation_age_sec:continue
            if track.hits<2 and track.box.confidence<self.low_conf_confirm:continue
            predicted=self._predicted_box(track,target);predicted.confidence=max(0.01,track.box.confidence*(1.0-0.30*min(1.0,age/self.hold_sec)));candidates.append(predicted)
        candidates.sort(key=lambda b:b.confidence,reverse=True);visible=[]
        for box in candidates:
            if not any(self._is_duplicate(box,other) for other in visible):visible.append(box)
        return visible
