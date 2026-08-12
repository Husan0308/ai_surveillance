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
    vx1: float = 0.0
    vy1: float = 0.0
    vx2: float = 0.0
    vy2: float = 0.0


def _width(box: VisualBox) -> float:
    return max(0.0, box.x2 - box.x1)


def _height(box: VisualBox) -> float:
    return max(0.0, box.y2 - box.y1)


def _area(box: VisualBox) -> float:
    return _width(box) * _height(box)


def _intersection(a: VisualBox, b: VisualBox) -> float:
    x1=max(a.x1,b.x1);y1=max(a.y1,b.y1);x2=min(a.x2,b.x2);y2=min(a.y2,b.y2)
    return max(0.0,x2-x1)*max(0.0,y2-y1)


def _iou(a: VisualBox, b: VisualBox) -> float:
    inter=_intersection(a,b)
    union=_area(a)+_area(b)-inter
    return inter/union if union>0 else 0.0


def _containment(a: VisualBox, b: VisualBox) -> float:
    inter=_intersection(a,b)
    smaller=min(_area(a),_area(b))
    return inter/smaller if smaller>0 else 0.0


def _center_distance(a: VisualBox,b: VisualBox) -> float:
    acx=(a.x1+a.x2)*0.5;acy=(a.y1+a.y2)*0.5
    bcx=(b.x1+b.x2)*0.5;bcy=(b.y1+b.y2)*0.5
    scale=max(20.0,math.sqrt(max(_area(a),_area(b))))
    return math.hypot(acx-bcx,acy-bcy)/scale


def _horizontal_overlap_ratio(a:VisualBox,b:VisualBox) -> float:
    overlap=max(0.0,min(a.x2,b.x2)-max(a.x1,b.x1))
    smaller=max(1.0,min(_width(a),_width(b)))
    return overlap/smaller


def _vertical_overlap_ratio(a:VisualBox,b:VisualBox) -> float:
    overlap=max(0.0,min(a.y2,b.y2)-max(a.y1,b.y1))
    smaller=max(1.0,min(_height(a),_height(b)))
    return overlap/smaller


def _vertical_gap_ratio(a:VisualBox,b:VisualBox) -> float:
    if a.y2 < b.y1:
        gap=b.y1-a.y2
    elif b.y2 < a.y1:
        gap=a.y1-b.y2
    else:
        gap=0.0
    return gap/max(1.0,max(_height(a),_height(b)))


def _x_center_ratio(a:VisualBox,b:VisualBox) -> float:
    acx=(a.x1+a.x2)*0.5;bcx=(b.x1+b.x2)*0.5
    return abs(acx-bcx)/max(1.0,max(_width(a),_width(b)))


def _area_ratio(a:VisualBox,b:VisualBox) -> float:
    aa=_area(a);bb=_area(b)
    return min(aa,bb)/max(1.0,max(aa,bb))


class VisualTracker:
    """Visual-only per-camera continuity and latency compensation layer.

    Detector boxes belong to an older captured frame. The display, however,
    renders the newest camera frame. Tracks therefore estimate short-term box
    velocity from detector *capture timestamps* and project the box to the
    timestamp of the frame currently being displayed. This removes the visible
    "box trailing behind the walking person" effect without feeding prediction
    into identity/ReID.
    """

    def __init__(self, *, hold_ms=1600, memory_ms=6000, prediction_ms=450,
                 match_iou=0.16, reacquire_distance=1.05,
                 duplicate_iou=0.50, duplicate_containment=0.82,
                 duplicate_center_distance=0.40,
                 fragment_duplicate=False,
                 fragment_horizontal_overlap=0.70,
                 fragment_x_center=0.30,
                 fragment_max_area_ratio=0.70,
                 fragment_min_vertical_overlap=0.12,
                 fragment_max_vertical_gap=0.10,
                 smoothing=0.88, velocity_smoothing=0.58,
                 max_prediction_shift_boxes=1.35,
                 low_conf_confirm=0.16, start_conf=0.18,
                 exclusion_zones=None, exclusion_max_box_height=0.34):
        self.hold_sec=max(0.05,float(hold_ms)/1000.0)
        self.memory_sec=max(self.hold_sec,float(memory_ms)/1000.0)
        self.prediction_sec=max(0.0,float(prediction_ms)/1000.0)
        self.match_iou=float(match_iou)
        self.reacquire_distance=max(0.1,float(reacquire_distance))
        self.duplicate_iou=float(duplicate_iou)
        self.duplicate_containment=float(duplicate_containment)
        self.duplicate_center_distance=max(0.0,float(duplicate_center_distance))
        self.fragment_duplicate=bool(fragment_duplicate)
        self.fragment_horizontal_overlap=float(fragment_horizontal_overlap)
        self.fragment_x_center=float(fragment_x_center)
        self.fragment_max_area_ratio=float(fragment_max_area_ratio)
        self.fragment_min_vertical_overlap=float(fragment_min_vertical_overlap)
        self.fragment_max_vertical_gap=float(fragment_max_vertical_gap)
        self.smoothing=min(0.98,max(0.05,float(smoothing)))
        self.velocity_smoothing=min(0.95,max(0.05,float(velocity_smoothing)))
        self.max_prediction_shift_boxes=max(0.25,float(max_prediction_shift_boxes))
        self.low_conf_confirm=float(low_conf_confirm)
        self.start_conf=float(start_conf)
        self.exclusion_zones=[tuple(float(v) for v in zone[:4]) for zone in (exclusion_zones or []) if len(zone)>=4]
        self.exclusion_max_box_height=max(0.0,float(exclusion_max_box_height))
        self._tracks:dict[int,_Track]={}
        self._next_id=1
        self._last_result_frame_id=-1
        self._weak_candidates:list[tuple[VisualBox,float,int]]=[]

    def _excluded(self, box:VisualBox, source_width:float|None, source_height:float|None) -> bool:
        if not self.exclusion_zones or not source_width or not source_height:return False
        width=max(1.0,float(source_width));height=max(1.0,float(source_height))
        cx=((box.x1+box.x2)*0.5)/width;cy=((box.y1+box.y2)*0.5)/height
        box_h=max(0.0,box.y2-box.y1)/height
        if box_h>self.exclusion_max_box_height:return False
        return any(x1<=cx<=x2 and y1<=cy<=y2 for x1,y1,x2,y2 in self.exclusion_zones)

    def _fragment_duplicate_match(self, box:VisualBox, other:VisualBox) -> bool:
        if not self.fragment_duplicate:return False
        if _area_ratio(box,other)>self.fragment_max_area_ratio:return False
        if _horizontal_overlap_ratio(box,other)<self.fragment_horizontal_overlap:return False
        if _x_center_ratio(box,other)>self.fragment_x_center:return False
        return (
            _vertical_overlap_ratio(box,other)>=self.fragment_min_vertical_overlap or
            _vertical_gap_ratio(box,other)<=self.fragment_max_vertical_gap
        )

    def _is_duplicate(self, box:VisualBox, other:VisualBox) -> bool:
        iou=_iou(box,other)
        containment=_containment(box,other)
        center_distance=_center_distance(box,other)
        return (
            iou>=self.duplicate_iou or
            (containment>=self.duplicate_containment and center_distance<=self.duplicate_center_distance) or
            self._fragment_duplicate_match(box,other)
        )

    def _dedupe(self, boxes, source_width=None, source_height=None) -> list[VisualBox]:
        candidates=[VisualBox(float(b.x1),float(b.y1),float(b.x2),float(b.y2),float(b.confidence)) for b in boxes]
        candidates=[b for b in candidates if not self._excluded(b,source_width,source_height)]
        candidates.sort(key=lambda b:b.confidence,reverse=True)
        kept=[]
        for box in candidates:
            if any(self._is_duplicate(box,other) for other in kept):continue
            kept.append(box)
        return kept

    def _predicted_box(self, track:_Track, target_time:float) -> VisualBox:
        dt=min(max(0.0,float(target_time)-track.last_observation),self.prediction_sec)
        max_shift=max(20.0,max(_width(track.box),_height(track.box))*self.max_prediction_shift_boxes)
        shifts=(track.vx1*dt,track.vy1*dt,track.vx2*dt,track.vy2*dt)
        shifts=tuple(max(-max_shift,min(max_shift,value)) for value in shifts)
        return VisualBox(
            track.box.x1+shifts[0],track.box.y1+shifts[1],
            track.box.x2+shifts[2],track.box.y2+shifts[3],
            track.box.confidence,
        )

    def _confirm_weak_new_track(self, det:VisualBox, now:float) -> bool:
        fresh=[];matched=None
        for box,ts,hits in self._weak_candidates:
            if now-ts<=1.5:
                fresh.append((box,ts,hits))
                if matched is None and (_iou(box,det)>=0.20 or _center_distance(box,det)<=0.55):
                    matched=(box,ts,hits)
        self._weak_candidates=fresh
        if matched is None:
            self._weak_candidates.append((det,now,1));return False
        self._weak_candidates.remove(matched)
        hits=matched[2]+1
        if hits>=2:return True
        self._weak_candidates.append((det,now,hits));return False

    def update(self, result, now:float, source_width=None, source_height=None) -> None:
        if result is None or int(result.frame_id)==self._last_result_frame_id:return
        self._last_result_frame_id=int(result.frame_id)
        detections=self._dedupe(result.boxes,source_width,source_height)
        observation=float(getattr(result,'frame_captured_monotonic',now) or now)
        if not math.isfinite(observation) or observation<=0:observation=now

        for tid in list(self._tracks):
            if now-self._tracks[tid].last_seen>self.memory_sec:
                del self._tracks[tid]

        unmatched_tracks=set(self._tracks)
        unmatched_dets=set(range(len(detections)))
        pairs=[]
        for tid,track in self._tracks.items():
            # Compare the new detector observation with where the previous track
            # should have been at the timestamp of that observation, not at wall
            # clock 'now'. This avoids matching against a box projected too far.
            reference=self._predicted_box(track,observation)
            dormant=(now-track.last_seen)>self.hold_sec
            max_distance=self.reacquire_distance if dormant else 0.82
            for di,det in enumerate(detections):
                iou=_iou(reference,det);dist=_center_distance(reference,det)
                if iou>=self.match_iou or dist<=max_distance:
                    score=iou+(max(0.0,max_distance-dist)*0.30)-(0.08 if dormant else 0.0)
                    pairs.append((score,tid,di))
        pairs.sort(reverse=True)

        matched_tracks=set()
        for _score,tid,di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:continue
            track=self._tracks[tid];det=detections[di]
            old=track.raw_box
            obs_dt=observation-track.last_observation
            if obs_dt<=0.005:
                obs_dt=max(0.01,now-track.last_update)
            nv=((det.x1-old.x1)/obs_dt,(det.y1-old.y1)/obs_dt,(det.x2-old.x2)/obs_dt,(det.y2-old.y2)/obs_dt)
            va=self.velocity_smoothing;vb=1.0-va
            track.vx1=vb*track.vx1+va*nv[0]
            track.vy1=vb*track.vy1+va*nv[1]
            track.vx2=vb*track.vx2+va*nv[2]
            track.vy2=vb*track.vy2+va*nv[3]

            # Favor the newest measurement strongly. Heavy smoothing on a sparse
            # detector is another source of visible trailing.
            a=self.smoothing;b=1.0-a
            track.box=VisualBox(
                b*track.box.x1+a*det.x1,b*track.box.y1+a*det.y1,
                b*track.box.x2+a*det.x2,b*track.box.y2+a*det.y2,
                max(det.confidence,track.box.confidence*0.82),
            )
            track.raw_box=det;track.last_seen=now;track.last_update=now;track.last_observation=observation;track.hits+=1
            matched_tracks.add(tid)
            unmatched_tracks.remove(tid);unmatched_dets.remove(di)

        for di in list(unmatched_dets):
            det=detections[di]
            if any(self._is_duplicate(det,self._tracks[tid].box) for tid in matched_tracks):
                unmatched_dets.remove(di)

        for di in unmatched_dets:
            det=detections[di]
            if det.confidence<self.start_conf and not self._confirm_weak_new_track(det,now):continue
            tid=self._next_id;self._next_id+=1
            self._tracks[tid]=_Track(
                track_id=tid,box=det,raw_box=det,last_seen=now,last_update=now,
                last_observation=observation,
            )

    def visible(self, now:float, target_time:float|None=None) -> list[VisualBox]:
        target=float(target_time if target_time is not None else now)
        candidates=[]
        for track in self._tracks.values():
            age=now-track.last_seen
            if age>self.hold_sec:continue
            if track.hits<2 and track.box.confidence<self.low_conf_confirm:continue
            predicted=self._predicted_box(track,target)
            predicted.confidence=max(0.01,track.box.confidence*(1.0-0.30*min(1.0,age/self.hold_sec)))
            candidates.append(predicted)

        candidates.sort(key=lambda b:b.confidence,reverse=True)
        visible=[]
        for box in candidates:
            if any(self._is_duplicate(box,other) for other in visible):continue
            visible.append(box)
        return visible
