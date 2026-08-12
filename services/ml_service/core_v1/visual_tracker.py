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
    hits: int = 1
    vx1: float = 0.0
    vy1: float = 0.0
    vx2: float = 0.0
    vy2: float = 0.0


def _area(box: VisualBox) -> float:
    return max(0.0, box.x2 - box.x1) * max(0.0, box.y2 - box.y1)


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


class VisualTracker:
    """Visual-only per-camera continuity layer.

    A real detector observation creates/refreshes a track. A short detector miss
    does not make the box blink off immediately. After the visible hold expires
    the track becomes dormant (not drawn) but remains in memory, so a person who
    walks fully behind a chair/desk can be associated and shown immediately when
    visible again. No dormant/predicted box is ever identity/ReID evidence.
    """

    def __init__(self, *, hold_ms=1600, memory_ms=6000, prediction_ms=450,
                 match_iou=0.16, reacquire_distance=1.05,
                 duplicate_iou=0.55, duplicate_containment=0.88,
                 smoothing=0.72, low_conf_confirm=0.16,
                 start_conf=0.18):
        self.hold_sec=max(0.05,float(hold_ms)/1000.0)
        self.memory_sec=max(self.hold_sec,float(memory_ms)/1000.0)
        self.prediction_sec=max(0.0,float(prediction_ms)/1000.0)
        self.match_iou=float(match_iou)
        self.reacquire_distance=max(0.1,float(reacquire_distance))
        self.duplicate_iou=float(duplicate_iou)
        self.duplicate_containment=float(duplicate_containment)
        self.smoothing=min(0.95,max(0.05,float(smoothing)))
        self.low_conf_confirm=float(low_conf_confirm)
        self.start_conf=float(start_conf)
        self._tracks:dict[int,_Track]={}
        self._next_id=1
        self._last_result_frame_id=-1
        self._weak_candidates:list[tuple[VisualBox,float,int]]=[]

    def _dedupe(self, boxes) -> list[VisualBox]:
        candidates=[VisualBox(float(b.x1),float(b.y1),float(b.x2),float(b.y2),float(b.confidence)) for b in boxes]
        candidates.sort(key=lambda b:b.confidence,reverse=True)
        kept=[]
        for box in candidates:
            if any(_iou(box,other)>=self.duplicate_iou or _containment(box,other)>=self.duplicate_containment for other in kept):
                continue
            kept.append(box)
        return kept

    def _predicted_box(self, track:_Track, now:float) -> VisualBox:
        age=max(0.0,now-track.last_seen);dt=min(age,self.prediction_sec)
        return VisualBox(
            track.box.x1+track.vx1*dt,track.box.y1+track.vy1*dt,
            track.box.x2+track.vx2*dt,track.box.y2+track.vy2*dt,
            track.box.confidence,
        )

    def _confirm_weak_new_track(self, det:VisualBox, now:float) -> bool:
        """Allow repeated weak people to start without accepting one-frame noise."""
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

    def update(self, result, now:float) -> None:
        if result is None or int(result.frame_id)==self._last_result_frame_id:return
        self._last_result_frame_id=int(result.frame_id)
        detections=self._dedupe(result.boxes)

        # Keep dormant tracks for reacquisition, but never forever.
        for tid in list(self._tracks):
            if now-self._tracks[tid].last_seen>self.memory_sec:
                del self._tracks[tid]

        unmatched_tracks=set(self._tracks)
        unmatched_dets=set(range(len(detections)))
        pairs=[]
        for tid,track in self._tracks.items():
            reference=self._predicted_box(track,now)
            dormant=(now-track.last_seen)>self.hold_sec
            max_distance=self.reacquire_distance if dormant else 0.78
            for di,det in enumerate(detections):
                iou=_iou(reference,det);dist=_center_distance(reference,det)
                if iou>=self.match_iou or dist<=max_distance:
                    # Prefer overlap, then spatial proximity; dormant tracks get a
                    # slight penalty so a currently visible track wins first.
                    score=iou+(max(0.0,max_distance-dist)*0.30)-(0.08 if dormant else 0.0)
                    pairs.append((score,tid,di))
        pairs.sort(reverse=True)

        for _score,tid,di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:continue
            track=self._tracks[tid];det=detections[di]
            dt=max(0.01,now-track.last_update);old=track.raw_box
            velocity_alpha=0.40
            nv=((det.x1-old.x1)/dt,(det.y1-old.y1)/dt,(det.x2-old.x2)/dt,(det.y2-old.y2)/dt)
            track.vx1=(1-velocity_alpha)*track.vx1+velocity_alpha*nv[0]
            track.vy1=(1-velocity_alpha)*track.vy1+velocity_alpha*nv[1]
            track.vx2=(1-velocity_alpha)*track.vx2+velocity_alpha*nv[2]
            track.vy2=(1-velocity_alpha)*track.vy2+velocity_alpha*nv[3]
            a=self.smoothing;b=1.0-a
            track.box=VisualBox(
                b*track.box.x1+a*det.x1,b*track.box.y1+a*det.y1,
                b*track.box.x2+a*det.x2,b*track.box.y2+a*det.y2,
                max(det.confidence,track.box.confidence*0.82),
            )
            track.raw_box=det;track.last_seen=now;track.last_update=now;track.hits+=1
            unmatched_tracks.remove(tid);unmatched_dets.remove(di)

        for di in unmatched_dets:
            det=detections[di]
            # Strong observations start immediately. Weak observations need to
            # repeat spatially before becoming a visible track. This lets detector
            # conf stay low for far people without making random chair-like noise
            # blink into existence from one frame.
            if det.confidence<self.start_conf and not self._confirm_weak_new_track(det,now):
                continue
            tid=self._next_id;self._next_id+=1
            self._tracks[tid]=_Track(tid,det,det,now,now)

    def visible(self, now:float) -> list[VisualBox]:
        visible=[]
        for track in self._tracks.values():
            age=now-track.last_seen
            if age>self.hold_sec:continue
            if track.hits<2 and track.box.confidence<self.low_conf_confirm:continue
            predicted=self._predicted_box(track,now)
            conf=max(0.01,track.box.confidence*(1.0-0.30*min(1.0,age/self.hold_sec)))
            predicted.confidence=conf
            visible.append(predicted)
        return visible
