from dataclasses import dataclass, field
from collections import deque
import time
import numpy as np
from .motion import BoxMotionModel
from .schemas import TrackState, TrackedPerson

@dataclass
class Track:
    camera_id: str
    local_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    first_frame_id: int
    created_at: float
    last_seen_at: float
    last_frame_id: int
    state: TrackState = TrackState.TENTATIVE
    age_frames: int = 1
    hits: int = 1
    misses: int = 0
    high_confidence_hits: int = 1
    appearance_embedding: np.ndarray | None = None
    appearance_version: int = 0
    appearance_quality: float = 0.0
    appearance_frame_id: int | None = None
    appearance_timestamp: float | None = None
    removed_at_monotonic: float | None = None
    detection_source: str = "FULL_FRAME"
    generation: int = 1
    source_width: int = 0
    source_height: int = 0
    visual_bbox: np.ndarray = field(init=False, repr=False)
    visual_velocity: np.ndarray = field(init=False, repr=False)
    real_observations: deque = field(init=False, repr=False)
    last_observed_velocity: np.ndarray = field(init=False, repr=False)
    last_real_observation_monotonic: float = field(init=False, repr=False)
    boundary_exit_candidate: bool = field(default=False, init=False)
    boundary_exit_started_monotonic: float | None = field(default=None, init=False, repr=False)
    visual_hidden: bool = field(default=False, init=False)
    visual_hidden_at_monotonic: float | None = field(default=None, init=False, repr=False)
    visual_hidden_at_timestamp: float | None = field(default=None, init=False, repr=False)
    detection_id: str | None = None
    motion: BoxMotionModel = field(init=False, repr=False)
    motion_monotonic: float = field(init=False, repr=False)
    last_detection_monotonic: float = field(init=False, repr=False)

    def __post_init__(self):
        self.motion=BoxMotionModel(self.bbox);self.motion_monotonic=time.monotonic();self.last_detection_monotonic=self.motion_monotonic
        self.last_real_observation_monotonic=self.motion_monotonic
        self.visual_bbox=np.asarray(self.bbox,np.float32);self.visual_velocity=np.zeros(4,np.float32);self.last_observed_velocity=np.zeros(4,np.float32)
        measurement=np.asarray(self.motion._measurement(self.bbox),np.float32)
        self.real_observations=deque(((self.last_real_observation_monotonic,measurement),),maxlen=3)
    @property
    def track_id(self): return f"{self.camera_id}:TRACK-{self.local_id:05d}"
    @property
    def velocity(self): return tuple(float(v) for v in self.visual_velocity)
    def predict(self,now_monotonic=None):
        current=time.monotonic() if now_monotonic is None else float(now_monotonic);return tuple(float(v) for v in self.motion.predict(max(0.0,current-self.motion_monotonic)))
    def predict_visual(self,now_monotonic=None,damping_start_ms=200.0,horizon_ms=650.0):
        """Project source-pixel geometry once, using source pixels/second."""
        current=time.monotonic() if now_monotonic is None else float(now_monotonic);dt=max(0.0,current-self.last_real_observation_monotonic);age_ms=dt*1000.0
        if age_ms<=damping_start_ms:factor=1.0
        else:factor=max(0.0,1.0-(age_ms-damping_start_ms)/max(1.0,horizon_ms-damping_start_ms))
        measurement=np.asarray(self.motion._measurement(self.visual_bbox),np.float32)
        observed_prediction=measurement+self.visual_velocity*(dt*factor)
        legacy_bbox=self.motion.predict(max(0.0,current-self.motion_monotonic));legacy_prediction=np.asarray(self.motion._measurement(legacy_bbox),np.float32)
        # Live backtests show observation-derived motion wins at short horizons,
        # while its acceleration/noise tail is less reliable after 500-800 ms.
        if age_ms<=300.0:observation_weight=1.0
        elif age_ms<=500.0:observation_weight=1.0-(age_ms-300.0)*.3/200.0
        elif age_ms<800.0:observation_weight=.7*(800.0-age_ms)/300.0
        else:observation_weight=0.0
        predicted=observation_weight*observed_prediction+(1.0-observation_weight)*legacy_prediction
        return tuple(float(v) for v in self.motion._bbox(predicted))

    def prediction_backtest(self,bbox,target_monotonic,horizon_ms=650.0):
        current=float(target_monotonic);dt=max(0.0,current-self.last_real_observation_monotonic)
        legacy=tuple(float(v) for v in self.motion.predict(max(0.0,current-self.motion_monotonic)))
        corrected=self.predict_visual(current,horizon_ms=horizon_ms)
        return {"track_id":self.track_id,"horizon_ms":dt*1000.0,"previous_bbox":tuple(float(v) for v in self.visual_bbox),"legacy_bbox":legacy,"corrected_bbox":corrected,"actual_bbox":tuple(float(v) for v in bbox),"predicted_velocity":self.velocity,"previous_confidence":float(self.confidence)}

    @staticmethod
    def _visible_ratio(bbox,width,height):
        x1,y1,x2,y2=map(float,bbox);area=max(1.0,(x2-x1)*(y2-y1))
        intersection=max(0.0,min(x2,width)-max(x1,0.0))*max(0.0,min(y2,height)-max(y1,0.0))
        return intersection/area

    def evaluate_boundary_exit(self,now_monotonic=None):
        current=time.monotonic() if now_monotonic is None else float(now_monotonic)
        if not self.source_width or not self.source_height or self.misses<=0:return False,False,0.0,1.0
        x1,y1,x2,y2=map(float,self.visual_bbox);width=float(self.source_width);height=float(self.source_height)
        vx,vy=self.velocity[:2];box_w=max(1.0,x2-x1);box_h=max(1.0,y2-y1)
        threshold_x=max(15.0,box_w*.05);threshold_y=max(15.0,box_h*.05)
        margin_x=max(20.0,width*.08);margin_y=max(20.0,height*.08)
        near_outward=(x1<=margin_x and vx < -threshold_x) or (x2>=width-margin_x and vx>threshold_x) or (y1<=margin_y and vy < -threshold_y) or (y2>=height-margin_y and vy>threshold_y)
        if not near_outward:
            self.boundary_exit_candidate=False;self.boundary_exit_started_monotonic=None
            return False,False,0.0,1.0
        # Visual-exit latency starts when all boundary evidence first agrees,
        # not at the last detector observation. Detector cadence/age is not
        # reliable exit evidence and previously made the reported delay exceed
        # the 250-600 ms visual-retention policy before a candidate even began.
        if self.boundary_exit_started_monotonic is None:self.boundary_exit_started_monotonic=current
        self.boundary_exit_candidate=True
        predicted=self.predict_visual(current,damping_start_ms=10_000.0,horizon_ms=10_001.0);ratio=self._visible_ratio(predicted,width,height)
        px1,py1,px2,py2=predicted;pcx=(px1+px2)*.5;pcy=(py1+py2)*.5
        crossed=pcx<0 or pcx>width or pcy<0 or pcy>height
        delay_ms=max(0.0,(current-self.boundary_exit_started_monotonic)*1000.0)
        hide=(delay_ms>=250.0 and (ratio<=.55 or crossed)) or (delay_ms>=600.0 and ratio<.80)
        if hide and not self.visual_hidden:self.visual_hidden=True;self.visual_hidden_at_monotonic=current;self.visual_hidden_at_timestamp=self.last_seen_at+(current-self.last_real_observation_monotonic)
        return True,self.visual_hidden,delay_ms,ratio

    def update(self, bbox, confidence, frame_id, timestamp, embedding=None, min_hits=3, confirmation_evidence=False, now_monotonic=None, detection_source="FULL_FRAME", detection_id=None):
        recovered = self.state == TrackState.LOST;current=time.monotonic() if now_monotonic is None else float(now_monotonic)
        measurement=np.asarray(self.motion._measurement(bbox),np.float32)
        samples=[*self.real_observations,(current,measurement)]
        slopes=[(right[1]-left[1])/max(.001,right[0]-left[0]) for left,right in zip(samples,samples[1:]) if right[0]-left[0]>.001]
        observed_velocity = None
        motion_threshold=max(12.0,min(float(measurement[2]),float(measurement[3]))*.15)
        if len(slopes)>=2:
            previous_velocity,latest_velocity=np.asarray(slopes[-2],np.float32),np.asarray(slopes[-1],np.float32)
            previous_speed=float(np.linalg.norm(previous_velocity[:2]));latest_speed=float(np.linalg.norm(latest_velocity[:2]))
            reversal=float(np.dot(previous_velocity[:2],latest_velocity[:2]))<0 and min(previous_speed,latest_speed)>motion_threshold
            stopped=latest_speed<=motion_threshold and previous_speed>motion_threshold*2.0
            started=previous_speed<=motion_threshold and latest_speed>motion_threshold*2.0
            if reversal or stopped or started:observed_velocity=latest_velocity
        if observed_velocity is None:
            observed_velocity=np.median(np.asarray(slopes,np.float32),axis=0) if slopes else np.zeros(4,np.float32)
        self.motion.update(bbox,max(0.0,current-self.motion_monotonic))
        limit=max(float(measurement[2]),float(measurement[3]),1.0)*4.0
        observed_velocity=np.clip(observed_velocity,-limit,limit);self.last_observed_velocity=observed_velocity.copy()
        kalman_velocity=np.clip(np.asarray(self.motion.state[4:],np.float32),-limit,limit);old_speed=float(np.linalg.norm(self.visual_velocity[:2]));observed_speed=float(np.linalg.norm(observed_velocity[:2]))
        direction=float(np.dot(self.visual_velocity[:2],observed_velocity[:2]))
        if direction<0 and min(old_speed,observed_speed)>motion_threshold:observed_weight=.90
        elif observed_speed<=motion_threshold and old_speed>motion_threshold*2.0:observed_weight=.85
        elif old_speed<=motion_threshold and observed_speed>motion_threshold*2.0:observed_weight=.80
        else:observed_weight=.70
        self.visual_velocity=observed_weight*observed_velocity+(1.0-observed_weight)*kalman_velocity;self.motion.state[4:]=.65*kalman_velocity+.35*self.visual_velocity
        if recovered:self.motion.state[:4]=measurement;self.motion.bbox=self.motion._bbox(self.motion.state)
        self.real_observations.append((current,measurement.copy()));self.visual_bbox=np.asarray(bbox,np.float32)
        self.motion_monotonic=current;self.last_detection_monotonic=current;self.last_real_observation_monotonic=current
        self.bbox=tuple(float(v) for v in self.visual_bbox)
        self.boundary_exit_candidate=False;self.boundary_exit_started_monotonic=None
        self.visual_hidden=False;self.visual_hidden_at_monotonic=None;self.visual_hidden_at_timestamp=None
        self.confidence = float(confidence); self.last_frame_id = frame_id; self.last_seen_at = timestamp; self.detection_source=str(detection_source); self.detection_id=detection_id
        self.age_frames += 1; self.hits += 1; self.misses = 0
        if confirmation_evidence:self.high_confidence_hits += 1
        if self.high_confidence_hits >= min_hits:self.state = TrackState.CONFIRMED
        if embedding is not None:
            emb = np.asarray(embedding, np.float32).reshape(-1); norm = float(np.linalg.norm(emb))
            if emb.size<2 or not np.all(np.isfinite(emb)) or not np.isfinite(norm) or norm<=1e-12:emb=None
            if emb is not None:emb /= norm
            if emb is not None and self.appearance_embedding is not None and self.appearance_embedding.shape == emb.shape:
                emb = .8 * self.appearance_embedding + .2 * emb; emb /= max(np.linalg.norm(emb), 1e-12)
            if emb is not None:
                self.appearance_version += 1;self.appearance_embedding = emb
        return recovered

    def miss(self,now_monotonic=None):
        current=time.monotonic() if now_monotonic is None else float(now_monotonic);self.motion.miss(max(0.0,current-self.motion_monotonic));self.motion_monotonic=current; self.bbox = tuple(float(v) for v in self.motion.bbox)
        self.age_frames += 1; self.misses += 1
        if self.state in (TrackState.CONFIRMED,TrackState.LOST):self.state = TrackState.LOST
    def output(self, now=None, now_monotonic=None, observation_type=None, visual_horizon_ms=None):
        now=self.last_seen_at if now is None else float(now);current=time.monotonic() if now_monotonic is None else float(now_monotonic);age_ms=max(0.0,(current-self.last_real_observation_monotonic)*1000.0);kind=observation_type or ("predicted" if self.misses else "detected")
        geometry=self.predict_visual(current,horizon_ms=visual_horizon_ms or 650.0) if kind=="predicted" else tuple(float(v) for v in self.visual_bbox)
        state_timestamp=now if kind=="predicted" else self.last_seen_at;geometry_monotonic=current if kind=="predicted" else self.last_real_observation_monotonic
        expires_at=self.last_seen_at+(float(visual_horizon_ms or 0.0)/1000.0)
        return TrackedPerson(track_id=self.track_id,state=self.state,bbox=geometry,confidence=self.confidence,age_frames=self.age_frames,hits=self.hits,misses=self.misses,velocity=self.velocity,camera_id=self.camera_id,local_track_id=self.local_id,first_seen=self.created_at,last_seen=self.last_seen_at,age_seconds=max(0.0,now-self.created_at),lost_duration=max(0.0,now-self.last_seen_at) if self.state==TrackState.LOST else 0.0,predicted_bbox=geometry,appearance_version=self.appearance_version,confirmed=self.state==TrackState.CONFIRMED,observation_type=kind,last_detection_timestamp=self.last_seen_at,prediction_age_ms=age_ms,detection_source=self.detection_source if kind!="predicted" else "PREDICTED",detection_id=self.detection_id,state_timestamp=state_timestamp,visual_expires_at=expires_at,track_generation=self.generation,geometry_monotonic=geometry_monotonic,visual_visible=not self.visual_hidden,boundary_exit=self.boundary_exit_candidate)
