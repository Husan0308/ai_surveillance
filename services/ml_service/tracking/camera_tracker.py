"""Independent custom two-stage motion/IoU tracker for one camera."""
import threading
import numpy as np
import time
from .disappearance_logger import disappearance_auditor
from .association import iou_matrix, greedy_match, appearance_matrix, motion_proximity_matrix,association_features
from .track import Track
from .schemas import TrackState, CameraTrackResult
from .metrics import CameraTrackingMetrics

def duplicate_observation(a,b):
    """Conservative same-observation geometry; adjacent people remain distinct."""
    ax1,ay1,ax2,ay2=map(float,a);bx1,by1,bx2,by2=map(float,b)
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    area_a=max(0,ax2-ax1)*max(0,ay2-ay1);area_b=max(0,bx2-bx1)*max(0,by2-by1);union=area_a+area_b-inter
    iou=inter/union if union else 0.0;smaller=inter/max(1.0,min(area_a,area_b))
    acx,acy=(ax1+ax2)*.5,(ay1+ay2)*.5;bcx,bcy=(bx1+bx2)*.5,(by1+by2)*.5
    scale=max(1.0,min(((ax2-ax1)**2+(ay2-ay1)**2)**.5,((bx2-bx1)**2+(by2-by1)**2)**.5))
    center=((acx-bcx)**2+(acy-bcy)**2)**.5/scale
    return (iou>=.62 and center<=.25) or (smaller>=.82 and center<=.22)

def prediction_quality(predicted,actual):
    px1,py1,px2,py2=map(float,predicted);ax1,ay1,ax2,ay2=map(float,actual)
    pcx,pcy=(px1+px2)*.5,(py1+py2)*.5;acx,acy=(ax1+ax2)*.5,(ay1+ay2)*.5
    center_error=((pcx-acx)**2+(pcy-acy)**2)**.5
    intersection=max(0.0,min(px2,ax2)-max(px1,ax1))*max(0.0,min(py2,ay2)-max(py1,ay1))
    predicted_area=max(0.0,px2-px1)*max(0.0,py2-py1);actual_area=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    iou=intersection/max(1.0,predicted_area+actual_area-intersection)
    normalized=((pcx-acx)/max(1.0,ax2-ax1))**2+((pcy-acy)/max(1.0,ay2-ay1))**2
    return {"center_error_px":center_error,"center_error_normalized":normalized**.5,"iou":iou,"bbox_width_error":abs((px2-px1)-(ax2-ax1)),"bbox_height_error":abs((py2-py1)-(ay2-ay1))}

class CameraTracker:
    def __init__(self, camera_id, config=None):
        cfg = config or {}; self.camera_id = camera_id; self._lock = threading.RLock()
        self.high = float(cfg.get("track_high_thresh", .22)); self.low = float(cfg.get("track_low_thresh", .05))
        self.new_threshold = float(cfg.get("new_track_thresh", .28)); self.match = float(cfg.get("match_thresh", .35))
        self.relaxed_match = float(cfg.get("relaxed_match_thresh", max(.1, self.match * .6)))
        self.min_hits = int(cfg.get("min_confirmed_hits", 3)); self.nominal_fps = max(1.0, float(cfg.get("effective_ai_fps", 10)))
        self.max_lost_ms = float(cfg.get("max_lost_time_ms", float(cfg.get("lost_memory_seconds", 1.5)) * 1000))
        self.max_lost_frames = int(cfg.get("max_lost_frames", max(1, int(round(self.max_lost_ms * self.nominal_fps / 1000)))))
        self.min_expiry_misses=max(1,int(cfg.get("min_expiry_misses",3)))
        # A confirmed track must remain publishable for its entire bounded
        # retention window; otherwise it is invisible before it is expired.
        self.observation_correction=bool(cfg.get("observation_centric_association",True))
        self.ambiguity_margin=max(0.0,float(cfg.get("association_ambiguity_margin",.06)))
        self.appearance_weight=max(0.0,min(.25,float(cfg.get("association_appearance_weight",.12))))
        self.prediction_horizon_ms=max(self.max_lost_ms,float(cfg.get("prediction_horizon_ms",650)))
        self.recovery_motion_enabled = bool(cfg.get("recovery_motion_enabled", True)); self.recovery_max_distance = float(cfg.get("recovery_max_normalized_distance", 1.5))
        self.tombstone_recovery_ms=float(cfg.get("tombstone_recovery_ms",3000));self.tombstone_recovery_iou=float(cfg.get("tombstone_recovery_iou",.70))
        self.tombstone_extended_ms=float(cfg.get("tombstone_extended_ms",5000));self.tombstone_extended_iou=float(cfg.get("tombstone_extended_iou",.90))
        self.new_min_width=float(cfg.get("new_track_min_width",12));self.new_min_height=float(cfg.get("new_track_min_height",36));self.new_min_area=float(cfg.get("new_track_min_area",432));self.new_min_ratio=float(cfg.get("new_track_min_aspect_ratio",.08));self.new_max_ratio=float(cfg.get("new_track_max_aspect_ratio",3.5))
        self.tracks = []; self._next_id = 1;self._generation=1; self.metrics = CameraTrackingMetrics();self._prediction_started=time.monotonic()
        # A single weak association result must not immediately split a visible
        # confirmed identity. Entries are short-lived and scoped to a plausible
        # predecessor plus a coarse spatial cell.
        self._deferred_admissions = {}

    def _record_prediction(self,audit,track):
        before=prediction_quality(audit["legacy_bbox"],audit["actual_bbox"]);after=prediction_quality(audit["corrected_bbox"],audit["actual_bbox"])
        actual=np.asarray(track.motion._measurement(audit["actual_bbox"]),np.float32);predicted=np.asarray(track.motion._measurement(audit["corrected_bbox"]),np.float32)
        predicted_velocity=np.asarray(audit["predicted_velocity"][:2],np.float32);observed_velocity=np.asarray(track.last_observed_velocity[:2],np.float32)
        observed_speed=float(np.linalg.norm(observed_velocity));velocity_ratio=float(np.linalg.norm(predicted_velocity))/observed_speed if observed_speed>1.0 else 1.0
        previous=np.asarray(audit["legacy_bbox"],np.float32);actual_box=np.asarray(audit["actual_bbox"],np.float32);previous_wh=np.maximum(1.0,previous[2:]-previous[:2]);actual_wh=np.maximum(1.0,actual_box[2:]-actual_box[:2]);scale_delta=float(np.max(np.abs(previous_wh-actual_wh)/actual_wh));direction=float(np.dot(predicted_velocity,observed_velocity))
        causes=[]
        if audit["horizon_ms"]>=800:causes.append("LONG_OBSERVATION_GAP")
        if scale_delta>=.25:causes.append("DETECTION_BOX_JITTER" if abs(float(previous[0]-actual_box[0]))<max(4.0,float(actual_wh[0])*.2) else "SCALE_CHANGE")
        if direction<0 and min(float(np.linalg.norm(predicted_velocity)),observed_speed)>2:causes.append("DIRECTION_CHANGE")
        if velocity_ratio<.7 and observed_speed>2:causes.append("MOTION_UNDERSHOOT")
        elif velocity_ratio>1.3 and observed_speed>2:causes.append("MOTION_OVERSHOOT")
        if audit.get("detection_confidence",1.0)<self.new_threshold:causes.append("LOW_CONFIDENCE_DETECTION")
        if after["center_error_normalized"]>1.0 and audit["horizon_ms"]>800:causes.append("ASSOCIATION_SUSPECT");self.metrics.association_wrong_match_suspected+=1
        audit.update(
            before=before,
            after=after,
            velocity_ratio=velocity_ratio,
            observed_velocity=tuple(float(value) for value in observed_velocity),
            bbox_scale_delta=scale_delta,
            motion_direction_dot=direction,
            classifications=tuple(causes) or ("UNCLASSIFIED",),
        )
        self.metrics.prediction_backtest_count+=1;self.metrics.prediction_backtests.append(audit);del self.metrics.prediction_backtests[:-200]
        samples=self.metrics.prediction_backtests
        def pct(values,fraction):
            ordered=sorted(float(value) for value in values);return ordered[min(len(ordered)-1,int((len(ordered)-1)*fraction))] if ordered else 0.0
        horizons=[item["horizon_ms"] for item in samples];self.metrics.prediction_horizon_ms_p50=pct(horizons,.50);self.metrics.prediction_horizon_ms_p95=pct(horizons,.95)
        before_errors=[item["before"]["center_error_px"] for item in samples];after_errors=[item["after"]["center_error_px"] for item in samples]
        self.metrics.prediction_center_error_px_before_p50=pct(before_errors,.50);self.metrics.prediction_center_error_px_before_p95=pct(before_errors,.95)
        self.metrics.prediction_center_error_px_after_p50=pct(after_errors,.50);self.metrics.prediction_center_error_px_after_p95=pct(after_errors,.95)
        after_norm=[item["after"]["center_error_normalized"] for item in samples]
        self.metrics.prediction_center_error_norm_p50=pct(after_norm,.50);self.metrics.prediction_center_error_norm_p95=pct(after_norm,.95)
        before_iou=[item["before"]["iou"] for item in samples];after_iou=[item["after"]["iou"] for item in samples]
        self.metrics.prediction_iou_before_p50=pct(before_iou,.50);self.metrics.prediction_iou_before_p05=pct(before_iou,.05);self.metrics.prediction_iou_after_p50=pct(after_iou,.50);self.metrics.prediction_iou_after_p05=pct(after_iou,.05)
        ratios=[item["velocity_ratio"] for item in samples];self.metrics.predicted_vs_observed_velocity_ratio_p50=pct(ratios,.50);self.metrics.predicted_vs_observed_velocity_ratio_p95=pct(ratios,.95)
        buckets={}
        for label,low,high in (("0-150",0,150),("150-300",150,300),("300-500",300,500),("500-800",500,800),("800+",800,float("inf"))):
            subset=[item for item in samples if low<=item["horizon_ms"]<high]
            if not subset:continue
            buckets[label]={"count":len(subset),"before_center_error_px_p50":pct([item["before"]["center_error_px"] for item in subset],.50),"before_iou_p50":pct([item["before"]["iou"] for item in subset],.50),"after_center_error_px_p50":pct([item["after"]["center_error_px"] for item in subset],.50),"after_iou_p50":pct([item["after"]["iou"] for item in subset],.50)}
        self.metrics.prediction_horizon_buckets=buckets

    def update(self, result, embeddings=None, now_monotonic=None):
        started=time.perf_counter();now=result.capture_timestamp
        mono=float(now_monotonic) if now_monotonic is not None else float(result.capture_monotonic or time.monotonic())
        detections = list(result.detections); embeddings = embeddings or [None] * len(detections)
        self.metrics.detections=len(detections)
        with self._lock:
            self.metrics.association_last_ambiguity=0.0
            visible_before={track.track_id for track in self.tracks if track.state in (TrackState.CONFIRMED,TrackState.LOST)}
            self._deferred_admissions = {key: value for key, value in self._deferred_admissions.items()
                                         if mono - value[1] <= 1.5}
            # Association must run before expiry: a returning real detection may
            # recover a retained track even after a long scheduling interval.
            active = [t for t in self.tracks if t.state != TrackState.REMOVED]
            motion_started = time.perf_counter(); predicted = [t.predict(mono) for t in active]
            self.metrics.motion_ms = (time.perf_counter() - motion_started) * 1000
            association_started = time.perf_counter(); matched_tracks, matched_detections, ambiguous_detections = set(), set(), set()

            def stage(track_indices, detection_indices, threshold, label, use_appearance=False):
                if not track_indices or not detection_indices:return
                tracks=[active[i] for i in track_indices];selected_detections=[detections[j] for j in detection_indices]
                association_boxes=[track.predict_visual(mono,horizon_ms=self.prediction_horizon_ms) if self.observation_correction else predicted[i] for i,track in zip(track_indices,tracks)]
                features=association_features(association_boxes,[item.bbox_xyxy for item in selected_detections],
                    [track.visual_bbox for track in tracks],[track.velocity for track in tracks],
                    [item.confidence for item in selected_detections],self.recovery_max_distance)
                observation_ages=np.asarray([max(0.0,(mono-track.last_real_observation_monotonic)*1000) for track in tracks],np.float32)[:,None]
                # Preserve short-cadence IoU thresholds. Motion evidence is
                # enabled after a real observation gap, where it prevents splits.
                motion_allowed=(observation_ages>=250.0)|(label=="recovery")
                geometry=np.where(motion_allowed,features["geometry_score"],features["iou"]) if self.recovery_motion_enabled else features["iou"]

                physical_gate=features["geometry_gate"]>.5 if self.recovery_motion_enabled else np.ones(geometry.shape,dtype=bool)
                eligible=physical_gate&(geometry>=threshold)
                # Even when the configured short-cadence score is IoU-only,
                # detect a two-track crossing from physically plausible motion
                # candidates and abstain before appearance can force a choice.
                plausible=np.where(physical_gate,features["geometry_score"],-1.0)
                for column in range(plausible.shape[1]):
                    alternatives=sorted(float(value) for value in plausible[:,column] if value>=threshold)
                    if len(alternatives)>=2 and alternatives[-1]-alternatives[-2]<self.ambiguity_margin:
                        ambiguous_detections.add(detection_indices[column]);eligible[:,column]=False
                        self.metrics.association_ambiguity_abstentions+=1
                scores=geometry.copy();similarity=np.full(scores.shape,np.nan,np.float32);appearance_used=False
                if use_appearance:
                    detection_appearance=[embeddings[j] for j in detection_indices];track_appearance=[track.appearance_embedding for track in tracks]
                    vectors=detection_appearance+track_appearance
                    valid=bool(vectors) and all(value is not None and np.asarray(value).ndim==1 and np.asarray(value).size>1 and np.all(np.isfinite(value)) for value in vectors)
                    valid=valid and len({np.asarray(value).shape for value in vectors})==1
                    if valid:
                        sim_started=time.perf_counter();similarity=appearance_matrix(detection_appearance,track_appearance)
                        appearance_used=True;self.metrics.appearance_similarity_ms+=(time.perf_counter()-sim_started)*1000
                        scores=geometry+self.appearance_weight*(np.clip(similarity,-1.0,1.0)-.5)
                scores=np.where(eligible,scores,-1.0)
                self.metrics.association_candidates_total+=scores.size
                self.metrics.association_geometry_rejections+=int(np.size(eligible)-np.count_nonzero(eligible))
                decisions=[]
                for row,col,score in greedy_match(scores,-.5):
                    if score<0:continue
                    col_alternatives=sorted(float(value) for index,value in enumerate(scores[:,col]) if index!=row and value>=0)
                    runner=col_alternatives[-1] if col_alternatives else -1.0
                    margin=float(score-runner) if runner>=0 else 1.0
                    self.metrics.association_last_ambiguity=max(self.metrics.association_last_ambiguity,max(0.0,self.ambiguity_margin-margin))
                    ambiguous=runner>=0 and margin<self.ambiguity_margin
                    if ambiguous:
                        self.metrics.association_ambiguity_abstentions+=1
                        ambiguous_detections.add(detection_indices[col])
                    decisions.append((row,col,float(score),runner,margin,ambiguous))
                selected_pairs={(row,col) for row,col,_score,_runner,_margin,ambiguous in decisions if not ambiguous}
                for row,track in enumerate(tracks):
                    valid_scores=sorted(float(value) for value in scores[row] if value>=0);runner=valid_scores[-2] if len(valid_scores)>1 else -1.0
                    observation_age=max(0.0,(mono-track.last_real_observation_monotonic)*1000)
                    for col,detection in enumerate(selected_detections):
                        score=float(scores[row,col]);entry={"frame_id":result.frame_id,"stage":label,"track_id":track.track_id,"detection_id":detection.detection_id,
                            "iou":float(features["iou"][row,col]),"normalized_center_distance":float(features["normalized_center_distance"][row,col]),
                            "height_ratio":float(features["height_ratio"][row,col]),"size_ratio":float(features["size_ratio"][row,col]),
                            "velocity_direction_consistency":float(features["direction_cosine"][row,col]),"confidence":float(detection.confidence),
                            "appearance_similarity":None if not np.isfinite(similarity[row,col]) else float(similarity[row,col]),
                            "geometry_passed":bool(eligible[row,col]),"selected_cost":None if score<0 else 1.0-score,
                            "runner_up_cost":None if runner<0 else 1.0-runner,"observation_age_ms":observation_age,
                            "selected":(row,col) in selected_pairs,"abstained":detection_indices[col] in ambiguous_detections or any(item[0]==row and item[1]==col and item[5] for item in decisions)}
                        self.metrics.association_candidates.append(entry)
                del self.metrics.association_candidates[:-500]
                for row,col,_score,_runner,_margin,ambiguous in decisions:
                    if ambiguous:continue
                    ti,di=track_indices[row],detection_indices[col]
                    if ti in matched_tracks or di in matched_detections:continue
                    track=active[ti];audit=track.prediction_backtest(detections[di].bbox_xyxy,mono,self.prediction_horizon_ms)
                    audit.update(previous_real_observation_timestamp=track.last_seen_at,previous_state_timestamp=track.last_seen_at,new_observation_timestamp=now,prediction_horizon_ms=audit["horizon_ms"],frame_id=result.frame_id,detection_confidence=detections[di].confidence,detection_source=detections[di].detection_source,detection_id=detections[di].detection_id)
                    was_hidden=track.visual_hidden
                    recovered=track.update(detections[di].bbox_xyxy,detections[di].confidence,
                                           result.frame_id,now,embeddings[di],self.min_hits,label=="high",mono,detections[di].detection_source,detections[di].detection_id)
                    disappearance_auditor.record_reacquisition(self.camera_id,track.track_id,detections[di].bbox_xyxy)
                    self._record_prediction(audit,track)
                    if was_hidden:self.metrics.visual_reacquisitions_total+=1
                    if label=="high":self.metrics.high_confidence_matches+=1
                    else:self.metrics.low_confidence_recovery_matches+=1
                    if recovered and appearance_used:self.metrics.appearance_assisted_recoveries+=1
                    if recovered:self.metrics.recovered_tracks+=1
                    matched_tracks.add(ti);matched_detections.add(di)

            confirmed=[i for i,t in enumerate(active) if t.state==TrackState.CONFIRMED]
            tentative=[i for i,t in enumerate(active) if t.state==TrackState.TENTATIVE]
            high=[i for i,d in enumerate(detections) if d.confidence>=self.high]
            stage(confirmed,high,self.match,"high")
            stage([i for i in tentative if i not in matched_tracks],[i for i in high if i not in matched_detections],self.relaxed_match,"high")
            remaining_dets=[i for i,d in enumerate(detections) if i not in matched_detections and d.confidence>=self.low]
            stage([i for i in confirmed if i not in matched_tracks],remaining_dets,self.relaxed_match,"low",True)
            lost = [i for i, track in enumerate(active) if track.state == TrackState.LOST and i not in matched_tracks]
            stage(lost, [i for i in remaining_dets if i not in matched_detections], self.relaxed_match * .8, "recovery", True)
            # A confirmed track may expire visually after max_lost_ms yet remain
            # as a short invisible tombstone. Recover it before creating a new ID
            # only when motion overlap is exceptionally strong.
            removed=[track for track in self.tracks if track.state==TrackState.REMOVED and track.removed_at_monotonic is not None and (mono-track.removed_at_monotonic)*1000<=self.tombstone_extended_ms]
            remaining=[index for index,detection in enumerate(detections) if index not in matched_detections and detection.confidence>=self.low]
            if removed and remaining:
                scores=iou_matrix([track.predict(mono) for track in removed],[detections[index].bbox_xyxy for index in remaining])
                for row,track in enumerate(removed):
                    age_ms=(mono-track.removed_at_monotonic)*1000;required=self.tombstone_recovery_iou if age_ms<=self.tombstone_recovery_ms else self.tombstone_extended_iou
                    scores[row,scores[row]<required]=0.0
                for row,col,score in greedy_match(scores,self.tombstone_recovery_iou):
                    track=removed[row];di=remaining[col]
                    if di in matched_detections:continue
                    track.state=TrackState.CONFIRMED;track.removed_at_monotonic=None
                    track.update(detections[di].bbox_xyxy,detections[di].confidence,result.frame_id,now,embeddings[di],self.min_hits,detections[di].confidence>=self.high,mono,detections[di].detection_source,detections[di].detection_id)
                    matched_detections.add(di);self.metrics.recovered_tracks+=1;self.metrics.tombstone_recoveries+=1
            self.metrics.association_ms = (time.perf_counter() - association_started) * 1000

            for i, track in enumerate(active):
                if i not in matched_tracks: track.miss(mono)
            admitted_this_observation=[]
            for index,detection in enumerate(detections):
                x1,y1,x2,y2=detection.bbox_xyxy;width=max(0,x2-x1);height=max(0,y2-y1);ratio=width/max(height,1);plausible=width>=self.new_min_width and height>=self.new_min_height and width*height>=self.new_min_area and self.new_min_ratio<=ratio<=self.new_max_ratio
                if index in matched_detections or index in ambiguous_detections or detection.confidence<self.new_threshold or not plausible:continue
                # Do not spawn a fragment from a second observation of a person
                # whose existing track was already matched in this same frame.
                matched_boxes=[active[i].bbox for i in matched_tracks]
                if any(duplicate_observation(box,detection.bbox_xyxy) for box in matched_boxes):
                    self.metrics.duplicate_new_track_suppressed+=1;continue
                if admitted_this_observation:
                    if any(duplicate_observation(box,detection.bbox_xyxy) for box in admitted_this_observation):
                        self.metrics.duplicate_new_track_suppressed+=1;continue
                suspect_candidates=[]
                for i,t in enumerate(active):
                    if t.state not in (TrackState.CONFIRMED,TrackState.LOST) or i in matched_tracks:continue
                    proximity=float(motion_proximity_matrix([predicted[i]],[detection.bbox_xyxy],self.recovery_max_distance)[0,0])
                    if proximity>=self.relaxed_match:suspect_candidates.append((proximity,t))
                if suspect_candidates:
                    _, predecessor=max(suspect_candidates,key=lambda item:item[0])
                    center_x=(x1+x2)*.5;center_y=(y1+y2)*.5
                    admission_key=(predecessor.track_id,int(center_x//64),int(center_y//64))
                    if admission_key not in self._deferred_admissions:
                        self._deferred_admissions[admission_key]=(1,mono)
                        self.metrics.deferred_new_track_admissions+=1
                        continue
                    self._deferred_admissions.pop(admission_key,None)
                    self.metrics.local_track_fragments+=1;self.metrics.id_switch_suspected+=1
                recent_removed=[t for t in self.tracks if t.state==TrackState.REMOVED and t.removed_at_monotonic is not None and mono-t.removed_at_monotonic<=3.0]
                if recent_removed:
                    def audit_score(old):
                        old_box=old.predict(mono);new_box=detection.bbox_xyxy;iou=float(iou_matrix([old_box],[new_box])[0,0])
                        ocx=(old_box[0]+old_box[2])*.5;ocy=(old_box[1]+old_box[3])*.5;ncx=(new_box[0]+new_box[2])*.5;ncy=(new_box[1]+new_box[3])*.5
                        scale=max(1.0,((old_box[2]-old_box[0])**2+(old_box[3]-old_box[1])**2)**.5);distance=((ocx-ncx)**2+(ocy-ncy)**2)**.5/scale
                        return iou-distance,old,old_box,iou,distance
                    _,old,old_box,iou,distance=max((audit_score(item) for item in recent_removed),key=lambda item:item[0])
                    event={"old_local_track":old.track_id,"new_local_track":f"{self.camera_id}:TRACK-{self._next_id:05d}","time_gap_ms":round((mono-old.last_detection_monotonic)*1000,1),"old_predicted_bbox":list(map(float,old_box)),"new_bbox":list(map(float,detection.bbox_xyxy)),"iou":round(iou,4),"normalized_center_distance":round(distance,4),"velocity":list(old.velocity),"old_state":"REMOVED","reid_similarity":None,"reason":"new_track_after_lost_timeout"}
                    self.metrics.fragment_events.append(event);del self.metrics.fragment_events[:-20]
                audit={"camera":self.camera_id,"new_track_id":f"{self.camera_id}:TRACK-{self._next_id:05d}","bbox":list(map(float,detection.bbox_xyxy)),"previous_compatible_track":None,"iou":0.0,"center_distance":None,"elapsed_ms":None,"reason_old_track_not_reused":"no_recent_physically_compatible_track"}
                candidates=[]
                for old in self.tracks:
                    old_box=old.predict(mono);overlap=float(iou_matrix([old_box],[detection.bbox_xyxy])[0,0]);ocx=(old_box[0]+old_box[2])*.5;ocy=(old_box[1]+old_box[3])*.5;ncx=(x1+x2)*.5;ncy=(y1+y2)*.5;scale=max(1.0,((old_box[2]-old_box[0])**2+(old_box[3]-old_box[1])**2)**.5);distance=((ocx-ncx)**2+(ocy-ncy)**2)**.5/scale;candidates.append((overlap-distance,old,overlap,distance))
                if candidates:
                    _,old,overlap,distance=max(candidates,key=lambda item:item[0]);audit.update(previous_compatible_track=old.track_id,iou=round(overlap,4),center_distance=round(distance,4),elapsed_ms=round((mono-old.last_detection_monotonic)*1000,1),bbox_scale_difference=round(abs(((x2-x1)*max(y2-y1,1))/max((old.bbox[2]-old.bbox[0])*max(old.bbox[3]-old.bbox[1],1),1)-1),4),last_real_detection_confidence=round(float(old.confidence),4),predecessor_state=old.state.value,reason_old_track_not_reused="association_below_recovery_guards" if old.state!=TrackState.REMOVED else "tombstone_recovery_guards_failed")
                self.metrics.new_track_audits.append(audit);del self.metrics.new_track_audits[:-50]
                track = Track(self.camera_id, self._next_id, detection.bbox_xyxy, detection.confidence,
                              result.frame_id, now, now, result.frame_id, appearance_embedding=embeddings[index],detection_source=detection.detection_source,generation=self._generation,source_width=result.source_width,source_height=result.source_height,detection_id=detection.detection_id)
                track.motion_monotonic=mono;track.last_detection_monotonic=mono;track.last_real_observation_monotonic=mono
                track.real_observations.clear();track.real_observations.append((mono,np.asarray(track.motion._measurement(detection.bbox_xyxy),np.float32)))
                self._next_id += 1; self.tracks.append(track);admitted_this_observation.append(detection.bbox_xyxy); self.metrics.new_tracks += 1
            self.metrics.unmatched_detections += sum(index not in matched_detections and detection.confidence < self.new_threshold for index,detection in enumerate(detections))
            for track in self.tracks:
                if track.state == TrackState.REMOVED: continue
                required_misses=1 if track.state==TrackState.TENTATIVE else self.min_expiry_misses
                if track.misses>=required_misses and (mono-track.last_detection_monotonic)*1000 > self.max_lost_ms:
                    track.state = TrackState.REMOVED; track.removed_at_monotonic=mono; self.metrics.removed_tracks += 1; self.metrics.deleted_tracks += 1;self.metrics.visual_expirations_total+=1; self.metrics.removal_reason="bounded_real_miss_timeout"
                    disappearance_auditor.record_disappearance(self.camera_id,track.track_id,track.visual_bbox,track.predict_visual(mono),track.confidence,(mono-track.last_real_observation_monotonic)*1000,max(0.0,(now-track.last_seen_at)*1000),"RETENTION_EXPIRED")
            # Reconcile a stale predicted fragment only when a current real
            # observation occupies the same physical geometry. Never compare
            # two currently detected tracks: adjacent/occluding people survive.
            current = [track for track in self.tracks if track.state != TrackState.REMOVED and track.misses == 0]
            stale = [track for track in self.tracks if track.state in (TrackState.CONFIRMED, TrackState.LOST) and track.misses > 0]
            for old in stale:
                if any(duplicate_observation(fresh.bbox, old.predict(mono)) for fresh in current):
                    old.state = TrackState.REMOVED
                    old.removed_at_monotonic = mono
                    self.metrics.duplicate_new_track_suppressed += 1
                    self.metrics.removed_tracks += 1
                    self.metrics.deleted_tracks += 1
                    self.metrics.visual_expirations_total+=1;self.metrics.removal_reason = "overlapping_stale_fragment_reconciled"
                    disappearance_auditor.record_disappearance(self.camera_id,old.track_id,old.visual_bbox,old.predict_visual(mono),old.confidence,(mono-old.last_real_observation_monotonic)*1000,max(0.0,(now-old.last_seen_at)*1000),"ASSOCIATION_FAIL")
            visible_tracks=[t for t in self.tracks if t.state in (TrackState.TENTATIVE,TrackState.CONFIRMED,TrackState.LOST)]
            for track in visible_tracks:
                was_candidate,was_hidden=track.boundary_exit_candidate,track.visual_hidden
                candidate,hidden,delay_ms,_ratio=track.evaluate_boundary_exit(mono)
                if candidate and not was_candidate:self.metrics.boundary_exit_candidates_total+=1
                if hidden and not was_hidden:
                    self.metrics.boundary_exit_visual_hides_total+=1;self.metrics.boundary_exit_removal_delays_ms.append(delay_ms);del self.metrics.boundary_exit_removal_delays_ms[:-100]
                    disappearance_auditor.record_disappearance(self.camera_id,track.track_id,track.visual_bbox,track.predict_visual(mono),track.confidence,(mono-track.last_real_observation_monotonic)*1000,max(0.0,(now-track.last_seen_at)*1000),"BOUNDARY_EXIT")
                if track.misses>0:self.metrics.temporary_miss_predictions_total+=1
            visible=[t.output(now,mono,visual_horizon_ms=self.prediction_horizon_ms) for t in visible_tracks]
            self.metrics.tracks_active = sum(t.state != TrackState.REMOVED for t in self.tracks)
            self.metrics.tracks_confirmed = sum(t.state == TrackState.CONFIRMED for t in self.tracks)
            self.metrics.average_track_age_seconds = sum(now-t.created_at for t in visible_tracks)/len(visible_tracks) if visible_tracks else 0.0
            lost_tracks = [t for t in visible_tracks if t.state == TrackState.LOST]
            self.metrics.average_lost_duration_seconds = sum(now-t.last_seen_at for t in lost_tracks)/len(lost_tracks) if lost_tracks else 0.0
            self.metrics.tracks_lost = sum(t.state == TrackState.LOST for t in self.tracks)
            visible_after={track.track_id for track in self.tracks if track.state in (TrackState.CONFIRMED,TrackState.LOST)}
            created_ids=visible_after-visible_before;removed_ids=visible_before-visible_after
            self.metrics.visual_track_created+=len(created_ids);self.metrics.visual_track_removed+=len(removed_ids)
            for track_id in sorted(created_ids):self.metrics.visual_lifecycle_events.append({"camera":self.camera_id,"track_id":track_id,"state":"DETECTED","timestamp":now})
            for track_id in sorted(removed_ids):self.metrics.visual_lifecycle_events.append({"camera":self.camera_id,"track_id":track_id,"state":"EXPIRED","timestamp":now,"removal_reason":self.metrics.removal_reason or "lost_timeout"})
            del self.metrics.visual_lifecycle_events[:-100]
            retained=[track for track in self.tracks if track.state in (TrackState.CONFIRMED,TrackState.LOST)]
            ages=[max(0.0,(mono-track.last_detection_monotonic)*1000) for track in retained];self.metrics.last_real_detection_age_ms=max(ages,default=0.0);self.metrics.prediction_age_ms=max([age for track,age in zip(retained,ages) if track.state==TrackState.LOST],default=0.0)
            self.metrics.tracker_update_ms = (time.perf_counter() - started) * 1000
            return CameraTrackResult(result.camera_id,result.frame_id,result.capture_timestamp,result.receive_timestamp,tuple(visible),mono,result.source_width,result.source_height)

    def predict_visual(self,frame_id,capture_timestamp,receive_timestamp,now_monotonic=None):
        mono=time.monotonic() if now_monotonic is None else float(now_monotonic)
        with self._lock:
            tracks=[]
            for track in self.tracks:
                age_ms=(mono-track.last_real_observation_monotonic)*1000
                if track.state not in (TrackState.CONFIRMED,TrackState.LOST) or age_ms>self.prediction_horizon_ms:continue
                was_candidate,was_hidden=track.boundary_exit_candidate,track.visual_hidden
                candidate,hidden,delay_ms,_ratio=track.evaluate_boundary_exit(mono)
                if candidate and not was_candidate:self.metrics.boundary_exit_candidates_total+=1
                if hidden and not was_hidden:self.metrics.boundary_exit_visual_hides_total+=1;self.metrics.boundary_exit_removal_delays_ms.append(delay_ms);del self.metrics.boundary_exit_removal_delays_ms[:-100]
                tracks.append(track.output(capture_timestamp,mono,"predicted",self.prediction_horizon_ms));self.metrics.temporary_miss_predictions_total+=int(track.misses>0)
            self.metrics.visual_prediction_frames+=1;self.metrics.visual_prediction_boxes+=len(tracks);self.metrics.visual_prediction_rate=self.metrics.visual_prediction_frames/max(.001,mono-self._prediction_started)
            ages=[track.prediction_age_ms for track in tracks];self.metrics.prediction_age_ms=max(ages,default=0.0);self.metrics.last_real_detection_age_ms=max(ages,default=0.0)
            width=max((track.source_width for track in self.tracks),default=0);height=max((track.source_height for track in self.tracks),default=0)
            return CameraTrackResult(self.camera_id,frame_id,capture_timestamp,receive_timestamp,tuple(tracks),mono,width,height)

    def reset(self):
        with self._lock:self.tracks.clear();self._next_id=1;self._generation+=1;self._deferred_admissions.clear()
