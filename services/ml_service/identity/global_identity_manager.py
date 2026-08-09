"""Topology-aware cross-camera identity association with stable unknown IDs."""
import threading,time
import numpy as np
from .candidate_filter import CameraTopology,CandidateFilter
from .identity_store import IdentityStore
from .matcher import IdentityMatcher
from .metrics import IdentityMetrics
from .schemas import GlobalTrack,GlobalTrackResult,IdentityStatus

class GlobalIdentityManager:
    def __init__(self,config=None,store=None):
        self.config=config or {}; cfg=self.config.get("identity",{})
        self.store=store or IdentityStore();self.topology=CameraTopology(self.config)
        self.filter=CandidateFilter(self.topology,float(cfg.get("max_candidate_age_ms",300000)))
        self.matcher=IdentityMatcher(self.config);self.metrics=IdentityMetrics();self._lock=threading.RLock();self._pending={};self._ambiguity={}
        self.match_threshold=float(cfg.get("match_threshold",.76));self.strong_threshold=float(cfg.get("strong_match_threshold",.86))
        self.new_threshold=float(cfg.get("new_identity_threshold",.58));self.ambiguity_margin=float(cfg.get("ambiguity_margin",.04))
        self.max_history=int(cfg.get("max_embeddings_per_identity",20));self.max_activity_history=int(cfg.get("max_activity_history",256));self.max_identities=int(cfg.get("max_runtime_identities",2000));self.min_quality=float(cfg.get("min_embedding_quality",.55))
        self.recently_lost=float(cfg.get("recently_lost_timeout",5));self.inactive=float(cfg.get("inactive_timeout",60));self.archive=float(cfg.get("archive_timeout",600))

    def update(self,observation):
        started=time.perf_counter(); key=(observation.camera_id,observation.local_track_id)
        with self._lock:
            bound=self.store.binding(*key)
            if bound:
                identity=self.store.get(bound)
                contradiction=any(camera!=observation.camera_id and (observation.timestamp-seen_at)*1000<2000 and self.topology.relationship(camera,observation.camera_id) not in ("same_camera","same_room","overlapping") for camera,seen_at in identity.active_track_seen.items())
                if contradiction:
                    self.metrics.values.identity_conflicts+=1;self.metrics.values.rejected_impossible_merges+=1
                    return self._ambiguous(observation,key,identity.global_id,0.0,"existing_binding_topology_conflict")
                self._touch(identity,observation,identity.confidence);self._refresh_metrics(observation.timestamp)
                return self._result(observation,identity,identity.confidence,"existing_binding")
            pending=self._ambiguity.get(key)
            if pending and observation.timestamp-pending["at"]<1.0:
                return self._result(observation,None,pending["confidence"],pending["reason"],IdentityStatus.AMBIGUOUS)
            if observation.appearance_embedding is None:
                return self._ambiguous(observation,key,None,0.0,"waiting_for_reid")
            identities=self.store.identities();self.metrics.values.candidate_count_before_filter=len(identities)
            filter_started=time.perf_counter();candidates=self.filter.filter(observation,identities)
            self.metrics.values.rejected_impossible_merges += self.filter.last_rejections
            self.metrics.values.candidate_filter_ms=(time.perf_counter()-filter_started)*1000
            self.metrics.values.candidate_count_after_filter=len(candidates)
            similarity_started=time.perf_counter();scores=self.matcher.score(observation,candidates)
            self.metrics.values.similarity_ms=(time.perf_counter()-similarity_started)*1000
            top=scores[0] if scores else None;second=scores[1][1] if len(scores)>1 else -1
            margin=top[1]-second if top else 1;identity=None;reason=""
            if top and top[1]>=self.strong_threshold and margin>=self.ambiguity_margin:
                identity=top[0];reason=f"strong_match reid={top[2]:.3f} relation={top[3]}";self.metrics.values.global_matches+=1
                if identity.status!=IdentityStatus.ACTIVE:self.metrics.values.recovered_identities+=1
            elif top and top[1]>=self.match_threshold:
                if margin<self.ambiguity_margin:
                    return self._ambiguous(observation,key,top[0].global_id,top[1],f"ambiguous margin={margin:.3f}")
                previous,count=self._pending.get(key,(None,0));count=count+1 if previous==top[0].global_id else 1
                self._pending[key]=(top[0].global_id,count)
                if count>=2:identity=top[0];reason="confirmed_pending_match";self.metrics.values.global_matches+=1
                else:
                    return self._ambiguous(observation,key,top[0].global_id,top[1],"pending_more_evidence")
            elif top and top[1]>=self.new_threshold:
                return self._ambiguous(observation,key,top[0].global_id,top[1],"below_match_pending")
            else:
                identity=self.store.create(observation);reason="new_identity_no_sufficient_match";self.metrics.values.new_identities+=1
            self.store.bind(*key,identity.global_id);self._pending.pop(key,None);self._ambiguity.pop(key,None);self._touch(identity,observation,top[1] if top and identity is top[0] else observation.quality_score)
            self.metrics.values.identity_match_ms=(time.perf_counter()-started)*1000;self._refresh_metrics(observation.timestamp)
            return self._result(observation,identity,identity.confidence,reason)

    def _ambiguous(self,observation,key,candidate,confidence,reason):
        signature=(candidate,reason);previous=self._ambiguity.get(key)
        if previous is None or previous["signature"]!=signature:self.metrics.values.ambiguous_matches+=1
        self._ambiguity[key]={"signature":signature,"confidence":float(confidence),"reason":reason,"at":observation.timestamp}
        return self._result(observation,None,confidence,reason,IdentityStatus.AMBIGUOUS)

    def _touch(self,identity,observation,confidence):
        cutoff=observation.timestamp-2.0
        for camera,seen_at in list(identity.active_track_seen.items()):
            if seen_at<cutoff:identity.active_track_seen.pop(camera,None);identity.active_tracks.pop(camera,None)
        relation=self.topology.relationship(identity.last_camera_id,observation.camera_id)
        if identity.active_tracks and relation=="impossible_transition" and observation.camera_id not in identity.active_tracks:
            self.metrics.values.identity_conflicts+=1
        identity.last_seen_at=observation.timestamp;identity.last_camera_id=observation.camera_id;identity.last_local_track_id=observation.local_track_id
        identity.active_track_seen[observation.camera_id]=observation.timestamp
        identity.status=IdentityStatus.ACTIVE;identity.confidence=float(confidence);identity.active_tracks[observation.camera_id]=observation.local_track_id
        identity.camera_history.append((observation.camera_id,observation.timestamp));identity.track_history.append((observation.camera_id,observation.local_track_id,observation.timestamp));del identity.camera_history[:-self.max_activity_history];del identity.track_history[:-self.max_activity_history]
        if observation.quality_score>=self.min_quality:identity.add_embedding(observation.appearance_embedding,observation.quality_score,self.max_history)

    def _refresh_metrics(self,now):
        active=0
        for identity in self.store.identities():
            age=now-identity.last_seen_at
            if age>=self.archive:identity.status=IdentityStatus.ARCHIVED
            elif age>=self.inactive:identity.status=IdentityStatus.INACTIVE
            elif age>=self.recently_lost:identity.status=IdentityStatus.RECENTLY_LOST
            if identity.status==IdentityStatus.ACTIVE:active+=1
        self.store.prune_archived(self.max_identities);self.metrics.values.global_identities_active=active;self.metrics.values.global_identities_total=len(self.store.identities())

    @staticmethod
    def _result(obs,identity,confidence,reason,status=None):
        return GlobalTrackResult(obs.camera_id,obs.frame_id,(GlobalTrack(obs.local_track_id,identity.global_id if identity else None,obs.bbox,obs.confidence,float(confidence),status or identity.status,reason,getattr(identity,"person_id",None) if identity else None,getattr(identity,"display_name",None) if identity else None),))

    def camera_failed(self,camera_id):
        self.store.remove_camera_bindings(camera_id)
        for identity in self.store.identities():identity.active_tracks.pop(camera_id,None);identity.active_track_seen.pop(camera_id,None)

    def close(self):self._pending.clear();self._ambiguity.clear()
