"""Conservative topology-aware global identity association."""
import logging,threading,time
from collections import OrderedDict,defaultdict,deque
import numpy as np
from .candidate_filter import CameraTopology,CandidateFilter
from .identity_store import IdentityStore
from .matcher import IdentityMatcher
from .metrics import IdentityMetrics
from .schemas import GlobalTrack,GlobalTrackResult,IdentityStatus
log=logging.getLogger(__name__)

class GlobalIdentityManager:
    def __init__(self,config=None,store=None):
        self.config=config or {};cfg=self.config.get("identity",{});reid_cfg=self.config.get("ai",{}).get("reid",{})
        self.store=store or IdentityStore();self.topology=CameraTopology(self.config);self.filter=CandidateFilter(self.topology,float(cfg.get("max_candidate_age_ms",300000)));self.matcher=IdentityMatcher(self.config);self.metrics=IdentityMetrics()
        self._lock=threading.RLock();self._pending={};self._ambiguity={};self._remaps=[];self._evaluated_embeddings={};self._started_monotonic=time.monotonic();self._last_decisions=OrderedDict();self._candidate_counts={};self._candidate_rankings={}
        self._same_room_reuse_by_room=defaultdict(int);self._samples=defaultdict(lambda:deque(maxlen=500));self._camera_quality=defaultdict(lambda:{"valid":0,"rejected":0,"widths":deque(maxlen=500),"heights":deque(maxlen=500),"self_similarities":deque(maxlen=500)})
        self.match_threshold=float(cfg.get("match_threshold",reid_cfg.get("ambiguous_threshold",.82)));self.strong_threshold=float(cfg.get("strong_match_threshold",reid_cfg.get("threshold",.85)));self.ambiguity_margin=float(cfg.get("ambiguity_margin",reid_cfg.get("unknown_margin",.04)))
        self.required_merge_evidence=max(1,int(cfg.get("required_merge_evidence",2)));self.max_history=int(cfg.get("max_embeddings_per_identity",20));self.max_activity_history=int(cfg.get("max_activity_history",256));self.max_identities=int(cfg.get("max_runtime_identities",2000));self.min_quality=float(cfg.get("min_embedding_quality",.55))
        self.min_crop_width=float(reid_cfg.get("min_crop_width",20));self.min_crop_height=float(reid_cfg.get("min_crop_height",45));self.active_window_ms=float(cfg.get("active_window_ms",2000));self.recently_lost=float(cfg.get("recently_lost_timeout",5));self.inactive=float(cfg.get("inactive_timeout",60));self.archive=float(cfg.get("archive_timeout",600))
        self.same_camera_threshold=float(cfg.get("same_camera_recovery_threshold",.90));self.same_camera_margin=float(cfg.get("same_camera_recovery_margin",.01));self.same_camera_gap_ms=float(cfg.get("same_camera_recovery_gap_ms",15000));self.same_camera_center_distance=float(cfg.get("same_camera_center_distance",.20));self.gallery_min_internal_similarity=float(cfg.get("gallery_min_internal_similarity",.65));self.max_evidence_age_ms=float(cfg.get("max_reid_evidence_age_ms",2000));self.min_independent_evidence_ms=float(cfg.get("min_independent_evidence_ms",0))

    def update(self,observation,_preferred_global_id=None,_joint=False):
        started=time.perf_counter();key=(observation.camera_id,observation.local_track_id)
        with self._lock:
            bound_id=self.store.binding(*key);identity=self.store.get(bound_id) if bound_id else None
            if identity is None:
                identity=self.store.create(observation);self.store.bind(*key,identity.global_id);self.metrics.values.new_identities+=1;self.metrics.values.provisional_created+=1;self.metrics.values.canonical_created+=1;self._record_new_reason(observation)
                usable,details=self._embedding_quality(observation)
                self._touch(identity,observation,observation.quality_score,update_gallery=usable)
                if usable:self.metrics.values.gallery_updates+=1;identity.audit_gallery(self.gallery_min_internal_similarity);self._evaluated_embeddings[key]=self._evidence_id(observation);self._record_quality(observation,details,True)
                else:self.metrics.values.gallery_updates_rejected+=1
                self._log_decision(observation,None,identity,0.0,-1.0,1.0,"PROVISIONAL",details["reason"],details)
                self._refresh_metrics(observation.timestamp);return self._result(observation,identity,identity.confidence,"provisional_global_id")
            payload_id=self._evidence_id(observation) if observation.appearance_embedding is not None else None;previous_evidence=self._evaluated_embeddings.get(key)
            if payload_id is not None and previous_evidence in (payload_id,("rejected",payload_id)):
                self._touch(identity,observation,identity.confidence,update_gallery=False);self._refresh_metrics(observation.timestamp);return self._result(observation,identity,identity.confidence,"cached_reid_evidence")
            usable,details=self._embedding_quality(observation);fingerprint=payload_id;new_evidence=usable and previous_evidence!=fingerprint
            if usable and identity.appearance_history:
                self_similarity=self.matcher.robust_similarity(self._normalized(observation.appearance_embedding),identity)
                if self_similarity is not None:self._camera_quality[observation.camera_id]["self_similarities"].append(self_similarity);self._record_sample("same_local_track",self_similarity,-1.0,2.0,observation,identity)
            if not usable and observation.appearance_embedding is not None and self._evaluated_embeddings.get(key)!=("rejected",fingerprint):
                self._evaluated_embeddings[key]=("rejected",fingerprint);self._record_quality(observation,details,False);self.metrics.values.global_merge_attempts+=1;self.metrics.values.global_merge_rejected_low_quality+=1;self.metrics.values.gallery_updates_rejected+=1;self._log_decision(observation,identity,None,0.0,-1.0,1.0,"REJECT",details["reason"],details)
            if new_evidence:
                self._evaluated_embeddings[key]=fingerprint;self._record_quality(observation,details,True);remapped=self._reconcile(identity,observation,key,details,_preferred_global_id,_joint)
                if remapped is not None:
                    self.metrics.values.identity_match_ms=(time.perf_counter()-started)*1000;self._refresh_metrics(observation.timestamp);return self._result(observation,remapped,remapped.confidence,"canonical_global_merge")
            # Evidence from this already-bound local trajectory may improve only its
            # own provisional gallery. It never contaminates a candidate identity.
            update_gallery=bool(new_evidence and usable);self._touch(identity,observation,max(identity.confidence,observation.quality_score),update_gallery=update_gallery)
            if update_gallery:self.metrics.values.gallery_updates+=1;identity.audit_gallery(self.gallery_min_internal_similarity)
            self._refresh_metrics(observation.timestamp);return self._result(observation,identity,identity.confidence,"existing_binding")

    def _reconcile(self,current,observation,key,quality,preferred_global_id=None,joint=False):
        candidates=tuple(item for item in self.store.identities() if item.global_id!=current.global_id);self.metrics.values.global_merge_attempts+=1;self.metrics.values.candidate_count_before_filter=len(candidates)
        filter_started=time.perf_counter();candidates=self.filter.filter(observation,candidates);self.metrics.values.rejected_impossible_merges+=self.filter.last_rejections;self.metrics.values.global_merge_rejected_same_camera+=self.filter.last_same_camera_conflicts;self.metrics.values.candidate_filter_ms=(time.perf_counter()-filter_started)*1000;self.metrics.values.candidate_count_after_filter=len(candidates)
        similarity_started=time.perf_counter();scores=self.matcher.score(observation,candidates);self.metrics.values.similarity_ms=(time.perf_counter()-similarity_started)*1000;self._candidate_counts[key]=len(scores);self._candidate_rankings[key]=tuple((self.store.canonicalize(item[0].global_id),float(item[2])) for item in scores)
        if not scores:
            self.metrics.values.reid_rejects+=1;self.metrics.values.gallery_updates_rejected+=1;self._log_decision(observation,current,None,0.0,-1.0,1.0,"REJECT","no_eligible_candidate",quality);return None
        if preferred_global_id is not None:
            preferred=next((item for item in scores if self.store.canonicalize(item[0].global_id)==self.store.canonicalize(preferred_global_id)),None)
            if preferred is None:
                self.metrics.values.reid_rejects+=1;self.metrics.values.gallery_updates_rejected+=1;self._log_decision(observation,current,None,0.0,-1.0,1.0,"REJECT","joint_candidate_no_longer_eligible",quality);return None
            top=preferred
        else:top=scores[0]
        second_sim=max((item[2] for item in scores if item[0].global_id!=top[0].global_id),default=-1.0);margin=top[2]-second_sim
        if joint:margin=max(margin,self.ambiguity_margin)
        candidate=top[0];relation=top[3];gap_ms=max(0.0,(observation.timestamp-candidate.last_seen_at)*1000)
        recovery=self._same_camera_recovery(candidate,observation,gap_ms);category="same_camera_candidate" if relation=="same_camera" else "cross_camera_candidate";self._record_sample(category,top[2],second_sim,margin,observation,candidate)
        conflict=self._active_conflict(candidate,observation)
        if conflict:
            if conflict=="same_camera_active_conflict":self.metrics.values.global_merge_rejected_same_camera+=1
            else:self.metrics.values.global_merge_rejected_active_conflict+=1
            self.metrics.values.gallery_updates_rejected+=1;self.metrics.values.gallery_contamination_guard+=1;self._log_decision(observation,current,candidate,top[2],second_sim,margin,"REJECT",conflict,quality);return None
        if top[2]<self.match_threshold:
            self.metrics.values.reid_rejects+=1;self.metrics.values.similarity_rejects+=1;self.metrics.values.gallery_updates_rejected+=1;self._pending.pop(key,None);self._log_decision(observation,current,candidate,top[2],second_sim,margin,"REJECT","raw_similarity_below_match_threshold",quality);return None
        required_similarity=self.same_camera_threshold if recovery else self.strong_threshold;required_margin=self.same_camera_margin if recovery else self.ambiguity_margin
        if top[2]<required_similarity or margin<required_margin:
            self.metrics.values.ambiguous_matches+=1;self.metrics.values.global_merge_rejected_ambiguous+=1;self.metrics.values.margin_rejects+=int(top[2]>=required_similarity and margin<required_margin);self.metrics.values.similarity_rejects+=int(top[2]<required_similarity);self.metrics.values.gallery_updates_rejected+=1;self.metrics.values.gallery_contamination_guard+=1;self._pending.pop(key,None);self._record_sample("ambiguous",top[2],second_sim,margin,observation,candidate);reason="same_camera_recovery_separation_required" if recovery else "strong_similarity_and_margin_required";self._log_decision(observation,current,candidate,top[2],second_sim,margin,"AMBIGUOUS",reason,quality);return None
        previous,count,previous_evidence_at=self._pending.get(key,(None,0,None));independent=previous==candidate.global_id and (previous_evidence_at is None or (float(observation.embedding_timestamp or observation.timestamp)-previous_evidence_at)*1000>=self.min_independent_evidence_ms);count=count+1 if independent else 1;self._pending[key]=(candidate.global_id,count,float(observation.embedding_timestamp or observation.timestamp))
        if count<self.required_merge_evidence:
            self.metrics.values.global_merge_rejected_ambiguous+=1;self.metrics.values.gallery_updates_rejected+=1;self._log_decision(observation,current,candidate,top[2],second_sim,margin,"PENDING",f"independent_evidence_{count}_of_{self.required_merge_evidence}",quality);return None
        old_ids=(current.global_id,candidate.global_id);canonical=self.store.merge(*old_ids,max_history=self.max_history);duplicate=old_ids[1] if canonical.global_id==old_ids[0] else old_ids[0]
        self._pending.pop(key,None);self._remaps.append((duplicate,canonical.global_id));self.metrics.values.merged_identities+=1;self.metrics.values.global_id_remaps+=1;self.metrics.values.global_merge_accepted+=1;self.metrics.values.global_matches+=1;self.metrics.values.reid_matches+=1;self.metrics.values.provisional_merged+=1;self.metrics.values.alias_count+=1
        if relation=="same_camera":self.metrics.values.global_reused_same_camera+=1
        else:
            self.metrics.values.global_reused_cross_camera+=1
            if relation=="same_room":
                self.metrics.values.same_room_reuse+=1;self._same_room_reuse_by_room[self.topology.room(observation.camera_id)]+=1
        canonical.audit_gallery(self.gallery_min_internal_similarity);self._record_sample("accepted",top[2],second_sim,margin,observation,candidate);self._touch(canonical,observation,top[2],update_gallery=False);self._log_decision(observation,current,candidate,top[2],second_sim,margin,"ACCEPT","same_camera_fragment_recovery" if recovery else "strong_independent_evidence_all_guards_passed",quality);return canonical


    def update_batch(self,observations):
        """Jointly reserve at most one canonical candidate per room/camera window."""
        observations=tuple(observations)
        with self._lock:
            edges={};keys=[]
            for observation in observations:
                key=(observation.camera_id,observation.local_track_id);bound=self.store.binding(*key);current=self.store.get(bound) if bound else None
                usable,_details=self._embedding_quality(observation)
                if current is None or not usable:continue
                candidates=self.filter.filter(observation,tuple(item for item in self.store.identities() if item.global_id!=current.global_id))
                viable=[item for item in self.matcher.score(observation,candidates) if item[3] in ("same_room","overlapping") and item[2]>=self.strong_threshold]
                if viable:keys.append(key);edges[key]=viable[:6]
            best=[]
            def search(index,used,total,choice):
                if index==len(keys):best.append((total,len(choice),dict(choice)));return
                key=keys[index];search(index+1,used,total,choice)
                for item in edges[key]:
                    candidate_id=self.store.canonicalize(item[0].global_id)
                    if candidate_id in used:continue
                    choice[key]=candidate_id;search(index+1,used|{candidate_id},total+item[2],choice);choice.pop(key,None)
            if keys:search(0,set(),0.0,{})
            best.sort(key=lambda item:(item[1],item[0]),reverse=True);assignment=best[0][2] if best else {}
            # A joint solution is trusted only when it is discriminated from the
            # best alternative with the same number of matched observations.
            best_score=best[0][0] if best else 0.0;alternative=next((item[0] for item in best[1:] if item[1]==best[0][1]),-1.0) if best else -1.0
            joint_ok=not best or alternative<0 or best_score-alternative>=self.ambiguity_margin
        results=[]
        for observation in observations:
            key=(observation.camera_id,observation.local_track_id);preferred=assignment.get(key) if joint_ok else None
            results.append(self.update(observation,preferred,preferred is not None))
        return tuple(results)

    def lookup_or_create(self,observation):
        """O(1) detector-side binding lookup with immediate stable provisional ID."""
        key=(observation.camera_id,observation.local_track_id)
        with self._lock:
            global_id=self.store.binding(*key);identity=self.store.get(global_id) if global_id else None;created=False
            if identity is None:
                identity=self.store.create(observation);self.store.bind(*key,identity.global_id);created=True
                self.metrics.values.new_identities+=1;self.metrics.values.provisional_created+=1;self.metrics.values.canonical_created+=1;self._record_new_reason(observation)
            self._touch_fast(identity,observation)
            return self._result(observation,identity,identity.confidence,"provisional_global_id" if created else "cached_global_binding"),created

    def _record_new_reason(self,observation):
        x1,y1,x2,y2=map(float,observation.bbox);width=x2-x1;height=y2-y1
        if observation.appearance_embedding is None:reason="NO_REID_EVIDENCE"
        elif observation.quality_score<self.min_quality or width<self.min_crop_width or height<self.min_crop_height:reason="LOW_QUALITY"
        else:reason="NEW_PHYSICAL_CANDIDATE"
        values=self.metrics.values.global_new_reasons;values[reason]=values.get(reason,0)+1

    def _touch_fast(self,identity,observation):
        stamp=max(float(identity.last_seen_at),float(observation.timestamp));identity.last_seen_at=stamp;identity.last_camera_id=observation.camera_id;identity.last_local_track_id=observation.local_track_id
        identity.last_bbox=tuple(observation.bbox);identity.last_source_size=(int(observation.source_width),int(observation.source_height));identity.active_track_seen[observation.camera_id]=stamp;identity.active_tracks[observation.camera_id]=observation.local_track_id;identity.status=IdentityStatus.ACTIVE

    def _active_conflict(self,candidate,observation):
        active=[]
        for camera,track in candidate.active_tracks.items():
            seen=float(candidate.active_track_seen.get(camera,0));
            if (observation.timestamp-seen)*1000<=self.active_window_ms:active.append((camera,track))
        for camera,track in active:
            if camera==observation.camera_id and str(track)!=str(observation.local_track_id):return "same_camera_active_conflict"
        other=[camera for camera,_track in active if camera!=observation.camera_id]
        if not other:return None
        if not self.topology.verified:return "unverified_topology_simultaneous_active_conflict"
        for camera in other:
            if self.topology.relationship(camera,observation.camera_id) not in ("same_room","overlapping"):return "verified_different_room_simultaneous_active_conflict"
        return None

    def _embedding_quality(self,observation):
        x1,y1,x2,y2=map(float,observation.bbox);width=max(0.0,x2-x1);height=max(0.0,y2-y1);aspect=width/max(height,1.0);embedding=observation.appearance_embedding
        if embedding is None:return False,{"width":width,"height":height,"quality":float(observation.quality_score),"norm":0.0,"reason":"waiting_for_reid"}
        value=np.asarray(embedding,np.float32).reshape(-1);norm=float(np.linalg.norm(value));quality=float(observation.quality_score)
        reason="quality_ok"
        if not value.size or not np.isfinite(value).all() or norm<.90 or norm>1.10:reason="invalid_embedding_norm"
        elif width<self.min_crop_width or height<self.min_crop_height:reason="crop_too_small"
        elif aspect<.08 or aspect>1.5:reason="invalid_person_crop_aspect"
        elif quality<self.min_quality:reason="crop_quality_below_minimum"
        if observation.embedding_timestamp is not None and (observation.timestamp-observation.embedding_timestamp)*1000>self.max_evidence_age_ms:reason="stale_reid_evidence"
        usable=reason=="quality_ok"
        return usable,{"width":width,"height":height,"quality":quality,"norm":norm,"reason":reason}

    @staticmethod
    def _fingerprint(embedding):
        if embedding is None:return None
        value=np.asarray(embedding,np.float32).reshape(-1);return tuple(np.round(value[:32],5))

    def _evidence_id(self,observation):
        # Extraction provenance is primary. A changed rolling embedding from the
        # same extraction must not count twice. Fingerprints are legacy fallback.
        if observation.embedding_frame_id is not None or observation.embedding_timestamp is not None:
            return ("extraction",observation.camera_id,observation.embedding_frame_id,round(float(observation.embedding_timestamp or 0),3))
        return ("legacy",self._fingerprint(observation.appearance_embedding))

    def _record_quality(self,observation,details,usable):
        camera=self._camera_quality[observation.camera_id];camera["valid" if usable else "rejected"]+=1;camera["widths"].append(float(details["width"]));camera["heights"].append(float(details["height"]))

    @staticmethod
    def _normalized(embedding):
        value=np.asarray(embedding,np.float32).reshape(-1);return value/max(float(np.linalg.norm(value)),1e-12)

    def _same_camera_recovery(self,candidate,observation,gap_ms):
        if candidate.last_camera_id!=observation.camera_id or gap_ms>self.same_camera_gap_ms or candidate.last_bbox is None:return False
        if any(camera==observation.camera_id and str(track)!=str(observation.local_track_id) and (observation.timestamp-candidate.active_track_seen.get(camera,0))*1000<=self.active_window_ms for camera,track in candidate.active_tracks.items()):return False
        width=max(int(observation.source_width),int(candidate.last_source_size[0]),1);height=max(int(observation.source_height),int(candidate.last_source_size[1]),1)
        def center(box):return ((box[0]+box[2])*.5,(box[1]+box[3])*.5)
        old=center(candidate.last_bbox);new=center(observation.bbox);distance=((old[0]-new[0])**2+(old[1]-new[1])**2)**.5/(width*width+height*height)**.5
        return distance<=self.same_camera_center_distance

    def _record_sample(self,category,top1,top2,margin,observation,candidate):
        self._samples[category].append({"top1":float(top1),"top2":float(top2),"margin":float(margin),"quality":float(observation.quality_score),"width":float(observation.bbox[2]-observation.bbox[0]),"height":float(observation.bbox[3]-observation.bbox[1]),"camera_pair":f"{getattr(candidate,'last_camera_id','none')}->{observation.camera_id}","time_gap_ms":max(0.0,(observation.timestamp-getattr(candidate,'last_seen_at',observation.timestamp))*1000)})

    def record_reid_submitted(self):
        with self._lock:self.metrics.values.reid_submitted+=1
    def record_reid_completed(self,orphaned=False):
        with self._lock:self.metrics.values.reid_completed+=1;self.metrics.values.reid_orphaned+=int(orphaned)
    def record_reid_stale(self,total):
        with self._lock:self.metrics.values.reid_stale=max(self.metrics.values.reid_stale,int(total))

    def _touch(self,identity,observation,confidence,update_gallery=False):
        cutoff=observation.timestamp-self.active_window_ms/1000
        for camera,seen_at in list(identity.active_track_seen.items()):
            if seen_at<cutoff:identity.active_track_seen.pop(camera,None);identity.active_tracks.pop(camera,None)
        newer=float(observation.timestamp)>=float(identity.last_seen_at);identity.last_seen_at=max(float(identity.last_seen_at),float(observation.timestamp));identity.last_camera_id=observation.camera_id if newer else identity.last_camera_id;identity.last_local_track_id=observation.local_track_id if newer else identity.last_local_track_id
        if newer:identity.last_bbox=tuple(observation.bbox);identity.last_source_size=(int(observation.source_width),int(observation.source_height))
        identity.active_track_seen[observation.camera_id]=max(float(identity.active_track_seen.get(observation.camera_id,0)),float(observation.timestamp));identity.active_tracks[observation.camera_id]=observation.local_track_id;identity.status=IdentityStatus.ACTIVE;identity.confidence=float(confidence)
        identity.camera_history.append((observation.camera_id,observation.timestamp));identity.track_history.append((observation.camera_id,observation.local_track_id,observation.timestamp));del identity.camera_history[:-self.max_activity_history];del identity.track_history[:-self.max_activity_history]
        if update_gallery:identity.add_embedding(observation.appearance_embedding,observation.quality_score,self.max_history)

    def _log_decision(self,observation,current,candidate,similarity,second,margin,decision,reason,quality=None):
        counts=self.metrics.values.identity_decision_reasons;counts[reason]=counts.get(reason,0)+1
        active_cameras=sorted(candidate.active_tracks) if candidate else [];active_tracks=dict(candidate.active_tracks) if candidate else {};last_seen=candidate.last_seen_at if candidate else None;quality=quality or {}
        time_gap_ms=max(0.0,(observation.timestamp-last_seen)*1000) if last_seen is not None else -1.0;candidate_state=str(getattr(getattr(candidate,"status",None),"value","none")).lower();gallery_trusted=getattr(candidate,"gallery_trusted",None);track_key=(observation.camera_id,observation.local_track_id);key=(str(observation.camera_id),str(observation.local_track_id));pending=self._pending.get(track_key,(None,0,None));relation=self.topology.relationship(candidate.last_camera_id,observation.camera_id) if candidate else None;top1_id=self.store.canonicalize(getattr(candidate,"global_id",None));ranking=self._candidate_rankings.get(track_key,());top2_item=next((item for item in ranking if item[0]!=top1_id),None);evidence_count=self.required_merge_evidence if decision=="ACCEPT" else int(pending[1]);self._last_decisions[key]={"candidate_count":int(self._candidate_counts.get(track_key,0)),"top1":top1_id,"top1_similarity":round(float(similarity),4),"top2":top2_item[0] if top2_item else None,"top2_similarity":round(float(top2_item[1]),4) if top2_item else None,"margin":round(float(margin),4),"room_relation":relation,"decision":decision,"reason":reason,"independent_evidence_count":evidence_count,"decision_at":time.time()};self._last_decisions.move_to_end(key);
        while len(self._last_decisions)>512:self._last_decisions.popitem(last=False)
        if decision=="ACCEPT":log.info("IDENTITY_MERGE camera=%s local=%s provisional=%s canonical=%s similarity=%.3f reason=%s",observation.camera_id,observation.local_track_id,getattr(current,"global_id",None),getattr(candidate,"global_id",None),similarity,reason)
        log.debug("GLOBAL_MATCH camera=%s local=%s current=%s candidate=%s sim=%.4f second=%.4f margin=%.4f quality=%.3f crop=%.1fx%.1f time_gap_ms=%.1f candidate_state=%s gallery_trusted=%s active_cameras=%s active_tracks=%s decision=%s reason=%s",observation.camera_id,observation.local_track_id,getattr(current,"global_id",None),getattr(candidate,"global_id",None),similarity,second,margin,float(quality.get("quality",0)),float(quality.get("width",0)),float(quality.get("height",0)),time_gap_ms,candidate_state,gallery_trusted,active_cameras,active_tracks,decision,reason)

    def _refresh_metrics(self,now):
        active_ids=set();camera_tracks={};camera_ids={};bindings={}
        for identity in self.store.identities():
            age=now-identity.last_seen_at
            if age>=self.archive:identity.status=IdentityStatus.ARCHIVED
            elif age>=self.inactive:identity.status=IdentityStatus.INACTIVE
            elif age>=self.recently_lost:identity.status=IdentityStatus.RECENTLY_LOST
            for camera,seen_at in list(identity.active_track_seen.items()):
                if (now-seen_at)*1000>self.active_window_ms:identity.active_track_seen.pop(camera,None);identity.active_tracks.pop(camera,None)
            for camera,track in identity.active_tracks.items():
                active_ids.add(identity.global_id);camera_tracks[camera]=camera_tracks.get(camera,0)+1;camera_ids.setdefault(camera,set()).add(identity.global_id);bindings[f"{camera}/{track}"]=identity.global_id
        identities=self.store.identities();self.store.prune_archived(self.max_identities);self.metrics.values.global_identities_active=len(active_ids);self.metrics.values.active_canonical_people=len(active_ids);self.metrics.values.global_identities_total=len(identities);self.metrics.values.camera_local_active=camera_tracks;self.metrics.values.camera_global_ids={key:len(value) for key,value in camera_ids.items()};self.metrics.values.active_local_tracks=sum(camera_tracks.values());self.metrics.values.active_local_to_global=bindings
        rooms={}
        for camera,count in camera_tracks.items():
            room=self.topology.room(camera);entry=rooms.setdefault(room,{"local":0,"canonical":set(),"same_room_reuse":0,"ambiguous":0});entry["local"]+=count;entry["canonical"].update(camera_ids.get(camera,set()))
        self.metrics.values.room_identity={room:{"local":item["local"],"canonical":len(item["canonical"]),"same_room_reuse":self._same_room_reuse_by_room.get(room,0),"ambiguous":self.metrics.values.ambiguous_matches} for room,item in sorted(rooms.items())}
        self.metrics.values.gallery_untrusted=sum(not item.gallery_trusted for item in identities);self.metrics.values.gallery_audit={item.global_id:{"embedding_count":len(item.appearance_history),"trusted":item.gallery_trusted,"min_similarity":round(item.gallery_min_similarity,4),"mean_similarity":round(item.gallery_mean_similarity,4),"camera_count":len({entry[0] for entry in item.camera_history}),"time_span_s":round(max(0,item.last_seen_at-item.created_at),2)} for item in identities}
        self.metrics.values.decision_distributions=self._distribution_snapshot();self.metrics.values.camera_reid_quality=self._camera_quality_snapshot();elapsed_minutes=max((time.monotonic()-self._started_monotonic)/60,1/60);self.metrics.values.identity_churn_rate=self.metrics.values.provisional_created/elapsed_minutes

    @staticmethod
    def _percentiles(values):
        if not values:return {}
        data=np.asarray(values,float);return {f"p{p}":round(float(np.percentile(data,p)),4) for p in (10,25,50,75,90,95)}

    def _distribution_snapshot(self):
        return {category:{"count":len(samples),"top1":self._percentiles([item["top1"] for item in samples]),"top2":self._percentiles([item["top2"] for item in samples]),"margin":self._percentiles([item["margin"] for item in samples]),"quality":self._percentiles([item["quality"] for item in samples])} for category,samples in self._samples.items()}

    def _camera_quality_snapshot(self):
        output={}
        for camera,item in self._camera_quality.items():
            output[camera]={"valid_embeddings":item["valid"],"rejected_low_quality":item["rejected"],"width":self._percentiles(item["widths"]),"height":self._percentiles(item["heights"]),"same_track_self_similarity":self._percentiles(item["self_similarities"])}
        return output

    def _result(self,obs,identity,confidence,reason,status=None):
        global_id=self.store.canonicalize(identity.global_id) if identity else None
        canonical=self.store.get(global_id) if global_id else None
        return GlobalTrackResult(obs.camera_id,obs.frame_id,(GlobalTrack(obs.local_track_id,global_id,obs.bbox,obs.confidence,float(confidence),status or canonical.status,reason,getattr(canonical,"person_id",None),getattr(canonical,"display_name",None) or global_id,identity_version=self.store.version,detection_source=obs.detection_source,detection_id=obs.detection_id),),identity_version=self.store.version)

    def canonicalize(self,global_id):return self.store.canonicalize(global_id)
    @property
    def identity_version(self):return self.store.version

    def decision_snapshot(self):
        with self._lock:return {key:dict(value) for key,value in self._last_decisions.items()}

    def consume_remaps(self):
        with self._lock:items=tuple(self._remaps);self._remaps.clear();return items
    def camera_failed(self,camera_id):
        self.store.remove_camera_bindings(camera_id)
        for identity in self.store.identities():identity.active_tracks.pop(camera_id,None);identity.active_track_seen.pop(camera_id,None)
    def close(self):self._pending.clear();self._ambiguity.clear();self._remaps.clear();self._evaluated_embeddings.clear();self._last_decisions.clear();self._candidate_counts.clear();self._candidate_rankings.clear()
