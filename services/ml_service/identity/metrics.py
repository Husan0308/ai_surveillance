from dataclasses import asdict,dataclass,field
import threading

@dataclass
class IdentityMetricValues:
    reid_extract_ms:float=0; reid_batch_size:int=0; candidate_filter_ms:float=0
    candidate_count_before_filter:int=0; candidate_count_after_filter:int=0
    similarity_ms:float=0; identity_match_ms:float=0
    global_identities_active:int=0; active_canonical_people:int=0; global_identities_total:int=0
    global_matches:int=0; new_identities:int=0; recovered_identities:int=0
    ambiguous_matches:int=0; identity_conflicts:int=0; identity_switch_suspected:int=0; rejected_impossible_merges:int=0; global_id_remaps:int=0
    merged_identities:int=0; reid_matches:int=0; reid_rejects:int=0
    global_merge_attempts:int=0; global_merge_accepted:int=0
    global_merge_rejected_same_camera:int=0; global_merge_rejected_active_conflict:int=0
    global_merge_rejected_ambiguous:int=0; global_merge_rejected_low_quality:int=0
    global_reused_same_camera:int=0; global_reused_cross_camera:int=0; same_room_reuse:int=0
    provisional_created:int=0; provisional_merged:int=0; canonical_created:int=0; alias_count:int=0; identity_churn_rate:float=0.0
    global_new_reasons:dict=field(default_factory=dict); identity_decision_reasons:dict=field(default_factory=dict)
    margin_rejects:int=0; similarity_rejects:int=0; gallery_untrusted:int=0
    reid_submitted:int=0; reid_completed:int=0; reid_stale:int=0; reid_orphaned:int=0
    gallery_updates:int=0; gallery_updates_rejected:int=0; gallery_contamination_guard:int=0
    active_local_tracks:int=0; camera_local_active:dict=None; camera_global_ids:dict=None; active_local_to_global:dict=None
    decision_distributions:dict=None; camera_reid_quality:dict=None; gallery_audit:dict=None; room_identity:dict=None

class IdentityMetrics:
    def __init__(self):self._lock=threading.Lock();self.values=IdentityMetricValues()
    def snapshot(self):
        with self._lock:return asdict(self.values)
    def format_compact(self):
        m=self.snapshot()
        return ("IDENTITY\nactive_canonical:{active_canonical_people} active_local:{active_local_tracks} total:{global_identities_total} "
                "new:{new_identities} reuse_same:{global_reused_same_camera} reuse_cross:{global_reused_cross_camera} merge:{global_merge_accepted}/{global_merge_attempts} same_cam_reject:{global_merge_rejected_same_camera} active_reject:{global_merge_rejected_active_conflict} ambiguous:{global_merge_rejected_ambiguous} low_quality:{global_merge_rejected_low_quality}\n"
                "gallery updates:{gallery_updates} rejected:{gallery_updates_rejected} guard:{gallery_contamination_guard} ReID batch:{reid_batch_size} extract:{reid_extract_ms:.2f}ms matching:{identity_match_ms:.2f}ms").format(**m)
