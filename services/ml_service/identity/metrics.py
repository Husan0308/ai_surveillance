from dataclasses import asdict,dataclass
import threading

@dataclass
class IdentityMetricValues:
    reid_extract_ms:float=0; reid_batch_size:int=0; candidate_filter_ms:float=0
    candidate_count_before_filter:int=0; candidate_count_after_filter:int=0
    similarity_ms:float=0; identity_match_ms:float=0
    global_identities_active:int=0; global_identities_total:int=0
    global_matches:int=0; new_identities:int=0; recovered_identities:int=0
    ambiguous_matches:int=0; identity_conflicts:int=0; identity_switch_suspected:int=0; rejected_impossible_merges:int=0; global_id_remaps:int=0

class IdentityMetrics:
    def __init__(self):self._lock=threading.Lock();self.values=IdentityMetricValues()
    def snapshot(self):
        with self._lock:return asdict(self.values)
    def format_compact(self):
        m=self.snapshot();return (f"IDENTITY\nactive_global:{m['global_identities_active']} total:{m['global_identities_total']} "
            f"new:{m['new_identities']} recovered:{m['recovered_identities']} ambiguous:{m['ambiguous_matches']} conflicts:{m['identity_conflicts']} rejected:{m['rejected_impossible_merges']}\n"
            f"ReID batch:{m['reid_batch_size']} extract:{m['reid_extract_ms']:.2f}ms matching:{m['identity_match_ms']:.2f}ms")
