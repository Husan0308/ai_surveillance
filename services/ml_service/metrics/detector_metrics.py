"""Correctly named detector timing and throughput metrics."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from collections import deque
import threading
import time
from .timing import TimingProfile,_percentile

def _distribution(values):
    values=list(values)
    if not values:return {"count":0,"p50":0.0,"p95":0.0,"max":0.0,"over_500":0,"over_800":0,"over_1000":0}
    return {"count":len(values),"p50":_percentile(values,50),"p95":_percentile(values,95),"max":max(values),
            "over_500":sum(value>500 for value in values),"over_800":sum(value>800 for value in values),"over_1000":sum(value>1000 for value in values)}

@dataclass(frozen=True)
class DetectorBatchMetrics:
    batch_id: int
    batch_size: int
    camera_ids: tuple[str, ...]
    scheduler_batch_size: int
    tensor_batch_size: int
    model_output_batch_size: int
    started_at: float
    completed_at: float
    preprocess_ms: float
    cpu_pack_ms: float
    h2d_ms: float
    gpu_inference_ms: float
    postprocess_ms: float
    result_parse_ms: float
    detector_wall_ms: float
    total_detection_latency_ms: float
    phases: dict[str,float]=field(default_factory=dict)
    camera_observation_ages_ms: dict[str,float]=field(default_factory=dict)

class DetectorMetrics:
    def __init__(self):
        self._lock = threading.Lock(); self._started = time.monotonic()
        self._batches = self.processed_frames_total = 0
        self.stale_drops_before_inference = self.duplicate_inference_prevented = 0
        self.last: DetectorBatchMetrics | None = None
        self.categories={"FAST":0,"NORMAL":0,"SLOW":0,"EXTREME":0};self.extreme_batches=deque(maxlen=100)
        self.profile=TimingProfile()
        self._observation_ages=deque(maxlen=4096);self._observation_gaps=deque(maxlen=4096)
        self._last_observation={};self._camera_observation_ages={};self._camera_observation_gaps={}
        self._camera_observation_counts={}

    def record(self, item, audit=None):
        with self._lock:
            self.last = item; self._batches += 1; self.processed_frames_total += item.batch_size
            category="FAST" if item.gpu_inference_ms<350 else "NORMAL" if item.gpu_inference_ms<700 else "SLOW" if item.gpu_inference_ms<1000 else "EXTREME";self.categories[category]+=1
            if category=="EXTREME":self.extreme_batches.append(dict(audit or {},category=category))
            for camera_id in item.camera_ids:
                age=float(item.camera_observation_ages_ms.get(camera_id,item.total_detection_latency_ms));self._observation_ages.append(age)
                self._camera_observation_ages.setdefault(camera_id,deque(maxlen=1024)).append(age)
                self._camera_observation_counts[camera_id]=self._camera_observation_counts.get(camera_id,0)+1
                previous=self._last_observation.get(camera_id)
                if previous is not None:
                    gap=max(0.0,(item.completed_at-previous)*1000);self._observation_gaps.append(gap);self._camera_observation_gaps.setdefault(camera_id,deque(maxlen=1024)).append(gap)
                self._last_observation[camera_id]=item.completed_at
            self.profile.record(item.phases)

    def snapshot(self):
        with self._lock:
            elapsed = max(1e-6, time.monotonic() - self._started)
            result = asdict(self.last) if self.last else {}
            result.update(batch_rate=self._batches / elapsed,
                          processed_frames_total=self.processed_frames_total,
                          stale_drops_before_inference=self.stale_drops_before_inference,
                          duplicate_inference_prevented=self.duplicate_inference_prevented,
                          cuda_categories=dict(self.categories),
                          extreme_batches=tuple(self.extreme_batches))
            result["observation_age_ms"]=_distribution(self._observation_ages)
            result["observation_gap_ms"]=_distribution(self._observation_gaps)
            result["camera_observations"]={camera_id:{"observations":self._camera_observation_counts.get(camera_id,0),"age_ms":_distribution(self._camera_observation_ages.get(camera_id,())),"gap_ms":_distribution(self._camera_observation_gaps.get(camera_id,()))}
                                           for camera_id in sorted(self._camera_observation_counts)}
            return result
    def profile_snapshot(self):return self.profile.snapshot()

    def format_compact(self):
        item = self.snapshot()
        if not self.last: return "Detector waiting for first batch"
        return (f"DETECTOR_BATCH id:{item['batch_id']} cameras:{list(item['camera_ids'])} "
                f"scheduler:{item['scheduler_batch_size']} tensor:{item['tensor_batch_size']} output:{item['model_output_batch_size']}\n"
                f"preprocess:{item['preprocess_ms']:.1f}ms cpu_pack:{item['cpu_pack_ms']:.1f}ms h2d_device:{item['h2d_ms']:.1f}ms "
                f"cuda_forward:{item['gpu_inference_ms']:.1f}ms postprocess:{item['postprocess_ms']:.1f}ms\n"
                f"detector_wall:{item['detector_wall_ms']:.1f}ms rate:{item['batch_rate']:.1f}/s")
