"""Bounded in-process latency distributions for production diagnostics."""
from collections import defaultdict,deque
import math,threading

def _percentile(values,percent):
    if not values:return 0.0
    ordered=sorted(values);position=(len(ordered)-1)*percent/100;lower=math.floor(position);upper=math.ceil(position)
    if lower==upper:return ordered[lower]
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)

class TimingProfile:
    def __init__(self,max_samples=512):self._values=defaultdict(lambda:deque(maxlen=max_samples));self._lock=threading.Lock()
    def record(self,values):
        with self._lock:
            for key,value in values.items():
                if isinstance(value,(int,float)) and math.isfinite(float(value)):self._values[key].append(float(value))
    def snapshot(self):
        with self._lock:items={key:list(values) for key,values in self._values.items()}
        return {key:{"count":len(values),"avg":sum(values)/len(values),"p50":_percentile(values,50),"p90":_percentile(values,90),"p95":_percentile(values,95),"p99":_percentile(values,99),"max":max(values)} for key,values in items.items() if values}
