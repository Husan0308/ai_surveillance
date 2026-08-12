"""Bounded event-driven global identity reconciliation.

The detector callback only allocates/looks up a binding and coalesces evidence.
Gallery work, candidate ranking, hysteresis and canonical merges run here.
"""
from __future__ import annotations
from collections import OrderedDict,deque
import threading,time
import numpy as np
from shared.logging import get_logger

log=get_logger(__name__)


class IdentityAssociationWorker:
    def __init__(self,manager,result_callback=None,queue_size=64,max_task_age_ms=2000):
        self.manager=manager;self.result_callback=result_callback;self.queue_size=max(1,int(queue_size));self.max_age=max(0.05,float(max_task_age_ms)/1000)
        self._condition=threading.Condition();self._pending=OrderedDict();self._versions={};self._stop=False;self._thread=None
        self._processing=deque(maxlen=500);self._fast=deque(maxlen=1000);self._metrics={"submitted":0,"processed":0,"coalesced":0,"stale":0,"dropped":0,"queue_max":0,"errors":0}

    @staticmethod
    def evidence_version(observation):
        if observation.appearance_embedding is None:return ("track",observation.camera_id,observation.local_track_id)
        if observation.embedding_frame_id is not None or observation.embedding_timestamp is not None:
            return ("reid",observation.embedding_frame_id,round(float(observation.embedding_timestamp or 0),3))
        value=np.asarray(observation.appearance_embedding,np.float32).reshape(-1)
        return ("legacy",tuple(np.round(value[:32],5)))

    def observe(self,observation):
        started=time.perf_counter();result,created=self.manager.lookup_or_create(observation);version=self.evidence_version(observation);key=(observation.camera_id,observation.local_track_id)
        with self._condition:
            previous=self._versions.get(key)
            if created or version!=previous:
                self._versions[key]=version
                if key in self._pending:self._pending[key]=(observation,version,time.monotonic());self._pending.move_to_end(key);self._metrics["coalesced"]+=1
                else:
                    if len(self._pending)>=self.queue_size:
                        dropped_key,_=self._pending.popitem(last=False);self._versions.pop(dropped_key,None);self._metrics["dropped"]+=1
                    self._pending[key]=(observation,version,time.monotonic());self._metrics["submitted"]+=1;self._metrics["queue_max"]=max(self._metrics["queue_max"],len(self._pending));self._condition.notify()
        self._fast.append((time.perf_counter()-started)*1000);return result

    def start(self):
        with self._condition:
            if self._thread and self._thread.is_alive():return
            self._stop=False;self._thread=threading.Thread(target=self._run,name="global-identity-worker",daemon=False);self._thread.start()

    def _run(self):
        while True:
            with self._condition:
                while not self._pending and not self._stop:self._condition.wait(.5)
                if self._stop and not self._pending:return
                pending=list(self._pending.items());self._pending.clear()
            fresh=[]
            for key,(observation,version,enqueued) in pending:
                if time.monotonic()-enqueued>self.max_age:
                    with self._condition:self._metrics["stale"]+=1;self._versions.pop(key,None)
                else:fresh.append(observation)
            if not fresh:continue
            started=time.perf_counter()
            try:
                results=self.manager.update_batch(fresh) if hasattr(self.manager,"update_batch") else tuple(self.manager.update(item) for item in fresh);remaps=self.manager.consume_remaps()
                if self.result_callback:
                    for index,result in enumerate(results):self.result_callback(result,remaps if index==len(results)-1 else ())
                with self._condition:self._metrics["processed"]+=len(results)
            except Exception:
                with self._condition:self._metrics["errors"]+=len(fresh)
                log.exception("Global identity worker failed for %d observations",len(fresh))
            finally:self._processing.append((time.perf_counter()-started)*1000)

    @staticmethod
    def _profile(values):
        if not values:return {"count":0,"p50":0.0,"p95":0.0,"max":0.0}
        data=np.asarray(tuple(values),float);return {"count":len(data),"p50":float(np.percentile(data,50)),"p95":float(np.percentile(data,95)),"max":float(np.max(data))}

    def snapshot(self):
        with self._condition:return dict(self._metrics,queue_depth=len(self._pending),processing=self._profile(self._processing),fast_lookup=self._profile(self._fast))

    def stop(self):
        with self._condition:self._stop=True;self._condition.notify_all()
    def join(self,timeout=5):
        thread=self._thread
        if thread:thread.join(timeout)
        return not thread or not thread.is_alive()
    def shutdown(self,timeout=5):self.stop();return self.join(timeout)
