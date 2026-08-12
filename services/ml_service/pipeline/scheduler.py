"""Thread-safe, event-driven fresh-frame scheduler with no inference code."""
import logging
import threading
import time

log=logging.getLogger(__name__)
from dataclasses import replace
from .batch import BatchOutput

class BatchScheduler:
    def __init__(self, max_frame_age_ms=250, on_batch=None, metrics_interval_sec=2,
                 starved_after_ms=500, batch_collect_window_ms=5, max_batch_size=None,
                 mode="fixed",min_batch_size=2,fairness_deadline_ms=900):
        self.max_frame_age_ms = max(1.0, float(max_frame_age_ms))
        self.starved_after_ms = max(1.0, float(starved_after_ms))
        self.metrics_interval = max(.25, float(metrics_interval_sec))
        self.collect_window = max(0.0, float(batch_collect_window_ms)) / 1000
        self.max_batch_size = None if max_batch_size is None else max(1, int(max_batch_size))
        self.mode=str(mode).strip().lower();self.min_batch_size=max(1,int(min_batch_size))
        self.fairness_deadline_ms=max(100.0,float(fairness_deadline_ms));self._risks={}
        self._adaptive_batch_size=self.max_batch_size or self.min_batch_size
        self._on_batch, self._condition = on_batch, threading.Condition(threading.RLock())
        self._buffers, self._states, self._last_dropped = {}, {}, {}
        self._camera_cursor = 0
        self._thread, self._running, self._last_batch = None, False, None
        self._batch_id = self._stale_drops = 0
        self._window_at = time.monotonic()
        self._window_batches = self._window_processed = self._window_stale = 0
        self._burst_cycles = {}

    def notify_frame_available(self):
        with self._condition: self._condition.notify()

    def register_camera(self, camera_id, buffer):
        with self._condition:
            self._buffers[camera_id] = buffer
            self._states.setdefault(camera_id, {"used_frame_id": None, "frame_age_ms": 0.0,
                "duplicate_count": 0, "stale_drops": 0, "starved_count": 0,
                "starved": False, "last_seen": 0.0,"last_selected":0.0,"selected_age_ms":None,"latest_available_age_ms":None,
                "selection_reason":"no_new_frame","priority":0.0})
            self._last_dropped.setdefault(camera_id, 0); self._condition.notify()

    def unregister_camera(self, camera_id):
        with self._condition:
            self._buffers.pop(camera_id, None); self._states.pop(camera_id, None)
            self._last_dropped.pop(camera_id, None); self._risks.pop(camera_id,None);self._burst_cycles.pop(camera_id,None);self._condition.notify()

    def update_camera_risks(self,risks):
        """Atomically publish tracker risk; scheduler remains the only selector."""
        with self._condition:
            self._risks={str(camera_id):dict(item) for camera_id,item in (risks or {}).items()}
            self._condition.notify()

    def start(self):
        with self._condition:
            if self._thread and self._thread.is_alive(): return
            self._running = True
            self._thread = threading.Thread(target=self.run, name="batch-scheduler", daemon=False)
            self._thread.start()

    def run(self):
        while True:
            with self._condition:
                if not self._running: break
                if not any(len(buffer) for buffer in self._buffers.values()):
                    self._condition.wait(timeout=min(self.metrics_interval, self.starved_after_ms / 1000)); continue
                deadline = time.monotonic() + self.collect_window
                while self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: break
                    self._condition.wait(timeout=remaining)
                if not self._running: break
                batch = self._collect_locked()
            if batch is not None and self._on_batch is not None:
                try:self._on_batch(batch)
                except Exception:log.exception("batch callback failed batch_id=%s cameras=%s",batch.batch_id,batch.camera_ids)

    def _selection_order_locked(self,now_mono):
        camera_ids=tuple(self._buffers)
        if not camera_ids:return (),0
        start=self._camera_cursor%len(camera_ids);round_robin=camera_ids[start:]+camera_ids[:start]
        limit=self.max_batch_size or len(camera_ids)
        if self.mode not in {"risk","risk_aware","adaptive"}:return round_robin,limit
        ready=[camera_id for camera_id in round_robin if len(self._buffers[camera_id])]
        priorities={};urgent=False
        _burst_threshold_ms=600.0;_max_burst_cycles=2
        for rank,camera_id in enumerate(round_robin):
            state=self._states[camera_id];risk=self._risks.get(camera_id,{})
            since_ms=self.fairness_deadline_ms if not state["last_selected"] else max(0.0,(now_mono-state["last_selected"])*1000)
            observation_age=max(since_ms,float(risk.get("observation_age_ms",0.0)))
            lost=max(0.0,float(risk.get("lost_tracks",0.0)));active=max(0.0,float(risk.get("active_person_count",0.0)));uncertainty=max(0.0,float(risk.get("motion_uncertainty",0.0)))
            ambiguity=max(0.0,float(risk.get("association_ambiguity",0.0)))
            priority=2.0*since_ms/self.fairness_deadline_ms+observation_age/self.fairness_deadline_ms+.8*active+.7*lost+.6*ambiguity+.15*min(4.0,uncertainty)
            is_lost=lost>0;old_enough=float(risk.get("observation_age_ms",0.0))>=_burst_threshold_ms
            if is_lost and old_enough:
                if self._burst_cycles.get(camera_id,0)<_max_burst_cycles:priority+=1000.0
            else:self._burst_cycles[camera_id]=0
            state["priority"]=priority;priorities[camera_id]=(priority,-rank)
            urgent=urgent or observation_age>=self.fairness_deadline_ms*.75 or lost>0 or ambiguity>0
        # Recovery is priority, never extra capacity. Risk-aware mode keeps the
        # configured hard batch size; adaptive mode may use the smaller baseline
        # only while the scene is calm. Urgency expands adaptive mode back up to
        # the configured limit rather than shrinking a busy batch.
        target=min(limit,len(ready))
        if self.mode=="adaptive" and not urgent:target=min(target,self.min_batch_size)
        selected=sorted(ready,key=lambda camera_id:priorities[camera_id],reverse=True)[:target]
        selected_set=set(selected)
        for camera_id in ready:
            if camera_id in selected_set:
                if priorities[camera_id][0]>=1000.0:
                    self._burst_cycles[camera_id]=self._burst_cycles.get(camera_id,0)+1
                    self._states[camera_id]["selection_reason"]="recovery_burst"
                else:self._burst_cycles[camera_id]=0
            else:self._states[camera_id]["selection_reason"]="deferred_by_risk_priority"
        self._adaptive_batch_size=len(selected)
        return tuple(selected),len(selected)

    def _collect_locked(self):
        build_started = time.monotonic()
        now_wall, now_mono, packets = time.time(), build_started, []
        camera_ids=tuple(self._buffers)
        if not camera_ids:return None
        start=self._camera_cursor % len(camera_ids);ordered,target=self._selection_order_locked(now_mono)
        examined=0
        for camera_id in ordered:
            if len(packets)>=target:break
            examined+=1;buffer=self._buffers[camera_id]
            packet = buffer.take()
            state = self._states[camera_id]
            if packet is None:state["selection_reason"]="no_new_frame";state["selected_age_ms"]=None;continue
            age = max(0.0, (now_wall - packet.receive_timestamp) * 1000);state["latest_available_age_ms"]=age
            state["last_seen"], state["frame_age_ms"], state["starved"] = now_mono, age, False
            if state["used_frame_id"] is not None and packet.frame_id <= state["used_frame_id"]:
                state["duplicate_count"] += 1; continue
            if age > self.max_frame_age_ms:
                state["stale_drops"] += 1; self._stale_drops += 1; self._window_stale += 1;state["selection_reason"]="stale_omitted";state["selected_age_ms"]=None; continue
            state["used_frame_id"] = packet.frame_id;state["last_selected"]=now_mono;state["selection_reason"]="selected_fresh";state["selected_age_ms"]=age; packets.append(replace(packet, scheduler_selected_timestamp=now_wall))
        if examined:self._camera_cursor=(start+examined)%len(camera_ids)
        if not packets: return None
        self._batch_id += 1; self._window_batches += 1; self._window_processed += len(packets)
        self._last_batch = BatchOutput(self._batch_id, now_wall, tuple(packets),build_started,time.monotonic())
        return self._last_batch

    def take_if_newer(self, camera_id, current_frame_id):
        """Re-fetch immediately before GPU launch if a newer frame arrived during queue wait."""
        with self._condition:
            buffer = self._buffers.get(camera_id)
            if not buffer:return None
            packet = buffer.peek()
            if packet is None or packet.frame_id <= current_frame_id:return None
            packet = buffer.take()
            if packet is not None:
                state=self._states.get(camera_id)
                if state is not None:state["used_frame_id"] = packet.frame_id
            return packet

    def snapshot_metrics(self, reader_metrics=None):
        now, reader_metrics = time.monotonic(), reader_metrics or {}
        elapsed = max(1e-6, now - self._window_at)
        with self._condition:
            cameras, dropped = {}, self._window_stale
            for camera_id, state in self._states.items():
                buffer=self._buffers.get(camera_id);latest=buffer.peek() if buffer is not None else None
                source = dict(reader_metrics.get(camera_id, {}));source["buffer_depth"]=len(buffer) if buffer is not None else 0;source["latest_frame_id"]=latest.frame_id if latest is not None else source.get("recv_frame_id")
                last_decode=float(source.get("last_decode_timestamp") or source.get("last_frame_timestamp") or 0.0)
                decode_stale=bool(last_decode and time.time()-last_decode>self.starved_after_ms/1000)
                starved = not source.get("online", False) or decode_stale
                if starved and not state["starved"]: state["starved_count"] += 1; state["starved"] = True
                for key in ("used_frame_id", "frame_age_ms", "duplicate_count", "starved_count","selected_age_ms","latest_available_age_ms","selection_reason","priority"): source[key] = state[key]
                source["stale_drops"], source["is_starved"] = state["stale_drops"], starved
                for key, default in (("recv_frame_id", 0), ("source_fps", 0.0), ("interarrival_ms", 0.0), ("max_interarrival_ms", 0.0), ("dropped_old", 0)): source.setdefault(key, default)
                current = source["dropped_old"]; dropped += max(0, current - self._last_dropped.get(camera_id, 0))
                self._last_dropped[camera_id] = current; cameras[camera_id] = source
            output = {"cameras": cameras,"scheduler_mode":self.mode,"scheduler_target_batch_size":self._adaptive_batch_size,
                "scheduler_batches_created":self._batch_id, "batch_size": len(self._last_batch.frames) if self._last_batch else 0,
                "batch_cameras": list(self._last_batch.camera_ids) if self._last_batch else [],
                "batch_rate": self._window_batches / elapsed, "stale_drops": self._stale_drops,
                "source_total_fps": sum(c["source_fps"] for c in cameras.values()),
                "processed_total_fps": self._window_processed / elapsed, "dropped_total_fps": dropped / elapsed}
            self._window_at = now; self._window_batches = self._window_processed = self._window_stale = 0
            return output

    def stop(self):
        with self._condition: self._running = False; self._condition.notify_all()
    def join(self, timeout=None):
        if self._thread: self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()
