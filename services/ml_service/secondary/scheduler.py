"""Bounded, stale-aware workers kept completely off the detector callback."""
from __future__ import annotations
from collections import defaultdict
import queue, threading, time
from .task import SecondaryTask, SecondaryTaskType
from shared.logging import get_logger
log=get_logger(__name__)

class SecondaryAIScheduler:
    def __init__(self, processors=None, result_callback=None, queue_size=36, max_task_age_ms=1000,
                 reid_batch_size=6, reid_batch_wait_ms=5):
        self.processors = processors or {}; self.result_callback = result_callback
        self.max_age = float(max_task_age_ms) / 1000; self.batch_size = max(1, int(reid_batch_size)); self.batch_wait = max(0, float(reid_batch_wait_ms)) / 1000
        self.queues = {kind: queue.Queue(max(1, int(queue_size))) for kind in SecondaryTaskType}
        self._stop = threading.Event(); self._threads = []
        self._lock = threading.Lock(); self._metrics = defaultdict(lambda: {"submitted":0,"completed":0,"dropped":0,"stale":0,"queue_depth":0,"batch_size":0,"processing_ms":0.0,"cache_hits":0})

    def start(self):
        if self._threads: return
        self._stop.clear()
        for kind in self.processors:
            thread = threading.Thread(target=self._run, args=(kind,), name=f"secondary-{kind.value.lower()}", daemon=False)
            self._threads.append(thread); thread.start()

    def submit(self, task: SecondaryTask):
        q = self.queues[task.task_type]
        try: q.put_nowait(task)
        except queue.Full:
            with self._lock: self._metrics[task.task_type]["dropped"] += 1
            return False
        with self._lock:
            metric=self._metrics[task.task_type];metric["submitted"]+=1;metric["queue_depth"]=q.qsize()
        return True

    def cache_hit(self, kind=SecondaryTaskType.REID):
        with self._lock:self._metrics[kind]["cache_hits"]+=1

    def _run(self, kind):
        q=self.queues[kind];processor=self.processors[kind]
        while not self._stop.is_set():
            try:first=q.get(timeout=.2)
            except queue.Empty:continue
            tasks=[first]
            if kind == SecondaryTaskType.REID:
                deadline=time.monotonic()+self.batch_wait
                while len(tasks)<self.batch_size:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:break
                    try:tasks.append(q.get(timeout=remaining))
                    except queue.Empty:break
            fresh=[];now=time.time()
            for task in tasks:
                if not (isinstance(task.context,dict) and task.context.get("kind")=="enrollment") and now-task.capture_timestamp>self.max_age:
                    with self._lock:self._metrics[kind]["stale"]+=1
                    q.task_done()
                else:fresh.append(task)
            if fresh:
                started=time.perf_counter()
                try:
                    outputs=processor(fresh) if kind == SecondaryTaskType.REID else [processor(task) for task in fresh]
                    if outputs is None:outputs=[None]*len(fresh)
                    for task,result in zip(fresh,outputs):
                        if result is not None and self.result_callback:self.result_callback(task,result)
                except Exception as exc:
                    log.exception("%s secondary worker failed: %s",kind.value,exc)
                finally:
                    elapsed=(time.perf_counter()-started)*1000
                    with self._lock:
                        metric=self._metrics[kind];metric["completed"]+=len(fresh);metric["batch_size"]=len(fresh);metric["processing_ms"]=elapsed;metric["queue_depth"]=q.qsize()
                    for _ in fresh:q.task_done()

    def snapshot(self):
        with self._lock:return {kind.value.lower():dict(self._metrics[kind],queue_depth=self.queues[kind].qsize()) for kind in SecondaryTaskType}

    def stop(self):self._stop.set()
    def join(self, timeout=5):
        deadline=time.monotonic()+timeout
        for thread in self._threads:thread.join(max(0,deadline-time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)
    def shutdown(self, timeout=5):self.stop();return self.join(timeout)
    def alive_threads(self):return tuple(thread.name for thread in self._threads if thread.is_alive())
