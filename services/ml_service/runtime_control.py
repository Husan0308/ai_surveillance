"""Independent bounded command consumer; it never depends on camera batches."""
import threading

class RuntimeCommandLoop:
    def __init__(self, poll, handler, interval_seconds=.1):
        self.poll,self.handler=poll,handler;self.interval=max(.01,float(interval_seconds));self._stop=threading.Event();self._thread=None
    def start(self):
        if self._thread and self._thread.is_alive():return
        self._stop.clear();self._thread=threading.Thread(target=self._run,name="ml-command-loop",daemon=False);self._thread.start()
    def _run(self):
        while not self._stop.is_set():
            for command in self.poll():self.handler(command)
            self._stop.wait(self.interval)
    def stop(self):self._stop.set()
    def join(self,timeout=None):
        if self._thread:self._thread.join(timeout)
        return not self._thread or not self._thread.is_alive()
