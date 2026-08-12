"""Direct ML control and bounded semantic event bridge."""
import threading
from shared.logging import get_logger
from services.ml_service.control import app,runtime
from services.ml_service.events.api_publisher import APIEventPublisher
from shared.settings import ServiceSettings
log=get_logger(__name__)
class MLMessageBridge:
 def __init__(self,_unused=None):
  settings=ServiceSettings.from_env();self.settings=settings;self.publisher=APIEventPublisher(settings.api_url);self.server=None;self.thread=None
 def start(self):
  import uvicorn
  self.publisher.start();self.server=uvicorn.Server(uvicorn.Config(app,host=self.settings.ml_host,port=self.settings.ml_port,log_level="warning"));self.thread=threading.Thread(target=self.server.run,name="ml-control-api",daemon=False);self.thread.start();return True
 def poll(self):return runtime.poll()
 def publish(self,_channel,payload):return self.publisher.publish(payload)
 def stop_server(self,timeout=6):
  if self.server:self.server.should_exit=True
  if self.thread:
   self.thread.join(timeout)
   if self.thread.is_alive() and self.server:
    self.server.force_exit=True;self.thread.join(2)
   if self.thread.is_alive():log.error("ML control API thread did not stop");return False
  return True
 def close(self):
  server_stopped=self.stop_server();self.publisher.close();return server_stopped
