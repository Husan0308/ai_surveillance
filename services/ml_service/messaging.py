"""Direct ML control and bounded semantic event bridge."""
import threading
from services.ml_service.control import app,runtime
from services.ml_service.events.api_publisher import APIEventPublisher
from shared.settings import ServiceSettings
class MLMessageBridge:
 def __init__(self,_unused=None):
  settings=ServiceSettings.from_env();self.settings=settings;self.publisher=APIEventPublisher(settings.api_url);self.server=None;self.thread=None
 def start(self):
  import uvicorn
  self.publisher.start();self.server=uvicorn.Server(uvicorn.Config(app,host=self.settings.ml_host,port=self.settings.ml_port,log_level="warning"));self.thread=threading.Thread(target=self.server.run,name="ml-control-api",daemon=True);self.thread.start();return True
 def poll(self):return runtime.poll()
 def publish(self,_channel,payload):return self.publisher.publish(payload)
 def close(self):
  self.publisher.close()
  if self.server:self.server.should_exit=True
  if self.thread:self.thread.join(3)
