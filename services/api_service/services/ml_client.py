"""Single API-to-ML control client."""
import asyncio,json
from urllib.error import HTTPError,URLError
from urllib.request import Request,urlopen
from fastapi import HTTPException

class MLClient:
 def __init__(self,base_url,timeout=1.5):self.base_url=base_url.rstrip("/");self.timeout=timeout;self.available=False
 def _request_sync(self,method,path,payload=None):
  body=json.dumps(payload).encode() if payload is not None else None
  request=Request(self.base_url+path,data=body,headers={"Content-Type":"application/json"},method=method)
  with urlopen(request,timeout=self.timeout) as response:
   raw=response.read();return json.loads(raw) if raw else {}
 async def request(self,method,path,payload=None,required=True):
  try:result=await asyncio.wait_for(asyncio.to_thread(self._request_sync,method,path,payload),self.timeout+.25);self.available=True;return result
  except (HTTPError,URLError,TimeoutError,OSError,ValueError,asyncio.TimeoutError) as exc:
   self.available=False
   if required:raise HTTPException(503,"ML service unavailable") from exc
   return None
 async def health(self):return await self.request("GET","/health",required=False)
 async def command(self,message):return await self.request("POST","/commands",message.model_dump(mode="json"))
