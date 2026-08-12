from __future__ import annotations
import json
from shared.enrollment_paths import stage_files
from urllib.error import HTTPError,URLError
from urllib.parse import urlencode
from urllib.request import Request,urlopen

class ApiError(Exception):pass
class ApiConnectionError(ApiError):pass
class ApiValidationError(ApiError):pass
class ApiNotFoundError(ApiError):pass
class ApiServerError(ApiError):pass

class ApiClient:
    def __init__(self,base_url="http://127.0.0.1:8000/api/v1",timeout=5):self.base_url=base_url.rstrip("/");self.timeout=timeout
    def _request(self,method,path,payload=None):
        data=json.dumps(payload).encode() if payload is not None else None
        request=Request(f"{self.base_url}/{path.lstrip('/')}",data=data,headers={"Content-Type":"application/json"},method=method)
        try:
            with urlopen(request,timeout=self.timeout) as response:
                raw=response.read();return json.loads(raw) if raw else None
        except HTTPError as exc:
            try:detail=json.loads(exc.read()).get("detail",str(exc))
            except Exception:detail=str(exc)
            if exc.code==404:raise ApiNotFoundError(detail) from exc
            if exc.code==422:raise ApiValidationError(detail) from exc
            raise ApiServerError(detail) from exc
        except (URLError,TimeoutError,OSError) as exc:raise ApiConnectionError(f"API unavailable: {exc}") from exc
    def get(self,path):return self._request("GET",path)
    def post(self,path,payload=None):return self._request("POST",path,payload or {})
    def patch(self,path,payload):return self._request("PATCH",path,payload)
    def delete(self,path):return self._request("DELETE",path)
    def get_persons(self):
        result=self.get("persons")
        if not isinstance(result,list) or any(not isinstance(item,dict) for item in result):raise ApiValidationError("Invalid persons response")
        return result
    def get_person(self,pid):
        result=self.get(f"persons/{pid}")
        if not isinstance(result,dict):raise ApiValidationError("Invalid person response")
        return result
    def create_person(self,data):return self.post("persons",data)
    def update_person(self,pid,data):return self.patch(f"persons/{pid}",data)
    def delete_person(self,pid):return self.delete(f"persons/{pid}")
    def start_enrollment(self,name,sample_paths,department=None):
        staged=stage_files(list(sample_paths));return self.post("enrollment/sessions",{"name":name,"sample_paths":staged,"department":department})
    def get_enrollment(self,sid):return self.get(f"enrollment/sessions/{sid}")
    def cancel_enrollment(self,sid):return self.post(f"enrollment/sessions/{sid}/cancel")
    def get_events(self,**filters):
        query=urlencode({k:v for k,v in filters.items() if v is not None});return self.get("events"+(f"?{query}" if query else ""))
    def acknowledge_event(self,eid):return self.post(f"events/{eid}/acknowledge")
    def get_cameras(self):
        result=self.get("cameras")
        if not isinstance(result,list) or any(not isinstance(item,dict) or not item.get("id") for item in result):raise ApiValidationError("Invalid cameras response")
        return result
    def create_camera(self,data):return self.post("cameras",data)
    def update_camera(self,cid,data):return self.patch(f"cameras/{cid}",data)
    def delete_camera(self,cid):return self.delete(f"cameras/{cid}")
    def get_settings(self):return self.get("settings")
    def update_settings(self,data):return self.patch("settings",data)
    def get_heatmap(self,camera_id,mode="live"):return self.get(f"heatmaps/{camera_id}/{mode}")
    def get_health(self):return self.get("health")
    def get_system_status(self):return self.get("system/status")
    def get_system_metrics(self):return self.get("system/metrics/summary")
