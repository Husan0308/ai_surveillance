import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from services.api_service.app import app

class ApiRouteTests(unittest.TestCase):
    def test_health_and_degraded_dependencies(self):
        class OfflineML:
            def __init__(self,*_args,**_kwargs):pass
            async def health(self):return None
        with patch("services.api_service.app.MLClient",OfflineML):
         with TestClient(app) as client:
            body=client.get("/api/v1/health").json()
            self.assertEqual(body["status"],"degraded");self.assertTrue(body["dependencies"]["sqlite"])
            self.assertEqual(client.get("/api/v1/persons").status_code,200)
    def test_validation(self):
        with TestClient(app) as client:
            self.assertEqual(client.post("/api/v1/persons",json={"name":""}).status_code,422)
            self.assertEqual(client.patch("/api/v1/settings",json={"detection_confidence":2}).status_code,422)
            self.assertEqual(client.get("/api/v1/heatmaps/CAM-01/floorplan").status_code,422)
    def test_enrollment_ml_unavailable_is_explicit(self):
        class OfflineML:
            def __init__(self,*_args,**_kwargs):pass
            async def health(self):return None
            async def command(self,_message):
                from fastapi import HTTPException
                raise HTTPException(503,"ML service unavailable")
        with patch("services.api_service.app.MLClient",OfflineML):
            with TestClient(app) as client:
                from shared.enrollment_paths import stage_files
                import tempfile,cv2,numpy as np
                with tempfile.TemporaryDirectory() as tmp:
                    sources=[]
                    for i in range(10):
                        path=f"{tmp}/face-{i}.jpg";cv2.imwrite(path,np.full((20,20,3),127,np.uint8));sources.append(path)
                    paths=stage_files(sources)
                    response=client.post("/api/v1/enrollment/sessions",json={"name":"Husan","sample_paths":paths})
                self.assertEqual(response.status_code,503)
