import time
from .face_extractor import FaceExtractor
from .schemas import FaceEmbeddingResult,FaceDecision

class FaceManager:
    def __init__(self,detector,quality,matcher,resolver,attempt_interval_ms=3000,max_attempts=3,retry_low_quality_ms=1500):
        self.detector,self.quality,self.matcher,self.resolver=detector,quality,matcher,resolver;self.extractor=FaceExtractor()
        self.interval=attempt_interval_ms/1000;self.max_attempts=max_attempts;self.retry_low=retry_low_quality_ms/1000;self.state={};self.model_identity=id(getattr(detector,"engine",None))
    def process(self,candidate,frame):
        if candidate.global_id in self.resolver.bindings:
            person_id,name,confidence=self.resolver.bindings[candidate.global_id]
            return __import__("services.ml_service.face.schemas",fromlist=["IdentityResolutionResult"]).IdentityResolutionResult(candidate.global_id,person_id,name,0,0,confidence,FaceDecision.CONFIRMED)
        now=time.time();state=self.state.setdefault((candidate.camera_id,candidate.local_track_id),{"last":0,"attempts":0})
        if now-state["last"]<self.interval or state["attempts"]>=self.max_attempts:return None
        state["last"]=now;state["attempts"]+=1;x1,y1,x2,y2=[int(v) for v in candidate.person_bbox];crop=frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
        if not getattr(crop,"size",0):return None
        detections=self.detector.detect(crop)
        if not detections:return self.resolver.resolve(candidate.global_id,self.matcher.match(None),0)
        detection=max(detections,key=lambda item:item.confidence);quality=self.quality.score(crop,detection)
        if not quality.accepted:
            state["last"]=now-self.interval+self.retry_low
            return self.resolver.resolve(candidate.global_id,self.matcher.match(None),quality.score)
        embedding=self.extractor.normalize(detection.embedding);match=self.matcher.match(embedding)
        return self.resolver.resolve(candidate.global_id,match,quality.score)
