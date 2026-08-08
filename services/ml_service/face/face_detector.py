from .schemas import FaceDetection

class FaceDetector:
    """Non-owning adapter around the single shared InsightFace engine."""
    def __init__(self,engine):self.engine=engine
    def detect(self,person_crop):
        return [FaceDetection(tuple(item["bbox"]),float(item.get("det_score",0)),item.get("landmarks"),item.get("pose"),item.get("embedding")) for item in self.engine.detect(person_crop,need_embedding=True)]
