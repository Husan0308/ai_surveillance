"""Single shared batched ReID wrapper; may reuse tracking AppearanceExtractor."""
import time

class ReIDExtractor:
    def __init__(self,appearance_extractor): self.appearance_extractor=appearance_extractor
    @property
    def model_identity(self): return id(getattr(self.appearance_extractor,"model",None))
    def extract_batch(self,crops):
        started=time.perf_counter(); embeddings,timing=self.appearance_extractor.extract_batch(crops)
        timing=dict(timing); timing.setdefault("total_ms",(time.perf_counter()-started)*1000)
        return embeddings,timing
