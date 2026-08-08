import numpy as np

class FaceExtractor:
    def normalize(self,embedding):
        if embedding is None:return None
        value=np.asarray(embedding,np.float32);norm=np.linalg.norm(value)
        return value/norm if norm else None
