import time
import numpy as np

class IdentityMatcher:
    def __init__(self,config=None):
        cfg=(config or {}).get("identity",config or {}); weights=cfg.get("weights",{})
        self.w_reid=float(weights.get("reid",.72)); self.w_time=float(weights.get("time",.10)); self.w_camera=float(weights.get("camera",.10)); self.w_quality=float(weights.get("quality",.08))
    def score(self,observation,candidates):
        if observation.appearance_embedding is None or not candidates:return []
        query=np.asarray(observation.appearance_embedding,np.float32); query/=max(np.linalg.norm(query),1e-12)
        valid=[item for item in candidates if item[0].appearance_embedding is not None]
        if not valid:return []
        gallery=np.stack([item[0].appearance_embedding for item in valid]); similarities=gallery@query
        relation_score={"same_camera":1.0,"same_room":1.0,"adjacent_room":.8,"possible_transition":.65}.get
        output=[]
        for index,(identity,relation,gap) in enumerate(valid):
            time_score=max(0.0,1.0-gap/300000); camera_score=relation_score(relation,0.0)
            final=self.w_reid*float(similarities[index])+self.w_time*time_score+self.w_camera*camera_score+self.w_quality*observation.quality_score
            output.append((identity,float(final),float(similarities[index]),relation))
        return sorted(output,key=lambda item:item[1],reverse=True)
