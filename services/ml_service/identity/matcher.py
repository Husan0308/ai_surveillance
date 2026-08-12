import numpy as np

class IdentityMatcher:
    def __init__(self,config=None):
        cfg=(config or {}).get("identity",config or {});weights=cfg.get("weights",{})
        self.w_reid=float(weights.get("reid",.72));self.w_time=float(weights.get("time",.10));self.w_camera=float(weights.get("camera",.10));self.w_quality=float(weights.get("quality",.08))

    @staticmethod
    def robust_similarity(query,identity):
        history=[(np.asarray(item[0],np.float32),float(item[1])) for item in getattr(identity,"appearance_history",()) if item[0] is not None]
        if not history and identity.appearance_embedding is not None:history=[(np.asarray(identity.appearance_embedding,np.float32),1.0)]
        if not history:return None
        scored=[]
        for embedding,quality in history:
            embedding=embedding/max(float(np.linalg.norm(embedding)),1e-12);scored.append((float(embedding@query),max(.01,quality)))
        # Quality-weighted top-k is robust to one weak representative without
        # allowing a single noisy maximum to dominate a populated gallery.
        selected=sorted(scored,reverse=True)[:3];weight=sum(item[1] for item in selected)
        return float(sum(similarity*quality for similarity,quality in selected)/max(weight,1e-12))

    def score(self,observation,candidates):
        if observation.appearance_embedding is None or not candidates:return []
        query=np.asarray(observation.appearance_embedding,np.float32).reshape(-1);query/=max(float(np.linalg.norm(query)),1e-12);output=[]
        relation_score={"same_camera":1.0,"same_room":1.0,"overlapping":1.0,"adjacent_room":.8,"possible_transition":.65,"unverified":.0}.get
        for identity,relation,gap in candidates:
            similarity=self.robust_similarity(query,identity)
            if similarity is None:continue
            time_score=max(0.0,1.0-gap/300000);camera_score=relation_score(relation,0.0);final=self.w_reid*similarity+self.w_time*time_score+self.w_camera*camera_score+self.w_quality*float(observation.quality_score)
            output.append((identity,float(final),similarity,relation))
        # Raw normalized OSNet similarity is the primary identity evidence.
        return sorted(output,key=lambda item:(item[2],item[1]),reverse=True)
