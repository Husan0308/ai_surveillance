from collections import defaultdict
from services.ml_service.face.schemas import IdentityResolutionResult,FaceDecision

class IdentityResolver:
    def __init__(self,identity_store,evidence_required=3,strong_quality=.8):self.store=identity_store;self.required=evidence_required;self.strong_quality=strong_quality;self.evidence=defaultdict(list);self.bindings={};self.conflicts=[]
    def resolve(self,global_id,match,quality):
        global_id=self.store.canonicalize(global_id) if hasattr(self.store,"canonicalize") else global_id
        existing=self.bindings.get(global_id)
        if match.person_id is None:return IdentityResolutionResult(global_id,existing[0] if existing else None,existing[1] if existing else None,match.similarity,quality,existing[2] if existing else 0,match.decision)
        if existing and existing[0]!=match.person_id:
            self.conflicts.append((global_id,existing[0],match.person_id,match.similarity))
            return IdentityResolutionResult(global_id,existing[0],existing[1],match.similarity,quality,existing[2],match.decision,True)
        self.evidence[global_id].append((match.person_id,match.name,match.similarity,quality));recent=self.evidence[global_id][-self.required:]
        confirm=(match.decision==FaceDecision.CONFIRMED and quality>=self.strong_quality) or (len(recent)>=self.required and len({e[0] for e in recent})==1)
        if confirm:
            confidence=sum(e[2]*e[3] for e in recent)/max(sum(e[3] for e in recent),1e-12);self.bindings[global_id]=(match.person_id,match.name,confidence)
            identity=self.store.get(global_id)
            if identity is not None:identity.identity_type="KNOWN";identity.person_id=match.person_id;identity.display_name=match.name
            return IdentityResolutionResult(global_id,match.person_id,match.name,match.similarity,quality,confidence,FaceDecision.CONFIRMED)
        return IdentityResolutionResult(global_id,None,None,match.similarity,quality,0,FaceDecision.PENDING)
