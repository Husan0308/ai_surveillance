import numpy as np
from .schemas import FaceMatch,FaceDecision

class KnownPersonMatcher:
    def __init__(self,gallery,threshold=.55,strong=.75,ambiguity_margin=.05,top_k=3):self.gallery=gallery;self.threshold=threshold;self.strong=strong;self.margin=ambiguity_margin;self.top_k=top_k
    def match(self,embedding):
        people=self.gallery.enabled()
        if embedding is None or not people:return FaceMatch(None,None,0,0,0,0,FaceDecision.UNKNOWN)
        query=np.asarray(embedding,np.float32);query/=max(np.linalg.norm(query),1e-12);scores=[]
        for person in people:
            sims=np.stack(person.embeddings)@query;best=float(sims.max());top=float(np.sort(sims)[-self.top_k:].mean());scores.append((.7*best+.3*top,person))
        scores.sort(key=lambda item:item[0],reverse=True);best,person=scores[0];second=scores[1][0] if len(scores)>1 else 0;margin=best-second
        if best<self.threshold:return FaceMatch(None,None,best,best,second,margin,FaceDecision.UNKNOWN)
        if margin<self.margin:return FaceMatch(None,None,best,best,second,margin,FaceDecision.AMBIGUOUS)
        return FaceMatch(person.person_id,person.name,best,best,second,margin,FaceDecision.CONFIRMED if best>=self.strong else FaceDecision.PENDING)
