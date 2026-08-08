import time,uuid,numpy as np
from .schemas import EnrollmentSession,EnrollmentSample

class EnrollmentService:
    def __init__(self,gallery,target_samples=10,min_samples=6,min_average_quality=.65,diversity_distance=.08,duplicate_threshold=.85,event_sink=None):
        self.gallery=gallery;self.target=target_samples;self.minimum=min_samples;self.min_average=min_average_quality;self.diversity=diversity_distance;self.duplicate_threshold=duplicate_threshold;self.event_sink=event_sink or (lambda *_:None);self.sessions={}
    def start(self,person_id,name):
        session=EnrollmentSession(str(uuid.uuid4()),person_id,name,time.time(),self.target);self.sessions[session.session_id]=session;return session
    def add_sample(self,session_id,embedding,quality):
        session=self.sessions[session_id]
        if quality<self.min_average or embedding is None:self.event_sink("enrollment.rejected",{"quality":quality});return False
        value=np.asarray(embedding,np.float32);value/=max(np.linalg.norm(value),1e-12)
        if any(1-float(np.dot(value,s.embedding))<self.diversity for s in session.samples):return False
        session.samples.append(EnrollmentSample(value,quality,time.time()));self.event_sink("enrollment.capture",{"count":len(session.samples)});self.event_sink("enrollment.progress",{"count":len(session.samples),"target":session.target_samples});return True
    def finish(self,session_id):
        session=self.sessions[session_id];qualities=[s.quality for s in session.samples]
        if len(qualities)<self.minimum or sum(qualities)/len(qualities)<self.min_average:
            session.state="FAILED";self.event_sink("enrollment.failed",{"reason":"insufficient_good_samples"});return {"ok":False,"reason":"insufficient_good_samples"}
        centroid=np.mean([s.embedding for s in session.samples],axis=0);duplicate=None
        for person in self.gallery.enabled():
            if max(float(np.dot(centroid,e)) for e in person.embeddings)>=self.duplicate_threshold:duplicate=person;break
        if duplicate:return {"ok":False,"reason":"potential_duplicate_person","person_id":duplicate.person_id}
        self.gallery.add(session.person_id,session.name,[s.embedding for s in session.samples]);session.state="COMPLETED";self.event_sink("enrollment.completed",{"person_id":session.person_id});return {"ok":True,"person_id":session.person_id}
