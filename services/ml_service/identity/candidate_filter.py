import time

class CameraTopology:
    def __init__(self, config=None):
        config=config or {}; self.camera_rooms=dict(config.get("camera_rooms",{}))
        topology=config.get("identity",{}).get("topology",config.get("topology",{}))
        self.relationships=topology.get("relationships",{}); self.min_travel_ms=topology.get("min_travel_time_ms",{})
    def room(self,camera): return self.camera_rooms.get(camera,camera)
    def relationship(self,a,b):
        if a==b:return "same_camera"
        ra,rb=self.room(a),self.room(b)
        if ra==rb:return "same_room"
        return self.relationships.get(f"{ra}:{rb}",self.relationships.get(f"{rb}:{ra}","possible_transition"))

class CandidateFilter:
    def __init__(self,topology,max_candidate_age_ms=300000): self.topology=topology; self.max_age=max_candidate_age_ms
    def filter(self,observation,identities):
        accepted=[]; now=observation.timestamp
        for identity in identities:
            if identity.status.value=="ARCHIVED" or (now-identity.last_seen_at)*1000>self.max_age: continue
            relation=self.topology.relationship(identity.last_camera_id,observation.camera_id)
            gap=(now-identity.last_seen_at)*1000
            if relation=="impossible_transition" and gap<self.max_age: continue
            key=f"{self.topology.room(identity.last_camera_id)}:{self.topology.room(observation.camera_id)}"
            if gap<float(self.topology.min_travel_ms.get(key,0)): continue
            accepted.append((identity,relation,gap))
        return accepted
