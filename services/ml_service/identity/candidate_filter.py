class CameraTopology:
    def __init__(self,config=None):
        config=config or {};topology=config.get("identity",{}).get("topology",config.get("topology",{}));self.camera_rooms=dict(topology.get("camera_rooms",config.get("camera_rooms",{})))
        self.relationships=topology.get("relationships",{});self.camera_relationships=topology.get("camera_relationships",{});self.min_travel_ms=topology.get("min_travel_time_ms",{})
        self.default_min_travel_ms=float(topology.get("default_min_travel_ms",20000))
        self.verified=bool(topology.get("verified",bool(self.camera_rooms)))
        self.overlapping={frozenset(pair) for pair in topology.get("overlapping_camera_pairs",()) if len(pair)==2}
    def room(self,camera):return self.camera_rooms.get(camera,camera)
    def relationship(self,a,b):
        if a==b:return "same_camera"
        if not self.verified:return "unverified"
        if frozenset((a,b)) in self.overlapping:return "overlapping"
        explicit=self.camera_relationships.get(f"{a}:{b}")
        if explicit:return explicit
        ra,rb=self.room(a),self.room(b)
        if ra==rb:return "same_room"
        return self.relationships.get(f"{ra}:{rb}",self.relationships.get(f"{rb}:{ra}","possible_transition"))
    def minimum_travel_ms(self,a,b):
        ra,rb=self.room(a),self.room(b);key=f"{ra}:{rb}";reverse=f"{rb}:{ra}"
        return float(self.min_travel_ms.get(key,self.min_travel_ms.get(reverse,self.default_min_travel_ms)))

class CandidateFilter:
    def __init__(self,topology,max_candidate_age_ms=300000,active_window_ms=2000):
        self.topology=topology;self.max_age=float(max_candidate_age_ms);self.active_window_ms=float(active_window_ms);self.last_rejections=0;self.last_same_camera_conflicts=0
    def filter(self,observation,identities):
        accepted=[];now=observation.timestamp;self.last_rejections=0;self.last_same_camera_conflicts=0
        for identity in identities:
            if identity.status.value=="ARCHIVED" or (now-identity.last_seen_at)*1000>self.max_age:continue
            if not getattr(identity,"gallery_trusted",True):self.last_rejections+=1;continue
            if not getattr(identity,"appearance_history",()) and getattr(identity,"appearance_embedding",None) is None:continue
            # Remove a canonical candidate already occupied by a different active
            # local trajectory in this camera before top-2 discrimination.
            active_track=getattr(identity,"active_tracks",{}).get(observation.camera_id)
            active_seen=float(getattr(identity,"active_track_seen",{}).get(observation.camera_id,0))
            if active_track is not None and str(active_track)!=str(observation.local_track_id) and (now-active_seen)*1000<=self.active_window_ms:
                self.last_rejections+=1;self.last_same_camera_conflicts+=1;continue
            relation=self.topology.relationship(identity.last_camera_id,observation.camera_id);gap=max(0,(now-identity.last_seen_at)*1000)
            # Unverified topology is neutral: never invent a relationship, but do not
            # reject a strong appearance match unless a verified rule contradicts it.
            if self.topology.verified and relation=="impossible_transition":self.last_rejections+=1;continue
            if self.topology.verified and relation not in ("same_camera","same_room","overlapping") and gap<self.topology.minimum_travel_ms(identity.last_camera_id,observation.camera_id):
                self.last_rejections+=1;continue
            accepted.append((identity,relation,gap))
        return accepted
