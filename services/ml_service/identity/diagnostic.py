import time,numpy as np
from .global_identity_manager import GlobalIdentityManager
from .schemas import IdentityTrackObservation

def run_identity_diagnostic():
    config={"camera_rooms":{"CAM-01":"ROOM-01","CAM-06":"ROOM-01","CAM-02":"ROOM-02"},
            "identity":{"strong_match_threshold":.85,"match_threshold":.75,"new_identity_threshold":.55}}
    manager=GlobalIdentityManager(config);now=time.time()
    def observe(camera,track,embedding):
        result=manager.update(IdentityTrackObservation(camera,1,track,(0,0,50,150),.9,now,np.asarray(embedding,np.float32),.9))
        item=result.tracks[0];print(f"{camera} {track} -> {item.global_id} score={item.identity_confidence:.3f} reason={item.decision_reason}")
        return item.global_id
    first=observe("CAM-01","CAM-01:TRACK-00012",[1,0,0])
    overlapping=observe("CAM-06","CAM-06:TRACK-00031",[.99,.01,0])
    different=observe("CAM-02","CAM-02:TRACK-00014",[0,1,0])
    assert first==overlapping and different!=first
    return manager.metrics.snapshot()

if __name__=="__main__":print(run_identity_diagnostic())
