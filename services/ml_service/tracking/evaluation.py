"""Deterministic P2 local-tracking evaluation with no detector inference."""
import json,time
from dataclasses import dataclass
from services.ml_service.detection.schemas import Detection,CameraDetectionResult
from .camera_tracker import CameraTracker
from .official_bytetrack import OfficialByteTrackAdapter,nvdcf_capability

CFG={"min_confirmed_hits":2,"track_high_thresh":.22,"track_low_thresh":.05,"new_track_thresh":.28,
     "match_thresh":.35,"relaxed_match_thresh":.18,"effective_ai_fps":10,"max_lost_time_ms":1500}

@dataclass(frozen=True)
class Scenario:
    name:str
    frames:tuple

def _box(x,y=10,w=20,h=60):return (x,y,x+w,y+h)
def scenarios():
    a=[((_box(10+i),),) for i in range(30)]+[((),)]*5+[((_box(70+i),),) for i in range(10)]
    b=[((_box(10+i),),) for i in range(20)]+[((),)]*15+[((_box(45+i),),) for i in range(10)]
    crossing=[];similar=[];turn=[]
    for i in range(30):
        crossing.append(((_box(5+3*i),_box(105-3*i)),))
        similar.append(((_box(5+3*i),_box(105-3*i)),))
        turn.append(((_box(20+i),),))
    return (Scenario("A_short_5",tuple(a)),Scenario("B_long_15",tuple(b)),
            Scenario("C_crossing",tuple(crossing)),Scenario("D_similar_overlap",tuple(similar)),
            Scenario("E_turnaround",tuple(turn)))

def _result(frame,boxes):
    stamp=1000+frame*.1
    return CameraDetectionResult("CAM-EVAL",frame,stamp,stamp,tuple(Detection(tuple(box),.9) for box in boxes))

def evaluate(factory,scenario):
    tracker=factory();truth_ids={};switches=false_merges=fragments=0;started=time.perf_counter()
    for frame,item in enumerate(scenario.frames,1):
        boxes=item[0];out=tracker.update(_result(frame,boxes))
        visible=[t for t in out.tracks if t.state.value!="LOST"]
        assigned=set()
        for truth,box in enumerate(boxes):
            cx=(box[0]+box[2])/2
            candidates=sorted(((abs((t.bbox[0]+t.bbox[2])/2-cx),t.track_id) for t in visible if t.track_id not in assigned))
            if not candidates:continue
            track_id=candidates[0][1];assigned.add(track_id)
            previous=truth_ids.get(truth)
            if previous is not None and previous!=track_id:switches+=1;fragments+=1
            truth_ids[truth]=track_id
    elapsed=(time.perf_counter()-started)*1000
    continuity=switches==0
    return {"continuity":continuity,"id_switches":switches,"fragments":fragments,"false_merges":false_merges,
            "cpu_ms_total":elapsed,"cpu_ms_per_update":elapsed/max(len(scenario.frames),1)}

def run_comparison():
    output={"current_baseline":{},"current":{},"official_bytetrack":{},"nvdcf":nvdcf_capability()}
    for scenario in scenarios():
        output["current_baseline"][scenario.name]=evaluate(lambda:CameraTracker("CAM-EVAL",{**CFG,"recovery_motion_enabled":False}),scenario)
        output["current"][scenario.name]=evaluate(lambda:CameraTracker("CAM-EVAL",CFG),scenario)
        output["official_bytetrack"][scenario.name]=evaluate(lambda:OfficialByteTrackAdapter("CAM-EVAL",CFG),scenario)
    return output
if __name__=="__main__":print(json.dumps(run_comparison(),indent=2,sort_keys=True))
