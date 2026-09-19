#!/usr/bin/env python3
"""Assess the 60-second live detector gate; visual inspection is separately required."""
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cam_six_validation.check_stability import parse_rows


def assess(text):
    failures=[]; report={'cameras':{}}
    group=parse_rows(text,'GROUP STATS ')
    yolo=parse_rows(text,'YOLO STATS ')
    if not group or group[-1].get('elapsed',0)<60:
        failures.append('Less than 60 seconds of camera runtime')
    if not re.search(r'GROUP END .*fatal=0 shared_errors=0 shared_warnings=0',text):
        failures.append('Missing clean group completion')
    if 'inference=1' not in text or 'nvinfer(YOLO26m,FP16,batch=6,interval=0)' not in text:
        failures.append('Missing primary inference graph evidence')
    if re.search(r'GROUP FATAL|CRITICAL|PARSER_ERROR|\bERROR\s',text):
        failures.append('Runtime error diagnostic')
    for i in range(1,7):
        cid=f'CAM-{i:02d}'
        source=parse_rows(text,f'{cid} STATS ')
        detect=parse_rows(text,f'{cid} DETECT ')
        steady=[r for r in source if r['elapsed']>=15]
        if len(steady)<9 or not detect or detect[-1].get('frames',0)<900:
            failures.append(f'{cid}: insufficient source/inference evidence');continue
        if not re.search(rf'{cid} END .*hardware_decoder=1 errors=0 warnings=0',text):
            failures.append(f'{cid}: missing clean decoder completion')
        if any(not 18<=r['fps']<=22 or r['age']>1 or r['queue']>=12 or r['queue_ms']>600 for r in steady):
            failures.append(f'{cid}: source throughput/backlog failure')
        if any(r[k]!=0 for r in source for k in ['rtp_lost','rtp_late','errors','warnings','pts_backwards','pts_duplicates']):
            failures.append(f'{cid}: packet/error/timestamp failure')
        if any(r['decoder']!=1 for r in source):failures.append(f'{cid}: hardware decode absent')
        for a,b in zip(steady,steady[1:]):
            if not 0<b['elapsed']-a['elapsed']<=7 or b['input']<=a['input'] or b['pts_ns']<=a['pts_ns']:
                failures.append(f'{cid}: frozen source or telemetry gap');break
        if any(r['errors'] or r['detections']!=r['persons'] for r in detect):
            failures.append(f'{cid}: invalid detection metadata')
        if any(b['frames']<=a['frames'] for a,b in zip(detect[:-2],detect[1:-1])):
            failures.append(f'{cid}: inference stopped advancing')
        if abs(source[-1]['input']-detect[-1]['frames'])>24:
            failures.append(f'{cid}: source/inference frame divergence')
        if steady[-1]['queue']>steady[0]['queue']+6:
            failures.append(f'{cid}: growing queue')
        report['cameras'][cid]=dict(source_frames=int(source[-1]['input']),
            source_fps_mean=statistics.mean(r['fps'] for r in steady),
            source_fps_min=min(r['fps'] for r in steady),source_fps_max=max(r['fps'] for r in steady),
            queue_max=max(r['queue'] for r in steady),inferred_frames=int(detect[-1]['frames']),
            person_detections=int(detect[-1]['persons']),last_detection_unix_ms=int(detect[-1]['last_detection_unix_ms']))
    if not yolo or yolo[-1].get('persons',0)==0:
        failures.append('No real person metadata observed')
    elif any(r[k]!=0 for r in yolo for k in ['errors','parser_errors','parser_rejected']):
        failures.append('Inference/parser failure')
    else:report['inference']=yolo[-1]
    if group:
        steady=[r for r in group if r['elapsed']>=15]
        if any(r[k] for r in group for k in ['dropped','shared_errors','shared_warnings','pts_backwards','pts_duplicates']):
            failures.append('Tiled output failure')
        if any(b['output']<=a['output'] or b['pts_ns']<=a['pts_ns'] for a,b in zip(steady,steady[1:])):
            failures.append('Frozen tiled output')
        if steady:
            if steady[-1]['rss_mib']-steady[0]['rss_mib']>96:failures.append('RSS growth exceeds 96 MiB during short gate')
            report.update(measured_seconds=group[-1]['elapsed'],output_frames=int(group[-1]['output']),
                cpu_pct_mean_one_core=statistics.mean(r['cpu_pct'] for r in steady),
                rss_mib_start=steady[0]['rss_mib'],rss_mib_end=steady[-1]['rss_mib'])
    report.update(status='PASS' if not failures else 'BLOCKED',failures=failures)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    out=args.directory; report=assess((out/'pipeline.log').read_text())
    failures=report['failures']
    preview=json.loads((out/'preview.json').read_text())
    if not preview.get('enabled') or not preview.get('alive_at_end'):failures.append('Preview unavailable or exited')
    with (out/'gpu.csv').open() as f:gpu=list(csv.DictReader(f))
    g=[{k:float(r[k]) for k in ['time','memory_used_mib','gpu_pct','decoder_pct','encoder_pct']} for r in gpu]
    g=[r for r in g if r['time']-float(gpu[0]['time'])>=15]
    if len(g)<8 or max(r['decoder_pct'] for r in g)<=0:failures.append('Insufficient GPU/NVDEC evidence')
    elif max(r['memory_used_mib'] for r in g)-min(r['memory_used_mib'] for r in g)>128:
        failures.append('VRAM growth/spread exceeds 128 MiB')
    report['gpu']={k:dict(min=min(r[k] for r in g),max=max(r[k] for r in g),mean=statistics.mean(r[k] for r in g)) for k in ['memory_used_mib','gpu_pct','decoder_pct','encoder_pct']} if g else {}
    video=subprocess.run(['ffprobe','-v','error','-count_packets','-select_streams','v:0',
        '-show_entries','stream=width,height,codec_name,nb_read_packets','-show_entries','format=duration',
        '-of','json',str(out/'CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv')],capture_output=True,text=True,timeout=60)
    try:
        v=json.loads(video.stdout);st=v['streams'][0]
        assert video.returncode==0 and float(v['format']['duration'])>=58
        assert (st['width'],st['height'],st['codec_name'])==(1920,1620,'h264')
        assert abs(int(st['nb_read_packets'])-report['output_frames'])<=16
        report['recording']=v
    except (ValueError,KeyError,IndexError,AssertionError):failures.append('Finalized recording validation failed')
    records=[json.loads(line) for line in (out/'detections.jsonl').read_text().splitlines()]
    if not records:failures.append('Missing object metadata evidence')
    frames={}
    for r in records:
        if r['class_id']!=0 or not .25<=r['confidence']<=1 or not all(math.isfinite(x) for x in r['box']):
            failures.append('Non-person or invalid sampled metadata');break
        frames.setdefault((r['source_id'],r['frame']),[]).append(r)
    duplicates=0
    for boxes in frames.values():
        for i,a in enumerate(boxes):
            x,y,w,h=a['box']
            for b in boxes[i+1:]:
                xx,yy,ww,hh=b['box'];intersection=max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))
                if intersection/(w*h+ww*hh-intersection)>0.9:duplicates+=1
    if duplicates:failures.append('Near-identical person boxes in sampled frames; visual review required')
    report['sampled_metadata_rows']=len(records);report['near_duplicate_pairs']=duplicates
    visual_path=out/'visual_review.json'
    if not visual_path.exists():
        failures.append('Missing separate visual review of live preview/person geometry')
    else:
        visual=json.loads(visual_path.read_text())
        report['visual_review']=visual
        digest=hashlib.sha256((out/'CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv').read_bytes()).hexdigest()
        if visual.get('status')!='PASS' or visual.get('recording_sha256')!=digest:
            failures.append('Visual review did not pass for this recording')
    report['status']='BLOCKED' if failures else 'PASS'
    (out/'gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
