#!/usr/bin/env python3
"""Fail closed on missing/short/frozen CAM-01 stability evidence. No camera access."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess


def rows_from_log(text):
    return [{k: float(v) for k,v in re.findall(r'(\w+)=([\d.]+)', line)}
            for line in text.splitlines() if line.startswith('CAM-01 STATS ')]


def assess(rows, text, gpu):
    errors=[]
    stable=[r for r in rows if r['elapsed']>=30]
    if len(stable)<2:return {'status':'BLOCKED','failures':['Insufficient telemetry']}
    span=stable[-1]['elapsed']-stable[0]['elapsed']
    if span<600:errors.append('Less than 600 seconds after warmup')
    if 'hardware_decoder=1 fatal=0' not in text or 'CAM-01 END ' not in text:
        errors.append('Missing clean completion/hardware decoder evidence')
    if 'memory:NVMM' not in text or 'nvstreammux(batch=1,' not in text:
        errors.append('Missing NVMM / single-source mux evidence')
    if re.search(r'CAM-01 FATAL|CRITICAL|MESA: error|libEGL warning|\bERROR\s',text):
        errors.append('Fatal/critical/runtime diagnostic in log')
    fields=['errors','warnings','recoveries','dropped','pts_backwards','pts_duplicates','rtp_lost','rtp_late']
    if any(r.get(k,0)>0 for r in rows for k in fields):errors.append('Error, recovery, packet loss, drop or non-advancing timestamp')
    for a,b in zip(stable,stable[1:]):
        dt=b['elapsed']-a['elapsed']
        if not 0<dt<=7:errors.append('Missing/nonmonotonic telemetry interval');break
        if b['output']<=a['output'] or b['pts_ns']<=a['pts_ns']:
            errors.append('Frozen frame/PTS counter');break
    if any(not 18<=r['fps']<=22 or r['age']>1 or r['max_gap_ms']>1000 for r in stable):
        errors.append('Frame rate/arrival stall threshold exceeded')
    if any(r['queue']>=12 or r['queue_ms']>600 for r in stable):errors.append('Queue backlog')
    if any(abs(r['decoded']-r['output'])>12 for r in stable):errors.append('Decode/output frames diverge')
    rss_growth=statistics.median(r['rss_mib'] for r in stable[-12:])-statistics.median(r['rss_mib'] for r in stable[:12])
    cpu_growth=statistics.median(r['cpu_pct'] for r in stable[-12:])-statistics.median(r['cpu_pct'] for r in stable[:12])
    if rss_growth>32:errors.append('RSS grew more than 32 MiB between steady-state windows')
    if cpu_growth>10 or statistics.mean(r['cpu_pct'] for r in stable)>35:
        errors.append('CPU growth/usage exceeded threshold')
    g=[r for r in gpu if 30<=r['elapsed']<=rows[-1]['elapsed']]
    if len(g)<100:errors.append('Insufficient GPU telemetry')
    else:
        if max(r['memory_used_mib'] for r in g)-min(r['memory_used_mib'] for r in g)>64:
            errors.append('VRAM spread exceeded 64 MiB')
        if max(r['decoder_pct'] for r in g)<=0:errors.append('No NVDEC utilization evidence')
        if max(b['elapsed']-a['elapsed'] for a,b in zip(g,g[1:]))>10:errors.append('GPU telemetry gap')
    summary={'status':'PASS' if not errors else 'BLOCKED','failures':errors,
             'measured_seconds':rows[-1]['elapsed'],'steady_seconds':span,
             'frames':int(rows[-1]['output']),
             'fps_min':min(r['fps'] for r in stable),'fps_max':max(r['fps'] for r in stable),
             'fps_mean':statistics.mean(r['fps'] for r in stable),
             'cpu_pct_mean_one_core':statistics.mean(r['cpu_pct'] for r in stable),
             'cpu_pct_max_one_core':max(r['cpu_pct'] for r in stable),'cpu_growth_points':cpu_growth,
             'rss_mib_start':stable[0]['rss_mib'],'rss_mib_end':stable[-1]['rss_mib'],'rss_growth_mib':rss_growth,
             'max_gap_ms':max(r['max_gap_ms'] for r in stable),'queue_max':max(r['queue'] for r in stable)}
    for key in ['memory_used_mib','gpu_pct','decoder_pct','encoder_pct']:
        if g:summary[key]={'min':min(r[key] for r in g),'max':max(r[key] for r in g),'mean':statistics.mean(r[key] for r in g)}
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);args=ap.parse_args()
    text=(args.directory/'pipeline.log').read_text();rows=rows_from_log(text)
    gpu=[]
    with (args.directory/'gpu.csv').open() as f:
        for row in csv.DictReader(f):
            gpu.append({k:float(row[k]) for k in ['time','memory_used_mib','gpu_pct','decoder_pct','encoder_pct']})
    if gpu:
        start=gpu[0]['time']
        for r in gpu:r['elapsed']=r['time']-start
    report=assess(rows,text,gpu)
    probe=subprocess.run(['ffprobe','-v','error','-count_packets','-select_streams','v:0',
        '-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_packets','-show_entries','format=duration',
        '-of','json',str(args.directory/'CAM-01.mkv')],capture_output=True,text=True,timeout=60)
    try:
        video=json.loads(probe.stdout);stream=video['streams'][0]
        if probe.returncode or float(video['format']['duration'])<600 or int(stream['nb_read_packets'])<12000:
            raise ValueError('Video too short or incomplete')
        if stream['width']!=2560 or stream['height']!=1440 or stream['codec_name']!='h264':
            raise ValueError('Unexpected output format')
        if abs(int(stream['nb_read_packets'])-int(rows[-1]['output']))>12:
            raise ValueError('Encoded output counters and recorded packets disagree')
        report['recorded_video']=video
    except (ValueError,KeyError,IndexError):
        report['status']='BLOCKED';report['failures'].append('Recorded video validation failed')
    (args.directory/'stability.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
