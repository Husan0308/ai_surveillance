#!/usr/bin/env python3
"""Read-only 30-minute ML soak sampler with rolling latency and growth warnings."""
from __future__ import annotations
import argparse,json,statistics,subprocess,time
from pathlib import Path
from urllib.request import urlopen

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("pid",type=int,help="ML service PID")
p.add_argument("--interval",type=float,default=60,help="seconds between samples (default: 60)")
p.add_argument("--samples",type=int,default=30,help="sample count (default: 30 = 30 minutes)")
p.add_argument("--metrics-url",default="http://127.0.0.1:8001/metrics")
p.add_argument("--database",default="data/surveillance.db")
a=p.parse_args();status=Path(f"/proc/{a.pid}/status");db=Path(a.database)
history={"rss":[],"vram":[],"threads":[],"db":[],"wal":[]};detector=[]

def percentile(values,q):
 values=sorted(values)
 if not values:return 0.0
 return values[min(len(values)-1,round((len(values)-1)*q))]
def increasing(values):
 return len(values)>=5 and all(b>=c for c,b in zip(values[-5:],values[-4:])) and values[-1]>values[-5]
def size(path):
 try:return path.stat().st_size
 except OSError:return 0
def gpu_memory():
 out=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
 return int(next((row.split(",",1)[1].strip() for row in out.splitlines() if row.split(",",1)[0].strip()==str(a.pid)),"0"))
def metric_payload():
 try:
  with urlopen(a.metrics_url,timeout=3) as response:return json.load(response)
 except Exception as exc:return {"monitor_error":str(exc)}

for index in range(a.samples):
 if not status.exists():print("process exited",flush=True);break
 values={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in status.read_text().splitlines() if ":" in line}
 rss=int(values.get("VmRSS","0 kB").split()[0]);threads=int(values.get("Threads",0));vram=gpu_memory();db_size=size(db);wal_size=size(Path(str(db)+"-wal"));m=metric_payload()
 cycle=m.get("cycle",{});current=float(cycle.get("pure_detector",0));detector.append(current)
 cameras=m.get("cameras",{});online=sum(bool(x.get("online")) for x in cameras.values());secondary=m.get("secondary",{});queues=sum(int(x.get("queue_depth",0)) for x in secondary.values());snap=m.get("unknown_snapshots",{});tracking=m.get("tracking",{});track_cameras=tracking.get("cameras",{});active=sum(int(x.get("tracks_active",0)) for x in track_cameras.values());lost=sum(int(x.get("tracks_lost",0)) for x in track_cameras.values());global_ids=int(m.get("identity",{}).get("global_identities_active",0))
 for key,value in (("rss",rss),("vram",vram),("threads",threads),("db",db_size),("wal",wal_size)):history[key].append(value)
 flags=[key+"_growth" for key,items in history.items() if increasing(items)]
 print(f"sample={index+1}/{a.samples} rss_mib={rss/1024:.1f} vram_mib={vram} threads={threads} cameras={online}/{len(cameras)} batch_rate={float(m.get('batch_rate',0)):.2f}/s detector_rolling_p50={percentile(detector,.50):.1f}ms detector_rolling_p95={percentile(detector,.95):.1f}ms secondary_queues={queues} snapshots={int(snap.get('identities',0))} snapshot_mib={int(snap.get('disk_bytes',0))/1048576:.2f} db_mib={db_size/1048576:.2f} wal_mib={wal_size/1048576:.2f} active_lost_tracks={active}/{lost} global_ids={global_ids} flags={','.join(flags) or 'none'}",flush=True)
 if index+1<a.samples:time.sleep(a.interval)
