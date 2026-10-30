#!/usr/bin/env python3
"""Private, single-run 13-hour supervisor for the existing production monitor."""
import fcntl, hashlib, json, os, signal, socket, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; RUNTIME=ROOT/'.runtime/six-camera-soak'
LIMIT=46800; RESERVE=300; SAMPLE=60; MAX_BYTES=16*1024*1024
VIEWER=ROOT/'.runtime/deepstream-rtsp-pause-suppression-reporting-20261030-r1/sources/apps/sample_apps/deepstream-app/deepstream-app'
VIEWER_HASH='2602a28a191e3b8671e5e9e4d4fda1fc98d428b70b2d4235bfea8684c269483e'
MONITOR=ROOT/'scripts/six_camera_live/monitor.sh'; MONITOR_HASH='fab0c7339ae627cd84cc6aa1b44e0b8919661767d04cfb72d5eafa2d551f0585'
LOCK=RUNTIME/'active.lock'; STATE=RUNTIME/'active.json'
def now(): return time.time_ns()//1000
def append(path,row):
    data=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
    if path.stat().st_size+len(data)>MAX_BYTES: raise OSError('report reached 16 MiB safety limit')
    with path.open('ab',buffering=0) as f: f.write(data); os.fsync(f.fileno())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def health(url):
    try:
        with urllib.request.urlopen(url,timeout=3) as r: return {'http':r.status,'body':json.load(r)}
    except Exception as e: return {'error':type(e).__name__}
def safe_run(args,timeout=5):
    try:return subprocess.run(args,check=False,capture_output=True,text=True,timeout=timeout).stdout.strip()
    except Exception:return None
def gpu():
    s=safe_run(['nvidia-smi','--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total','--format=csv,noheader,nounits'])
    try:
        a=[float(x.strip()) for x in s.splitlines()[0].split(',')]
        return dict(zip(('utilization_percent','temperature_c','vram_used_mib','vram_total_mib'),a))
    except Exception:return None
def write_state(d):
    tmp=d/'active.json.tmp'; tmp.write_text(json.dumps(d)); os.chmod(tmp,0o600); os.replace(tmp,STATE)
def session_ok():
    disp=os.environ.get('DISPLAY'); xa=os.environ.get('XAUTHORITY')
    if not disp or not xa or not Path(xa).is_file(): return False
    return subprocess.run(['xdpyinfo','-display',disp],capture_output=True,timeout=4).returncode==0
def preflight():
    if digest(VIEWER)!=VIEWER_HASH or digest(MONITOR)!=MONITOR_HASH: raise RuntimeError('production hash mismatch')
    if not session_ok(): raise RuntimeError('authenticated X11 unavailable; export DISPLAY and XAUTHORITY')
    if health('http://127.0.0.1:8100/health').get('http')!=200 or health('http://127.0.0.1:8001/health').get('http')!=200: raise RuntimeError('API/ML health prerequisite failed')
    if safe_run(['docker','ps','-a','--format','{{.Names}}']) is None: raise RuntimeError('Docker inspection failed')
    names=safe_run(['docker','ps','-a','--format','{{.Names}}']).splitlines()
    if any(n.startswith('ai-surveillance-six-live-') for n in names): raise RuntimeError('existing or orphaned owned container; inspect before start')
    return {'viewer_sha256':VIEWER_HASH,'monitor_sha256':MONITOR_HASH}
def child_alive(pid,start):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return int(stat[19])==start
    except Exception:return False
def proc_start(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def stop_owned(d,reason):
    d['stop_reason']=reason; write_state(d)
    # Existing launcher stop validates its own lock PID/start ticks and performs bounded graceful cleanup.
    rc=subprocess.run(['bash',str(MONITOR),'stop'],cwd=ROOT,timeout=35).returncode
    d['monitor_stop_command_exit']=rc
    return rc
def supervise(d):
    report=Path(d['report']); start_m=time.monotonic(); deadline=start_m+LIMIT-RESERVE; last=0; hourly=0; api_fail=ml_fail=0
    # Launch existing production monitor foreground under this detached supervisor; one owner only.
    env=os.environ.copy(); child=subprocess.Popen(['bash',str(MONITOR),'start'],cwd=ROOT,env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    d.update(supervisor_pid=os.getpid(),monitor_pid=child.pid,monitor_start_ticks=proc_start(child.pid),deadline_epoch=time.time()+LIMIT-RESERVE); write_state(d)
    append(report,{'type':'start','utc':datetime.now(timezone.utc).isoformat(),'limit_seconds':LIMIT,'shutdown_reserve_seconds':RESERVE,'production':d['hashes'],'pid':child.pid})
    stop_reason='automatic_deadline'; exit_code=None
    while time.monotonic()<deadline:
        rc=child.poll()
        if rc is not None: exit_code=rc; stop_reason='monitor_exited'; break
        if not session_ok(): stop_reason='x11_session_lost'; break
        if time.monotonic()-last>=SAMPLE:
            api=health('http://127.0.0.1:8100/health'); ml=health('http://127.0.0.1:8001/health')
            api_fail+=api.get('http')!=200; ml_fail+=ml.get('http')!=200
            body=ml.get('body') or {}; cameras=body.get('cameras',[])
            append(report,{'type':'health','utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.monotonic()-start_m,1),'api':api,'ml':{'http':ml.get('http'),'error':ml.get('error'),'monitoring_status':body.get('monitoring_status'),'analytics_status':body.get('analytics_status'),'six_camera_readiness':body.get('six_camera_readiness'),'cameras':cameras},'gpu':gpu(),'host':host_stats(),'monitor_pid':child.pid,'monitor_running':True})
            last=time.monotonic()
            sec=int(last-start_m)//3600
            if sec>hourly:
                hourly=sec; append(report,{'type':'hourly_summary','elapsed_seconds':int(last-start_m),'api_fail_samples':api_fail,'ml_fail_samples':ml_fail})
        time.sleep(min(1,max(0,deadline-time.monotonic())))
    stoprc=stop_owned(d,stop_reason)
    try: child.wait(timeout=RESERVE-10)
    except subprocess.TimeoutExpired: stoprc=stoprc or 1
    if exit_code is None: exit_code=child.returncode
    cleanup={'launcher_stop_exit':stoprc,'monitor_exit':exit_code,'monitor_lock_present':(ROOT/'.runtime/six-live-preview.lock').exists(),'owned_containers': [n for n in (safe_run(['docker','ps','-a','--format','{{.Names}}']) or '').splitlines() if n.startswith('ai-surveillance-six-live-')]}
    append(report,{'type':'final','utc':datetime.now(timezone.utc).isoformat(),'stop_reason':stop_reason,'monitor_exit':exit_code,'cleanup':cleanup,'status':'clean' if stoprc==0 and exit_code in (0,None) and not cleanup['monitor_lock_present'] and not cleanup['owned_containers'] else 'failed'})
    d.update(state='stopped' if cleanup['status']=='clean' else 'failed',stop_reason=stop_reason,exit_code=exit_code,finished_epoch=time.time()); write_state(d)
    return 0 if cleanup['status']=='clean' else 1
def host_stats():
    try:
        vals={k:int(v) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)}
        load=[float(x) for x in Path('/proc/loadavg').read_text().split()[:3]]
        return {'ram_total_mib':vals.get('MemTotal',0)/1024,'ram_available_mib':vals.get('MemAvailable',0)/1024,'loadavg':load}
    except Exception:return None

def command():
    if len(sys.argv)!=2 or sys.argv[1] not in ('start','stop','status','report'): raise SystemExit('usage: soak_13h.sh {start|stop|status|report}')
    cmd=sys.argv[1]; RUNTIME.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(RUNTIME,0o700)
    if cmd=='start':
        lockfd=os.open(LOCK,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC,0o600)
        try: fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('soak already active')
        hashes=preflight(); run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid()); rd=RUNTIME/run; rd.mkdir(mode=0o700)
        d={'run_id':run,'run_dir':str(rd),'report':str(rd/'monitor_13h.jsonl'),'hashes':hashes,'state':'starting','started_epoch':time.time()}; write_state(d)
        pid=os.fork()
        if pid: print(d['report']); return
        os.setsid(); os.chdir(ROOT); os.umask(0o077)
        with open(os.devnull,'rb') as i, open(rd/'supervisor.log','ab',buffering=0) as o:
            os.dup2(i.fileno(),0); os.dup2(o.fileno(),1); os.dup2(o.fileno(),2)
        try: code=supervise(d)
        except BaseException as e:
            try: append(Path(d['report']),{'type':'supervisor_failure','utc':datetime.now(timezone.utc).isoformat(),'error':type(e).__name__})
            except Exception: pass
            try: stop_owned(d,'supervisor_failure')
            except Exception: pass
            d.update(state='failed'); write_state(d); code=1
        os._exit(code)
    state=json.loads(STATE.read_text()) if STATE.exists() else None
    if cmd=='status':
        if not state: print('stopped (no run)'); return
        elapsed=time.time()-state['started_epoch']; rem=max(0,state.get('deadline_epoch',state['started_epoch']+LIMIT)-time.time())
        latest=[]
        try:
            for line in Path(state['report']).read_text().splitlines():
                x=json.loads(line)
                if x.get('type')=='health':latest=x['ml']
        except Exception:pass
        print(json.dumps({'state':state.get('state'),'elapsed_seconds':int(elapsed),'remaining_seconds':int(rem),'report':state['report'],'latest_health':latest}))
    elif cmd=='report':
        if not state: print('no report'); return
        p=Path(state['report']); records=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        print(json.dumps({'report':str(p),'records':len(records),'last_record':records[-1] if records else None},indent=2))
    else:
        if not state or state.get('state') not in ('running','starting'): raise SystemExit('no active soak')
        if not child_alive(state.get('supervisor_pid',-1),state.get('supervisor_start_ticks',-2)): raise SystemExit('supervisor identity unavailable; inspect report and owned launcher')
        state['manual_stop_requested']=True; write_state(state)
        subprocess.run(['bash',str(MONITOR),'stop'],cwd=ROOT,timeout=35,check=False)
        print('graceful stop requested through owned monitor lock')

def main(): command()
if __name__=='__main__':main()
