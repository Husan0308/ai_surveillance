#!/usr/bin/env python3
"""Interrupt only the dedicated CAM-01 validator's bridge network, then restore it."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def latest(path):
    rows = [line for line in path.read_text().splitlines() if line.startswith('CAM-01 STATS ')]
    if not rows:
        return {}
    return {k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', rows[-1])}


def inspect():
    item = json.loads(subprocess.check_output(['docker','inspect','ai-surveillance-cam01-validation'], text=True))[0]
    if item['Path'] != '/work/cam01-validator' or not item['State']['Running']:
        raise RuntimeError('Refusing to interrupt anything except the running CAM-01 validator')
    return item


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    log=args.out/'pipeline.log'
    deadline=time.monotonic()+45
    while time.monotonic()<deadline:
        if log.exists() and latest(log).get('output',0)>=200:break
        time.sleep(1)
    before=latest(log)
    if before.get('output',0)<200:raise RuntimeError('No healthy baseline; network not touched')
    item=inspect(); cid=item['Id']; pid=item['State']['Pid']
    if set(item['NetworkSettings']['Networks']) != {'bridge'}:
        raise RuntimeError('Expected isolated bridge network; refusing other networking')
    report={'before':before,'container_id':cid,'pid_before':pid,'outage_seconds':12}
    disconnected=False
    try:
        subprocess.run(['docker','network','disconnect','bridge',cid],check=True)
        disconnected=True; report['disconnected_at']=time.time()
        print('CAM-01 TEST network disconnected for 12 seconds',flush=True)
        time.sleep(12)
        report['during']=latest(log)
    finally:
        if disconnected:
            subprocess.run(['docker','network','connect','bridge',cid],check=True)
            report['restored_at']=time.time()
            print('CAM-01 TEST network restored',flush=True)
    deadline=time.monotonic()+65
    report['status']='BLOCKED'
    while time.monotonic()<deadline:
        after=latest(log)
        if after.get('output',0)>report['during'].get('output',0)+100 and after.get('fps',0)>=18:
            state=inspect()
            report.update(after=after,pid_after=state['State']['Pid'],recovery_seconds=time.time()-report['restored_at'])
            if state['Id']==cid and state['State']['Pid']==pid and report['during'].get('age',0)>=2:
                report['status']='PASS'
            break
        time.sleep(1)
    (args.out/'reconnect.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
