#!/usr/bin/env python3
"""Temporary ML-ready camera lifecycle release check; always removes its DB row."""
from __future__ import annotations
import json,sqlite3,subprocess,sys,time
from pathlib import Path
from urllib.parse import quote,urlsplit,urlunsplit
from urllib.request import Request,urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.settings import ServiceSettings
ID='CAM-RB'
def request(method,path,payload=None,required=True):
 try:
  raw=json.dumps(payload).encode() if payload is not None else None
  with urlopen(Request('http://127.0.0.1:8000/api/v1/'+path,data=raw,headers={'Content-Type':'application/json'},method=method),timeout=2) as response:return json.loads(response.read() or b'{}')
 except Exception:
  if required:raise
  return None
def ml(path):
 try:
  with urlopen('http://127.0.0.1:8001/'+path,timeout=2) as response:return json.loads(response.read())
 except Exception:return None
def wait_for(fn,timeout=20):
 end=time.time()+timeout
 while time.time()<end:
  value=fn()
  if value:return value
  time.sleep(.2)
 return None
def threads(pid):
 result=[]
 for path in Path(f'/proc/{pid}/task').glob('*/comm'):
  try:result.append(path.read_text().strip())
  except OSError:pass
 return result
def main():
 settings=ServiceSettings.from_env();python=str(ROOT/'venv/bin/python');api=subprocess.Popen([python,'-u','-m','services.api_service.app'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);worker=None
 try:
  if not wait_for(lambda:request('GET','ready',required=False),10):raise RuntimeError('API not ready')
  worker=subprocess.Popen([python,'-u','-m','services.ml_service.app'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  ready=wait_for(lambda:(x if (x:=ml('ready')) and x.get('ready') else None),25)
  if not ready:raise RuntimeError('ML not ready')
  with sqlite3.connect(settings.database_path) as db:
   raw=db.execute("select data from api_resources where resource='cameras' and id='CAM-05'").fetchone()[0]
  base=json.loads(raw);source=base.get('source') or base.get('rtsp_url');parsed=urlsplit(source);user=base.get('username');password=base.get('password')
  if user and password:source=urlunsplit((parsed.scheme,f'{quote(str(user),safe="")}:{quote(str(password),safe="")}@{parsed.netloc}',parsed.path,parsed.query,parsed.fragment))
  payload={'id':ID,'name':'Release CRUD','source':source,'enabled':True,'heatmap_enabled':False}
  request('POST','cameras',payload)
  online=wait_for(lambda:(m if (m:=ml('metrics')).get('cameras',{}).get(ID,{}).get('online') else None),12)
  appeared=bool(online);thread_appeared=any(ID[:10] in name for name in threads(worker.pid))
  request('PATCH',f'cameras/{ID}',{'name':'Release CRUD Edited'});edited=request('GET',f'cameras/{ID}').get('name')=='Release CRUD Edited'
  request('PATCH',f'cameras/{ID}',{'enabled':False});disabled=bool(wait_for(lambda:ID not in (ml('metrics') or {}).get('cameras',{}),8));thread_stopped=not any(ID[:10] in name for name in threads(worker.pid))
  request('PATCH',f'cameras/{ID}',{'enabled':True});restarted=bool(wait_for(lambda:(m if (m:=ml('metrics')).get('cameras',{}).get(ID,{}).get('online') else None),12))
  request('DELETE',f'cameras/{ID}');deleted=bool(wait_for(lambda:ID not in (ml('metrics') or {}).get('cameras',{}),8));thread_deleted=not any(ID[:10] in name for name in threads(worker.pid))
  with sqlite3.connect(settings.database_path) as db:row_count=db.execute("select count(*) from api_resources where resource='cameras' and id=?",(ID,)).fetchone()[0];camera_count=db.execute("select count(*) from api_resources where resource='cameras'").fetchone()[0]
  result={'ml_ready':True,'reader_appeared':appeared,'thread_appeared':thread_appeared,'metadata_edited':edited,'disabled':disabled,'thread_stopped':thread_stopped,'restarted':restarted,'deleted':deleted,'thread_deleted':thread_deleted,'temporary_rows':row_count,'canonical_rows':camera_count};print(json.dumps(result,sort_keys=True));return 0 if all(v for k,v in result.items() if k not in ('temporary_rows','canonical_rows')) and row_count==0 and camera_count==6 else 1
 finally:
  cleanup=request('DELETE',f'cameras/{ID}',required=False)
  if cleanup is None:print('release check cleanup: temporary camera already absent or API unavailable',file=sys.stderr)
  for process in (worker,api):
   if process and process.poll() is None:process.terminate()
  for process in (worker,api):
   if process:
    try:process.wait(10)
    except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':raise SystemExit(main())
