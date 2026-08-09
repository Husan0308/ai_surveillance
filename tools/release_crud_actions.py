#!/usr/bin/env python3
from __future__ import annotations
import json,sqlite3,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urllib.parse import quote,urlsplit,urlunsplit
from urllib.request import Request,urlopen
from shared.settings import ServiceSettings
ID='CAM-RB'
def call(method,path,payload=None,base='http://127.0.0.1:8000/api/v1/'):
 data=json.dumps(payload).encode() if payload is not None else None
 with urlopen(Request(base+path,data=data,headers={'Content-Type':'application/json'},method=method),timeout=3) as r:return json.loads(r.read() or b'{}')
def metrics():
 with urlopen('http://127.0.0.1:8001/metrics',timeout=3) as r:return json.loads(r.read())
def wait(test,seconds=12):
 end=time.time()+seconds
 while time.time()<end:
  if test():return True
  time.sleep(.2)
 return False
def main():
 settings=ServiceSettings.from_env()
 with sqlite3.connect(settings.database_path) as db:base=json.loads(db.execute("select data from api_resources where resource='cameras' and id='CAM-05'").fetchone()[0])
 source=base.get('source') or base.get('rtsp_url');parts=urlsplit(source);user=base.get('username');password=base.get('password')
 if user and password:source=urlunsplit((parts.scheme,f'{quote(str(user),safe="")}:{quote(str(password),safe="")}@{parts.netloc}',parts.path,parts.query,parts.fragment))
 result={}
 try:
  call('POST','cameras',{'id':ID,'name':'Release CRUD','source':source,'enabled':True,'heatmap_enabled':False,'codec':base['codec'],'latency_ms':base.get('latency_ms',50),'decoder_backend':base.get('decoder_backend','nvv4l2decoder')})
  result['reader_appeared']=wait(lambda:metrics().get('cameras',{}).get(ID,{}).get('online') is True)
  call('PATCH',f'cameras/{ID}',{'name':'Release CRUD Edited'});result['metadata_edited']=call('GET',f'cameras/{ID}').get('name')=='Release CRUD Edited'
  call('PATCH',f'cameras/{ID}',{'enabled':False});result['disabled']=wait(lambda:ID not in metrics().get('cameras',{}))
  call('PATCH',f'cameras/{ID}',{'enabled':True});result['restarted']=wait(lambda:metrics().get('cameras',{}).get(ID,{}).get('online') is True)
  call('DELETE',f'cameras/{ID}');result['deleted']=wait(lambda:ID not in metrics().get('cameras',{}))
 finally:
  try:call('DELETE',f'cameras/{ID}')
  except Exception:pass
 with sqlite3.connect(settings.database_path) as db:result['temporary_rows']=db.execute("select count(*) from api_resources where resource='cameras' and id=?",(ID,)).fetchone()[0];result['canonical_rows']=db.execute("select count(*) from api_resources where resource='cameras'").fetchone()[0]
 print(json.dumps(result,sort_keys=True));return 0 if all(result[k] for k in ('reader_appeared','metadata_edited','disabled','restarted','deleted')) and result['temporary_rows']==0 and result['canonical_rows']==6 else 1
if __name__=='__main__':raise SystemExit(main())
