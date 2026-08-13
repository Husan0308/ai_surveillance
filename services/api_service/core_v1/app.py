from __future__ import annotations
import json,urllib.request
from pathlib import Path
import yaml
from fastapi import FastAPI

ROOT=Path(__file__).resolve().parents[3]
core_cfg=(yaml.safe_load((ROOT/'config/core_v1.yaml').read_text()) or {}).get('core_v1',{})
ml_url=f"http://127.0.0.1:{int(core_cfg.get('ml_port',8001))}"
app=FastAPI(title='AI Surveillance API Core v1',version='1.0')

@app.get('/health')
def health():
    ml=None
    try:
        with urllib.request.urlopen(ml_url+'/health',timeout=.75) as response:ml=json.loads(response.read().decode())
    except Exception as exc:ml={'status':'unavailable','error':str(exc)}
    return {'status':'ok','ml':ml}

@app.get('/api/v1/cameras')
def cameras():
    config=yaml.safe_load((ROOT/'config/cameras.yaml').read_text()) or {}
    live={}
    try:
        with urllib.request.urlopen(ml_url+'/health',timeout=.75) as response:
            live=(json.loads(response.read().decode()).get('cameras') or {})
    except Exception:
        pass
    return [
        {
            'id':camera.get('id'),
            'name':camera.get('name'),
            'location':camera.get('location'),
            'enabled':bool(camera.get('online',True)),
            'online':bool((live.get(str(camera.get('id'))) or {}).get('online',False)),
            'source_fps':float((live.get(str(camera.get('id'))) or {}).get('source_fps',0.0) or 0.0),
            'last_frame_age_ms':(live.get(str(camera.get('id'))) or {}).get('last_frame_age_ms'),
            'last_error':str((live.get(str(camera.get('id'))) or {}).get('last_error') or ''),
        }
        for camera in config.get('cameras',[])
    ]
