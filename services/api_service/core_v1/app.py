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
    return [{'id':c.get('id'),'name':c.get('name'),'location':c.get('location'),'online':c.get('online',True)} for c in config.get('cameras',[])]
