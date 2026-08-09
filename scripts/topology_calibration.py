#!/usr/bin/env python3
"""Human-assisted topology inventory; it never infers physical relationships."""
import json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from urllib.parse import urlsplit,urlunsplit
from shared.settings import ServiceSettings
from shared.config import topology_config

def mask(value):
    try:
        parts=urlsplit(value);host=parts.hostname or "";port=f":{parts.port}" if parts.port else "";auth=f"{parts.username}:***@" if parts.username else "";return urlunsplit((parts.scheme,auth+host+port,parts.path,parts.query,parts.fragment))
    except Exception:return "<invalid endpoint>"
def main():
    settings=ServiceSettings.from_env();rows=[]
    with sqlite3.connect(settings.database_path) as db:
        for cid,name,data in db.execute("select id,name,data from api_resources where resource='cameras' order by id"):
            item=json.loads(data or "{}");rows.append((cid,name or item.get("name"),mask(item.get("source") or item.get("rtsp_url") or ""),item.get("room_id")))
    print(f"Topology verified: {bool(topology_config().get('verified',False))}")
    for cid,name,endpoint,room in rows:print(f"{cid:8} {name or '-':20} room={room or 'UNASSIGNED':12} endpoint={endpoint}\n         video=http://127.0.0.1:8001/video/{cid}")
    print("Edit config/topology.yaml after physical inspection; this tool makes no automatic topology decisions.")
if __name__=="__main__":main()
