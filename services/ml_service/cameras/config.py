import json
from urllib.request import urlopen
from shared.config import camera_config
from shared.logging import get_logger
from shared.settings import ServiceSettings
log=get_logger(__name__)

def _normalize(item,defaults):
    item=dict(item);base=dict(defaults.get(str(item.get("id")),{}));merged={**base,**item}
    legacy=item.get("rtsp_url",item.get("source",base.get("source")))
    ai_source=item.get("ai_source") or legacy or base.get("ai_source")
    display_source=item.get("display_source") or legacy or base.get("display_source")
    # CameraReader consumes only source, which is deliberately bound to ai_source here.
    merged.update({"source":ai_source,"ai_source":ai_source,"display_source":display_source,"codec":item.get("ai_codec") or item.get("codec") or base.get("ai_codec") or base.get("codec")})
    merged["online"]=bool(item.get("online",item.get("enabled",base.get("online",False))))
    for key in ("username","password"):
        if not merged.get(key) and base.get(key):merged[key]=base[key]
    return merged

def fetch_camera_configs(api_url=None,fetcher=None) -> list[dict]:
    """Fetch canonical API/SQLite cameras, raising while that authority is unavailable."""
    bootstrap=camera_config();defaults={str(c["id"]):c for c in bootstrap.get("cameras",[])}
    endpoint=(api_url or ServiceSettings.from_env().api_url).rstrip("/")+"/api/v1/cameras"
    if fetcher is None:
        with urlopen(endpoint,timeout=2) as response:items=json.load(response)
    else:items=fetcher(endpoint)
    cameras=[_normalize(item,defaults) for item in items]
    log.info("Camera authority: API/SQLite (%d records)",len(cameras))
    return [c for c in cameras if c.get("online",False)]

def load_camera_configs(path=None,api_url=None,fetcher=None) -> list[dict]:
    """Load API/SQLite runtime cameras; YAML is bootstrap and local-secret fallback only."""
    if path is not None:raise ValueError("Camera configuration must use config/cameras.yaml")
    try:return fetch_camera_configs(api_url,fetcher)
    except Exception as exc:
        bootstrap=camera_config();defaults={str(c["id"]):c for c in bootstrap.get("cameras",[])}
        log.warning("Camera API unavailable; using YAML bootstrap for this startup: %s",exc)
        return [_normalize(item,defaults) for item in bootstrap.get("cameras",[]) if item.get("online",item.get("enabled",False))]
