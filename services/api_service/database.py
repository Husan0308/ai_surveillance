"""API-owned SQLite lifecycle and transaction boundary."""
from __future__ import annotations
import asyncio,json,sqlite3
from pathlib import Path
from shared.logging import get_logger
log=get_logger(__name__)

class SQLiteDatabase:
    def __init__(self,path):self.path=str(Path(path).expanduser().resolve());self.available=False
    def _connect(self):
        db=sqlite3.connect(self.path,timeout=5);db.row_factory=sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON");db.execute("PRAGMA busy_timeout=5000");return db
    async def connect(self):
        try:Path(self.path).parent.mkdir(parents=True,exist_ok=True);await asyncio.to_thread(self._initialize);self.available=True
        except Exception as exc:log.error("SQLite initialization failed: %s",exc);self.available=False
        return self.available
    def _initialize(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=NORMAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS api_resources(resource TEXT NOT NULL,id TEXT NOT NULL,name TEXT,data TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,PRIMARY KEY(resource,id));
            CREATE INDEX IF NOT EXISTS idx_api_resources_kind ON api_resources(resource,updated_at);
            CREATE TABLE IF NOT EXISTS enrollment_sessions(id TEXT PRIMARY KEY,person_id TEXT,status TEXT NOT NULL,data TEXT NOT NULL DEFAULT '{}',created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS heatmaps(camera_id TEXT NOT NULL,mode TEXT NOT NULL,timestamp TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(camera_id,mode));
            CREATE TABLE IF NOT EXISTS api_face_embeddings(id INTEGER PRIMARY KEY AUTOINCREMENT,person_id TEXT NOT NULL,embedding BLOB NOT NULL,quality REAL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            """)
            self._import_existing(db)
    @staticmethod
    def _table(db,name):return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(name,)).fetchone() is not None
    @staticmethod
    def _put(db,resource,item_id,name,data):db.execute("INSERT OR IGNORE INTO api_resources(resource,id,name,data) VALUES(?,?,?,?)",(resource,str(item_id),name,json.dumps(data,default=str)))
    def _import_existing(self,db):
        if db.execute("SELECT COUNT(*) FROM api_resources WHERE resource='persons'").fetchone()[0]==0 and self._table(db,"persons"):
            for row in db.execute("SELECT * FROM persons"):
                data=dict(row);data["id"]=str(data["id"]);self._put(db,"persons",data["id"],data.get("name"),data)
        if db.execute("SELECT COUNT(*) FROM api_resources WHERE resource='cameras'").fetchone()[0]==0 and self._table(db,"camera_config"):
            for row in db.execute("SELECT * FROM camera_config"):
                data=dict(row);data.update({"rtsp_url":data.get("source"),"enabled":bool(data.get("online",1))});self._put(db,"cameras",data["id"],data.get("name"),data)
        if db.execute("SELECT COUNT(*) FROM api_resources WHERE resource='events'").fetchone()[0]==0 and self._table(db,"events"):
            for row in db.execute("SELECT * FROM events"):
                data=dict(row);data.update({"id":str(data["id"]),"event_type":data.get("type"),"timestamp":data.get("time"),"acknowledged":bool(data.get("ack",0))});self._put(db,"events",data["id"],data.get("person_name"),data)
        if db.execute("SELECT COUNT(*) FROM api_resources WHERE resource='settings'").fetchone()[0]==0 and self._table(db,"settings"):
            values={}
            for row in db.execute("SELECT key,value FROM settings"):
                try:values[row["key"]]=json.loads(row["value"])
                except (TypeError,ValueError):values[row["key"]]=row["value"]
            if values:self._put(db,"settings","application","Application",values)
    async def run(self,operation):
        if not self.available:raise RuntimeError("SQLite unavailable")
        def execute():
            with self._connect() as db:return operation(db)
        return await asyncio.to_thread(execute)
    async def ping(self):
        try:return bool(await self.run(lambda db:db.execute("SELECT 1").fetchone()[0]))
        except Exception:return False
    async def close(self):self.available=False
