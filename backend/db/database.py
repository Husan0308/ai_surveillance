import os
import json
import sqlite3
import threading
from datetime import datetime

from backend.core.logger import get_logger

log = get_logger("db.database")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Database:
    def __init__(self, config):
        print("DB init start...", flush=True)

        self.config = config

        db_path = config.get("database.sqlite_path", "data/surveillance.db")

        if not os.path.isabs(db_path):
            db_path = os.path.join(BASE_DIR, db_path)

        print("DB path:", db_path, flush=True)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.path = db_path
        self.lock = threading.Lock()

        print("DB connect...", flush=True)

        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        print("DB pragma...", flush=True)

        with self.lock:
            self.conn.execute("PRAGMA busy_timeout = 5000")

        print("DB schema...", flush=True)

        self.init_schema()

        print("DB init OK", flush=True)

    # ================= schema =================
    def init_schema(self):
        with self.lock:
            cur = self.conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    department TEXT,
                    employee_id TEXT UNIQUE,
                    status TEXT DEFAULT 'Active',
                    avatar BLOB,
                    created_at TEXT,
                    updated_at TEXT,
                    last_seen TEXT,
                    rec_count INTEGER DEFAULT 0,
                    stay_total INTEGER DEFAULT 0
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    embedding BLOB NOT NULL,
                    image BLOB,
                    quality REAL DEFAULT 0,
                    created_at TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id TEXT,
                    camera_id TEXT,
                    image BLOB,
                    embedding BLOB,
                    first_seen TEXT,
                    last_seen TEXT,
                    count INTEGER DEFAULT 1,
                    converted_person_id INTEGER
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    camera_id TEXT,
                    person_id INTEGER,
                    person_name TEXT,
                    type TEXT NOT NULL,
                    level TEXT DEFAULT 'info',
                    confidence REAL DEFAULT 0,
                    snapshot_path TEXT,
                    ack INTEGER DEFAULT 0,
                    extra TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    camera_id TEXT,
                    track_id TEXT,
                    entered_at TEXT,
                    left_at TEXT,
                    duration_sec INTEGER DEFAULT 0
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS analytics_hourly (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    hour INTEGER,
                    camera_id TEXT,
                    occupancy_sum INTEGER DEFAULT 0,
                    known_count INTEGER DEFAULT 0,
                    unknown_count INTEGER DEFAULT 0,
                    detection_count INTEGER DEFAULT 0,
                    recognition_count INTEGER DEFAULT 0,
                    UNIQUE(date, hour, camera_id)
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS camera_config (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    location TEXT,
                    source TEXT,
                    username TEXT,
                    password TEXT,
                    online INTEGER DEFAULT 0,
                    ai_enabled INTEGER DEFAULT 1,
                    heatmap_enabled INTEGER DEFAULT 0,
                    recording_enabled INTEGER DEFAULT 0,
                    zone_enabled INTEGER DEFAULT 0,
                    overstay_enabled INTEGER DEFAULT 0,
                    resolution TEXT DEFAULT '1920x1080',
                    fps INTEGER DEFAULT 25,
                    reconnect_interval INTEGER DEFAULT 10,
                    connection_timeout INTEGER DEFAULT 5,
                    latency_warn_ms INTEGER DEFAULT 200,
                    packet_loss_warn_percent REAL DEFAULT 2.0,
                    updated_at TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON events(time)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_visits_person ON visits(person_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_unknown_last_seen ON unknown_faces(last_seen)")

            self.conn.commit()

    # ================= helpers =================
    def _now(self):
        return datetime.now().isoformat()

    # ================= persons =================
    def save_person(self, person: dict):
        print(f"[DB] save_person called: {person.get('name')}", flush=True)
        try:
            with self.lock:
                cur = self.conn.cursor()
                now = self._now()

                person_id = person.get("id")

                if person_id:
                    print(f"[DB] UPDATE person id={person_id}", flush=True)
                    cur.execute(
                        """
                        UPDATE persons
                        SET name=?, department=?, employee_id=?, status=?, avatar=?,
                            updated_at=?, last_seen=?, rec_count=?, stay_total=?
                        WHERE id=?
                        """,
                        (
                            person.get("name"),
                            person.get("department"),
                            person.get("employee_id"),
                            person.get("status", "Active"),
                            person.get("avatar"),
                            now,
                            person.get("last_seen"),
                            int(person.get("rec_count", 0)),
                            int(person.get("stay_total", 0)),
                            person_id,
                        ),
                    )
                    self.conn.commit()
                    print(f"[DB] UPDATE OK: id={person_id}", flush=True)
                    return person_id

                print(f"[DB] INSERT new person", flush=True)
                cur.execute(
                    """
                    INSERT INTO persons(
                        name, department, employee_id, status, avatar,
                        created_at, updated_at, last_seen, rec_count, stay_total
                    )
                    VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        person.get("name"),
                        person.get("department"),
                        person.get("employee_id"),
                        person.get("status", "Active"),
                        person.get("avatar"),
                        now,
                        now,
                        person.get("last_seen"),
                        int(person.get("rec_count", 0)),
                        int(person.get("stay_total", 0)),
                    ),
                )
                self.conn.commit()
                new_id = cur.lastrowid
                print(f"[DB] INSERT OK: new_id={new_id}", flush=True)
                return new_id

        except Exception as e:
            print(f"[DB] save_person ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
        
    def get_persons(self):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id,name,department,employee_id,status,avatar,
                       created_at,updated_at,last_seen,rec_count,stay_total
                FROM persons ORDER BY name
                """
            ).fetchall()

        return [dict(r) for r in rows]

    def get_person(self, person_id: int):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT id,name,department,employee_id,status,avatar,
                       created_at,updated_at,last_seen,rec_count,stay_total
                FROM persons WHERE id=?
                """,
                (person_id,),
            ).fetchone()

        return dict(row) if row else None

    def delete_person(self, person_id: int):
        with self.lock:
            self.conn.execute("DELETE FROM face_embeddings WHERE person_id=?", (person_id,))
            self.conn.execute("DELETE FROM visits WHERE person_id=?", (person_id,))
            self.conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
            self.conn.commit()

    def update_person_last_seen(self, person_id: int, rec_count_increment: int = 1):
        with self.lock:
            self.conn.execute(
                """
                UPDATE persons
                SET last_seen=?, rec_count = rec_count + ?
                WHERE id=?
                """,
                (self._now(), rec_count_increment, person_id),
            )
            self.conn.commit()

    # ================= embeddings =================
    def add_embedding(self, person_id: int, embedding: bytes, image: bytes = None, quality: float = 0.0):
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO face_embeddings(person_id, embedding, image, quality, created_at)
                VALUES(?,?,?,?,?)
                """,
                (person_id, embedding, image, float(quality), self._now()),
            )
            self.conn.commit()

    def get_embeddings(self):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT e.id, e.person_id, p.name, e.embedding, e.quality
                FROM face_embeddings e
                JOIN persons p ON p.id = e.person_id
                ORDER BY e.id DESC
                """
            ).fetchall()

        return [dict(r) for r in rows]

    def get_embeddings_by_person(self, person_id: int):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, person_id, embedding, quality
                FROM face_embeddings
                WHERE person_id=?
                ORDER BY id DESC
                """,
                (person_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    def delete_embeddings(self, person_id: int):
        with self.lock:
            self.conn.execute("DELETE FROM face_embeddings WHERE person_id=?", (person_id,))
            self.conn.commit()

    # ================= unknown faces =================
    def add_unknown_face(self, data: dict):
        with self.lock:
            cur = self.conn.cursor()
            now = self._now()

            track_id = data.get("track_id")
            camera_id = data.get("camera_id")

            row = cur.execute(
                """
                SELECT id, count FROM unknown_faces
                WHERE track_id=? AND camera_id=? AND converted_person_id IS NULL
                ORDER BY id DESC LIMIT 1
                """,
                (track_id, camera_id),
            ).fetchone()

            if row:
                cur.execute(
                    """
                    UPDATE unknown_faces
                    SET count=count+1, last_seen=?, image=?, embedding=?
                    WHERE id=?
                    """,
                    (now, data.get("image"), data.get("embedding"), row["id"]),
                )
                self.conn.commit()
                return row["id"]

            cur.execute(
                """
                INSERT INTO unknown_faces(
                    track_id, camera_id, image, embedding,
                    first_seen, last_seen, count
                )
                VALUES(?,?,?,?,?,?,1)
                """,
                (
                    track_id,
                    camera_id,
                    data.get("image"),
                    data.get("embedding"),
                    now,
                    now,
                ),
            )

            self.conn.commit()
            return cur.lastrowid

    def get_unknown_faces(self, limit: int = 200):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, track_id, camera_id, image, embedding,
                       first_seen, last_seen, count, converted_person_id
                FROM unknown_faces
                WHERE converted_person_id IS NULL
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_unknown_face(self, unknown_id: int):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT id, track_id, camera_id, image, embedding,
                       first_seen, last_seen, count, converted_person_id
                FROM unknown_faces WHERE id=?
                """,
                (unknown_id,),
            ).fetchone()

        return dict(row) if row else None

    def convert_unknown_to_person(self, unknown_id: int, person_id: int):
        with self.lock:
            self.conn.execute(
                "UPDATE unknown_faces SET converted_person_id=? WHERE id=?",
                (person_id, unknown_id),
            )
            self.conn.commit()

    def delete_unknown_face(self, unknown_id: int):
        with self.lock:
            self.conn.execute("DELETE FROM unknown_faces WHERE id=?", (unknown_id,))
            self.conn.commit()

    # ================= events =================
    def add_event(self, event: dict):
        with self.lock:
            extra = event.get("extra")

            if extra is not None and not isinstance(extra, str):
                extra = json.dumps(extra, ensure_ascii=False)

            self.conn.execute(
                """
                INSERT INTO events(
                    time, camera_id, person_id, person_name,
                    type, level, confidence, snapshot_path, ack, extra
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event.get("time", self._now()),
                    event.get("camera_id"),
                    event.get("person_id"),
                    event.get("person_name"),
                    event.get("type"),
                    event.get("level", "info"),
                    float(event.get("confidence", 0.0)),
                    event.get("snapshot_path"),
                    1 if event.get("ack") else 0,
                    extra,
                ),
            )
            self.conn.commit()

    def get_events(self, limit: int = 300, event_type: str = None):
        query = """
            SELECT id,time,camera_id,person_id,person_name,type,level,
                   confidence,snapshot_path,ack,extra
            FROM events
        """
        params = []

        if event_type:
            query += " WHERE type=?"
            params.append(event_type)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self.lock:
            rows = self.conn.execute(query, params).fetchall()

        out = []

        for r in rows:
            d = dict(r)

            if d.get("extra"):
                try:
                    d["extra"] = json.loads(d["extra"])
                except Exception:
                    pass

            out.append(d)

        return out

    def get_events_by_date(self, date_str: str, limit: int = 300):
        """Get events for specific date. date_str: YYYY-MM-DD"""
        try:
            cursor = self.conn.execute(
                "SELECT * FROM events WHERE date(time) = ? ORDER BY time DESC LIMIT ?",
                (date_str, limit)
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[DB] get_events_by_date error: {e}")
            return []

    def get_event_dates(self):
        """Get all unique dates that have events (newest first)."""
        try:
            cursor = self.conn.execute(
                "SELECT DISTINCT date(time) as d FROM events WHERE d IS NOT NULL ORDER BY d DESC LIMIT 90"
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[DB] get_event_dates error: {e}")
            return []


    def ack_event(self, event_id: int):
        with self.lock:
            self.conn.execute("UPDATE events SET ack=1 WHERE id=?", (event_id,))
            self.conn.commit()

    def ack_event_by_key(self, time_val, camera_id: str, person_name: str):
        if hasattr(time_val, "isoformat"):
            time_val = time_val.isoformat()

        with self.lock:
            self.conn.execute(
                """
                UPDATE events SET ack=1
                WHERE time=? AND camera_id=? AND person_name=?
                """,
                (time_val, camera_id, person_name),
            )
            self.conn.commit()

    def delete_old_events(self, retention_days: int):
        with self.lock:
            self.conn.execute(
                "DELETE FROM events WHERE datetime(time) < datetime('now', ?)",
                (f"-{int(retention_days)} days",),
            )
            self.conn.commit()

    # ================= visits =================
    def start_visit(self, person_id: int, camera_id: str):
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO visits(person_id, camera_id, entered_at, duration_sec)
                VALUES(?,?,?,0)
                """,
                (person_id, camera_id, self._now()),
            )
            self.conn.commit()
            return cur.lastrowid

    def open_visit(self, person_id: int, camera_id: str, track_id: str):
        with self.lock:
            cur = self.conn.execute(
                """
                INSERT INTO visits(person_id, camera_id, track_id, entered_at, duration_sec)
                VALUES(?,?,?,?,0)
                """,
                (person_id, camera_id, str(track_id), self._now()),
            )
            self.conn.commit()
            return cur.lastrowid

    def close_visit(self, visit_id: int):
        with self.lock:
            row = self.conn.execute(
                "SELECT entered_at FROM visits WHERE id=?",
                (visit_id,),
            ).fetchone()

            if not row:
                return

            try:
                entered = datetime.fromisoformat(row["entered_at"])
                duration = int((datetime.now() - entered).total_seconds())
            except Exception:
                duration = 0

            self.conn.execute(
                "UPDATE visits SET left_at=?, duration_sec=? WHERE id=?",
                (self._now(), duration, visit_id),
            )
            self.conn.commit()

    def close_visit_by_track(self, camera_id: str, track_id: str):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT id, entered_at FROM visits
                WHERE camera_id=? AND track_id=? AND left_at IS NULL
                ORDER BY id DESC LIMIT 1
                """,
                (camera_id, str(track_id)),
            ).fetchone()

            if not row:
                return None, 0

            try:
                entered = datetime.fromisoformat(row["entered_at"])
                duration = int((datetime.now() - entered).total_seconds())
            except Exception:
                duration = 0

            self.conn.execute(
                "UPDATE visits SET left_at=?, duration_sec=? WHERE id=?",
                (self._now(), duration, row["id"]),
            )
            self.conn.commit()

            return row["id"], duration

    def get_visits(self, person_id: int = None, limit: int = 200):
        query = """
            SELECT v.id, v.person_id, p.name, v.camera_id,
                   v.entered_at, v.left_at, v.duration_sec
            FROM visits v
            LEFT JOIN persons p ON p.id = v.person_id
        """
        params = []

        if person_id is not None:
            query += " WHERE v.person_id=?"
            params.append(person_id)

        query += " ORDER BY v.id DESC LIMIT ?"
        params.append(limit)

        with self.lock:
            rows = self.conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    # ================= analytics =================
    def upsert_analytics_hourly(self, data: dict):
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO analytics_hourly(
                    date, hour, camera_id,
                    occupancy_sum, known_count, unknown_count,
                    detection_count, recognition_count
                )
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(date, hour, camera_id)
                DO UPDATE SET
                    occupancy_sum = occupancy_sum + excluded.occupancy_sum,
                    known_count = known_count + excluded.known_count,
                    unknown_count = unknown_count + excluded.unknown_count,
                    detection_count = detection_count + excluded.detection_count,
                    recognition_count = recognition_count + excluded.recognition_count
                """,
                (
                    data.get("date"),
                    int(data.get("hour", 0)),
                    data.get("camera_id"),
                    int(data.get("occupancy_sum", 0)),
                    int(data.get("known_count", 0)),
                    int(data.get("unknown_count", 0)),
                    int(data.get("detection_count", 0)),
                    int(data.get("recognition_count", 0)),
                ),
            )
            self.conn.commit()

    def get_analytics_hourly(self, date_str: str = None):
        query = "SELECT * FROM analytics_hourly"
        params = []

        if date_str:
            query += " WHERE date=?"
            params.append(date_str)

        query += " ORDER BY hour"

        with self.lock:
            rows = self.conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def get_peak_hour(self, date_str: str):
        peak = [0] * 24

        with self.lock:
            rows = self.conn.execute(
                """
                SELECT hour, SUM(occupancy_sum) occ
                FROM analytics_hourly
                WHERE date=?
                GROUP BY hour
                """,
                (date_str,),
            ).fetchall()

        for r in rows:
            h = r["hour"]

            if h is not None and 0 <= int(h) < 24:
                peak[int(h)] = int(r["occ"] or 0)

        return peak

    def get_avg_stay_today(self):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT AVG(duration_sec) avg_stay
                FROM visits
                WHERE date(entered_at) = date('now')
                  AND duration_sec > 0
                """
            ).fetchone()

        if row and row["avg_stay"] is not None:
            return float(row["avg_stay"])

        return 0.0

    def get_detection_recognition_today(self):
        with self.lock:
            row = self.conn.execute(
                """
                SELECT
                    SUM(detection_count) detections,
                    SUM(recognition_count) recognitions
                FROM analytics_hourly
                WHERE date = date('now')
                """
            ).fetchone()

        if not row:
            return 0, 0

        return int(row["detections"] or 0), int(row["recognitions"] or 0)

    # ================= history =================
    def get_recognition_history(self, person_id: int, limit: int = 200):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, time, camera_id, person_id, person_name,
                       type, level, confidence, snapshot_path
                FROM events
                WHERE person_id=? AND type='person_recognized'
                ORDER BY id DESC
                LIMIT ?
                """,
                (person_id, limit),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_events_by_person(self, person_id: int, limit: int = 200):
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, time, camera_id, person_id, person_name,
                       type, level, confidence, snapshot_path, ack, extra
                FROM events
                WHERE person_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (person_id, limit),
            ).fetchall()

        return [dict(r) for r in rows]

    def get_person_timeline(self, person_id: int):
        timeline = [0] * 24

        with self.lock:
            rows = self.conn.execute(
                """
                SELECT CAST(substr(time,12,2) AS INTEGER) h, COUNT(*) c
                FROM events
                WHERE person_id=? AND type='person_recognized'
                GROUP BY h
                """,
                (person_id,),
            ).fetchall()

        for r in rows:
            h = r["h"]

            if h is not None and 0 <= int(h) < 24:
                timeline[int(h)] = int(r["c"])

        return timeline

    # ================= camera config =================
    def save_camera_config(self, cam: dict):
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO camera_config(
                    id, name, location, source, username, password,
                    online, ai_enabled, heatmap_enabled, recording_enabled,
                    zone_enabled, overstay_enabled, resolution, fps,
                    reconnect_interval, connection_timeout,
                    latency_warn_ms, packet_loss_warn_percent, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    location=excluded.location,
                    source=excluded.source,
                    username=excluded.username,
                    password=excluded.password,
                    online=excluded.online,
                    ai_enabled=excluded.ai_enabled,
                    heatmap_enabled=excluded.heatmap_enabled,
                    recording_enabled=excluded.recording_enabled,
                    zone_enabled=excluded.zone_enabled,
                    overstay_enabled=excluded.overstay_enabled,
                    resolution=excluded.resolution,
                    fps=excluded.fps,
                    reconnect_interval=excluded.reconnect_interval,
                    connection_timeout=excluded.connection_timeout,
                    latency_warn_ms=excluded.latency_warn_ms,
                    packet_loss_warn_percent=excluded.packet_loss_warn_percent,
                    updated_at=excluded.updated_at
                """,
                (
                    cam.get("id"),
                    cam.get("name"),
                    cam.get("location"),
                    cam.get("source"),
                    cam.get("username"),
                    cam.get("password"),
                    1 if cam.get("online") else 0,
                    1 if cam.get("ai_enabled", True) else 0,
                    1 if cam.get("heatmap_enabled", False) else 0,
                    1 if cam.get("recording_enabled", False) else 0,
                    1 if cam.get("zone_enabled", False) else 0,
                    1 if cam.get("overstay_enabled", False) else 0,
                    cam.get("resolution", "1920x1080"),
                    int(cam.get("fps", 25)),
                    int(cam.get("reconnect_interval", 10)),
                    int(cam.get("connection_timeout", 5)),
                    int(cam.get("latency_warn_ms", 200)),
                    float(cam.get("packet_loss_warn_percent", 2.0)),
                    self._now(),
                ),
            )
            self.conn.commit()

    def get_camera_configs(self):
        with self.lock:
            rows = self.conn.execute("SELECT * FROM camera_config ORDER BY id").fetchall()

        out = []

        for r in rows:
            d = dict(r)

            for key in (
                "online",
                "ai_enabled",
                "heatmap_enabled",
                "recording_enabled",
                "zone_enabled",
                "overstay_enabled",
            ):
                d[key] = bool(d.get(key))

            out.append(d)

        return out

    def delete_camera_config(self, camera_id: str):
        with self.lock:
            self.conn.execute("DELETE FROM camera_config WHERE id=?", (camera_id,))
            self.conn.commit()

    # ================= settings =================
    def save_setting(self, key: str, value):
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO settings(key, value, updated_at)
                VALUES(?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), self._now()),
            )
            self.conn.commit()

    def get_settings(self):
        with self.lock:
            rows = self.conn.execute("SELECT key, value FROM settings").fetchall()

        out = {}

        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]

        return out

    def close(self):
        with self.lock:
            self.conn.close()