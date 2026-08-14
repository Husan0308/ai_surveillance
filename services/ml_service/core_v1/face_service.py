from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

import cv2
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vector) -> np.ndarray | None:
    if vector is None:
        return None
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    norm = float(np.linalg.norm(array))
    if norm <= 1e-9:
        return None
    return array / norm


def cosine_similarity(left, right) -> float:
    a = _normalize(left)
    b = _normalize(right)
    if a is None or b is None or a.size != b.size:
        return -1.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _safe_person_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip()).strip("-_")
    return cleaned[:64]


@dataclass(slots=True)
class GalleryMatch:
    person_id: str
    name: str
    department: str
    similarity: float
    second_best_similarity: float
    margin: float


class FaceGallery:
    """Small persistent multi-prototype face gallery.

    Embeddings are normalized before storage. Recognition uses the mean of the
    best two prototypes for a person (or the single prototype when only one is
    present), then applies both an absolute threshold and a second-best margin.
    """

    def __init__(self, root: Path, config: dict):
        self.root = Path(root)
        self.data_dir = self.root / str(config.get("data_dir", "data/faces"))
        self.db_path = self.root / str(config.get("db_path", "data/face_db.json"))
        self.match_similarity = float(config.get("match_similarity", 0.52))
        self.strong_similarity = float(config.get("strong_similarity", 0.68))
        self.second_best_margin = float(config.get("second_best_margin", 0.06))
        self._lock = threading.RLock()
        self._people: dict[str, dict] = {}
        self._load()

    def _load(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            return
        try:
            payload = json.loads(self.db_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for person in payload.get("people") or []:
            person_id = str(person.get("person_id") or "").strip()
            if not person_id:
                continue
            embeddings = []
            for value in person.get("embeddings") or []:
                embedding = _normalize(value)
                if embedding is not None:
                    embeddings.append(embedding)
            if not embeddings:
                continue
            self._people[person_id] = {
                "person_id": person_id,
                "name": str(person.get("name") or person_id),
                "department": str(person.get("department") or ""),
                "employee_id": str(person.get("employee_id") or person_id),
                "created_at": person.get("created_at") or _utc_now(),
                "last_seen": person.get("last_seen"),
                "recognitions": int(person.get("recognitions") or 0),
                "avatar": str(person.get("avatar") or ""),
                "embeddings": embeddings,
            }

    def _save_locked(self):
        payload = {"version": 1, "people": []}
        for person in sorted(self._people.values(), key=lambda item: item["name"].lower()):
            payload["people"].append(
                {
                    key: value
                    for key, value in person.items()
                    if key != "embeddings"
                }
                | {
                    "embeddings": [
                        [round(float(x), 7) for x in embedding.tolist()]
                        for embedding in person["embeddings"]
                    ]
                }
            )
        temporary = self.db_path.with_suffix(self.db_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.db_path)

    def list_people(self) -> list[dict]:
        with self._lock:
            rows = []
            for person in sorted(self._people.values(), key=lambda item: item["name"].lower()):
                rows.append(
                    {
                        "person_id": person["person_id"],
                        "name": person["name"],
                        "department": person["department"],
                        "employee_id": person["employee_id"],
                        "created_at": person["created_at"],
                        "last_seen": person["last_seen"],
                        "recognitions": person["recognitions"],
                        "samples": len(person["embeddings"]),
                        "has_avatar": bool(person["avatar"]),
                    }
                )
            return rows

    def match(self, embedding) -> GalleryMatch | None:
        query = _normalize(embedding)
        if query is None:
            return None
        with self._lock:
            candidates = []
            for person in self._people.values():
                sims = sorted(
                    (float(np.dot(query, prototype)) for prototype in person["embeddings"]),
                    reverse=True,
                )
                if not sims:
                    continue
                top = sims[: min(2, len(sims))]
                score = float(sum(top) / len(top))
                candidates.append((score, person))
            if not candidates:
                return None
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best = candidates[0]
            second = candidates[1][0] if len(candidates) > 1 else -1.0
            margin = best_score - second
            if best_score < self.match_similarity:
                return None
            if best_score < self.strong_similarity and margin < self.second_best_margin:
                return None
            return GalleryMatch(
                best["person_id"],
                best["name"],
                best["department"],
                best_score,
                second,
                margin,
            )

    def enroll(self, name: str, department: str, employee_id: str, samples: list[dict]) -> dict:
        clean_name = str(name or "").strip()
        if not clean_name:
            raise ValueError("name is required")
        requested_id = _safe_person_id(employee_id)
        with self._lock:
            if not requested_id:
                base = _safe_person_id(clean_name) or "person"
                requested_id = base
                counter = 1
                while requested_id in self._people:
                    counter += 1
                    requested_id = f"{base}-{counter}"
            if requested_id in self._people:
                raise ValueError(f"person_id already exists: {requested_id}")

            embeddings = []
            for sample in samples:
                embedding = _normalize(sample.get("embedding"))
                if embedding is not None:
                    embeddings.append(embedding)
            if not embeddings:
                raise ValueError("no valid face embeddings")

            person_dir = self.data_dir / requested_id
            person_dir.mkdir(parents=True, exist_ok=True)
            avatar_rel = ""
            for index, sample in enumerate(samples, start=1):
                jpeg = sample.get("jpeg")
                if not jpeg:
                    continue
                path = person_dir / f"sample_{index:02d}.jpg"
                path.write_bytes(jpeg)
                if not avatar_rel:
                    avatar = person_dir / "avatar.jpg"
                    avatar.write_bytes(jpeg)
                    avatar_rel = str(avatar.relative_to(self.root))

            person = {
                "person_id": requested_id,
                "name": clean_name,
                "department": str(department or ""),
                "employee_id": str(employee_id or requested_id),
                "created_at": _utc_now(),
                "last_seen": None,
                "recognitions": 0,
                "avatar": avatar_rel,
                "embeddings": embeddings,
            }
            self._people[requested_id] = person
            self._save_locked()
            return self.person(requested_id)

    def person(self, person_id: str) -> dict | None:
        with self._lock:
            person = self._people.get(str(person_id))
            if person is None:
                return None
            return {
                "person_id": person["person_id"],
                "name": person["name"],
                "department": person["department"],
                "employee_id": person["employee_id"],
                "created_at": person["created_at"],
                "last_seen": person["last_seen"],
                "recognitions": person["recognitions"],
                "samples": len(person["embeddings"]),
                "has_avatar": bool(person["avatar"]),
            }

    def avatar(self, person_id: str) -> bytes | None:
        with self._lock:
            person = self._people.get(str(person_id))
            if person is None or not person.get("avatar"):
                return None
            path = self.root / person["avatar"]
        try:
            return path.read_bytes()
        except Exception:
            return None

    def note_recognition(self, person_id: str):
        with self._lock:
            person = self._people.get(str(person_id))
            if person is None:
                return
            person["recognitions"] += 1
            person["last_seen"] = _utc_now()
            self._save_locked()

    def delete(self, person_id: str) -> bool:
        with self._lock:
            person = self._people.pop(str(person_id), None)
            if person is None:
                return False
            self._save_locked()
        shutil.rmtree(self.data_dir / str(person_id), ignore_errors=True)
        return True


class FaceRecognitionService:
    """CPU-only face side-path that decorates ReID/local identities with names."""

    def __init__(self, stores: dict, publishers: dict, config: dict, root: Path, base_identity=None):
        self.stores = stores
        self.publishers = publishers
        self.config = dict(config or {})
        self.root = Path(root)
        self.base_identity = base_identity
        self.gallery = FaceGallery(self.root, self.config)

        self.enabled = bool(self.config.get("enabled", False))
        self.model_pack = str(self.config.get("model_pack", "buffalo_l"))
        self.model_root = str(self.root / str(self.config.get("model_root", "models/insightface")))
        self.det_size = int(self.config.get("det_size", 320))
        self.det_thresh = float(self.config.get("det_thresh", 0.55))
        self.poll_interval = max(0.05, float(self.config.get("poll_interval_ms", 180)) / 1000.0)
        self.sample_interval = max(0.25, float(self.config.get("sample_interval_ms", 1100)) / 1000.0)
        self.max_people = max(1, int(self.config.get("max_people_per_camera", 3)))
        self.upper_body_ratio = min(0.80, max(0.35, float(self.config.get("upper_body_ratio", 0.60))))
        self.horizontal_pad = min(0.40, max(0.0, float(self.config.get("horizontal_pad_ratio", 0.16))))
        self.min_face_size = max(12, int(self.config.get("min_face_size_px", 28)))
        self.min_blur = max(0.0, float(self.config.get("min_blur_variance", 18.0)))
        self.confirm_hits = max(1, int(self.config.get("confirm_hits", 2)))
        self.confirm_window = max(0.5, float(self.config.get("confirm_window_sec", 4.0)))
        self.binding_ttl = max(2.0, float(self.config.get("local_binding_ttl_sec", 45.0)))
        self.global_binding_ttl = max(self.binding_ttl, float(self.config.get("global_binding_ttl_sec", 21600.0)))
        self.enrollment_target = max(3, int(self.config.get("enrollment_samples", 10)))
        self.enrollment_min_quality = float(self.config.get("enrollment_min_quality", 0.56))
        self.enrollment_token_ttl = max(30.0, float(self.config.get("enrollment_token_ttl_sec", 600.0)))

        self._app = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._ready = False
        self._last_error = ""
        self._last_sample: dict[tuple[str, int], float] = {}
        self._pending: dict[str, dict] = {}
        self._local_bindings: dict[tuple[str, int], dict] = {}
        self._global_bindings: dict[str, dict] = {}
        self._enrollment_tokens: dict[str, dict] = {}
        self._rr_index = 0
        self._inferences = 0
        self._faces_detected = 0
        self._matches = 0
        self._confirmed = 0
        self._quality_rejects = 0
        self._ambiguous_or_unknown = 0
        self._last_inference_ms = 0.0
        self._total_inference_ms = 0.0

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="core-v1-face-cpu", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self, timeout=5):
        if self._thread:
            self._thread.join(timeout)

    def _load_engine(self):
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name=self.model_pack,
                root=self.model_root,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=(self.det_size, self.det_size))
            with self._lock:
                self._app = app
                self._ready = True
                self._last_error = ""
        except Exception as exc:
            with self._lock:
                self._app = None
                self._ready = False
                self._last_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _track_id(row: dict) -> int:
        try:
            return int(row.get("track_id") or 0)
        except (TypeError, ValueError):
            return 0

    def _base(self, camera_id: str, track_id: int) -> dict:
        if self.base_identity is not None:
            try:
                value = self.base_identity.identity_for_track(camera_id, int(track_id))
            except Exception:
                value = None
            if value:
                return dict(value)
        publisher = self.publishers.get(camera_id)
        label = None
        if publisher is not None:
            try:
                label = publisher.visual_tracker.display_label(track_id)
            except Exception:
                pass
        return {"global_id": label or f"{camera_id}:{track_id}", "known": False}

    def _identity_key(self, camera_id: str, track_id: int) -> tuple[str, str]:
        base = self._base(camera_id, track_id)
        gid = str(base.get("global_id") or "").strip()
        if gid:
            return "global", gid
        return "local", f"{camera_id}:{track_id}"

    def _person_crop(self, image: np.ndarray, bbox) -> np.ndarray | None:
        if image is None or image.size == 0 or bbox is None or len(bbox) < 4:
            return None
        h, w = image.shape[:2]
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            return None
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        x1 -= bw * self.horizontal_pad
        x2 += bw * self.horizontal_pad
        y1 -= bh * 0.04
        y2 = y1 + bh * self.upper_body_ratio
        ix1 = max(0, min(w - 1, int(math.floor(x1))))
        iy1 = max(0, min(h - 1, int(math.floor(y1))))
        ix2 = max(ix1 + 1, min(w, int(math.ceil(x2))))
        iy2 = max(iy1 + 1, min(h, int(math.ceil(y2))))
        if ix2 - ix1 < 16 or iy2 - iy1 < 16:
            return None
        return image[iy1:iy2, ix1:ix2].copy()

    def _best_face(self, crop: np.ndarray, *, enrollment=False) -> dict | None:
        with self._lock:
            app = self._app
        if app is None or crop is None or crop.size == 0:
            return None
        started = time.perf_counter()
        faces = app.get(crop)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._inferences += 1
            self._last_inference_ms = elapsed_ms
            self._total_inference_ms += elapsed_ms
            self._faces_detected += len(faces)
        if not faces:
            return None

        candidates = []
        for face in faces:
            bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
            if bbox.size < 4:
                continue
            fw = max(0.0, float(bbox[2] - bbox[0]))
            fh = max(0.0, float(bbox[3] - bbox[1]))
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if min(fw, fh) < self.min_face_size or score < self.det_thresh:
                continue
            x1 = max(0, min(crop.shape[1] - 1, int(bbox[0])))
            y1 = max(0, min(crop.shape[0] - 1, int(bbox[1])))
            x2 = max(x1 + 1, min(crop.shape[1], int(math.ceil(bbox[2]))))
            y2 = max(y1 + 1, min(crop.shape[0], int(math.ceil(bbox[3]))))
            face_crop = crop[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < self.min_blur:
                with self._lock:
                    self._quality_rejects += 1
                continue
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            embedding = _normalize(embedding)
            if embedding is None:
                continue
            size_ratio = min(1.0, min(fw, fh) / max(1.0, float(self.min_face_size * 2)))
            blur_quality = min(1.0, blur / max(1.0, self.min_blur * 4.0))
            quality = 0.50 * min(1.0, score) + 0.28 * size_ratio + 0.22 * blur_quality
            if enrollment and quality < self.enrollment_min_quality:
                with self._lock:
                    self._quality_rejects += 1
                continue
            candidates.append((quality, fw * fh, score, blur, embedding, face_crop))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        quality, _area, score, blur, embedding, face_crop = candidates[0]
        ok, encoded = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {
            "quality": float(quality),
            "det_score": float(score),
            "blur": float(blur),
            "embedding": embedding,
            "jpeg": encoded.tobytes() if ok else b"",
            "face_width": int(face_crop.shape[1]),
            "face_height": int(face_crop.shape[0]),
        }

    def _binding_for(self, camera_id: str, track_id: int) -> dict | None:
        now = time.monotonic()
        base = self._base(camera_id, track_id)
        gid = str(base.get("global_id") or "").strip()
        with self._lock:
            if gid:
                binding = self._global_bindings.get(gid)
                if binding and now - binding["last_seen_mono"] <= self.global_binding_ttl:
                    return dict(binding)
            binding = self._local_bindings.get((camera_id, int(track_id)))
            if binding and now - binding["last_seen_mono"] <= self.binding_ttl:
                return dict(binding)
        return None

    def identity_for_track(self, camera_id: str, track_id: int) -> dict:
        base = self._base(camera_id, int(track_id))
        binding = self._binding_for(camera_id, int(track_id))
        if not binding:
            return base
        payload = dict(base)
        payload.update(
            {
                "known": True,
                "name": binding["name"],
                "person_id": binding["person_id"],
                "known_id": binding["person_id"],
                "department": binding.get("department", ""),
                "face_similarity": binding.get("similarity"),
                "face_reason": binding.get("reason", "face_confirmed"),
            }
        )
        return payload

    def _confirm_match(self, camera_id: str, track_id: int, match: GalleryMatch):
        now = time.monotonic()
        key_type, key_value = self._identity_key(camera_id, track_id)
        pending_key = f"{key_type}:{key_value}"
        strong = match.similarity >= self.gallery.strong_similarity
        with self._lock:
            pending = self._pending.get(pending_key)
            if strong:
                hits = self.confirm_hits
            elif (
                pending
                and pending["person_id"] == match.person_id
                and now - pending["last_seen_mono"] <= self.confirm_window
            ):
                hits = int(pending["hits"]) + 1
            else:
                hits = 1
            self._pending[pending_key] = {
                "person_id": match.person_id,
                "hits": hits,
                "last_seen_mono": now,
                "similarity": match.similarity,
            }
            if hits < self.confirm_hits:
                return
            binding = {
                "person_id": match.person_id,
                "name": match.name,
                "department": match.department,
                "similarity": match.similarity,
                "last_seen_mono": now,
                "reason": "face_strong" if strong else "face_confirmed",
            }
            if key_type == "global":
                existing = self._global_bindings.get(key_value)
                is_new = existing is None or existing.get("person_id") != match.person_id
                self._global_bindings[key_value] = binding
            else:
                local_key = (camera_id, int(track_id))
                existing = self._local_bindings.get(local_key)
                is_new = existing is None or existing.get("person_id") != match.person_id
                self._local_bindings[local_key] = binding
            self._confirmed += 1
            self._pending.pop(pending_key, None)
        if is_new:
            self.gallery.note_recognition(match.person_id)

    def _process_track(self, camera_id: str, frame, row: dict):
        track_id = self._track_id(row)
        if track_id <= 0:
            return
        now = time.monotonic()
        sample_key = (camera_id, track_id)
        with self._lock:
            if now - self._last_sample.get(sample_key, 0.0) < self.sample_interval:
                return
            self._last_sample[sample_key] = now
        crop = self._person_crop(frame.image, row.get("bbox"))
        sample = self._best_face(crop)
        if sample is None:
            return
        match = self.gallery.match(sample["embedding"])
        if match is None:
            with self._lock:
                self._ambiguous_or_unknown += 1
            return
        with self._lock:
            self._matches += 1
        self._confirm_match(camera_id, track_id, match)

    def _run(self):
        self._load_engine()
        with self._lock:
            ready = self._ready
        if not ready:
            return
        camera_ids = sorted(self.stores)
        while not self._stop.is_set():
            if not camera_ids:
                self._stop.wait(self.poll_interval)
                continue
            camera_id = camera_ids[self._rr_index % len(camera_ids)]
            self._rr_index += 1
            store = self.stores.get(camera_id)
            publisher = self.publishers.get(camera_id)
            if store is not None and publisher is not None:
                frame, _version = store.get()
                if frame is not None:
                    rows = sorted(
                        publisher.track_snapshot(),
                        key=lambda row: float(row.get("confidence") or 0.0),
                        reverse=True,
                    )[: self.max_people]
                    for row in rows:
                        if self._stop.is_set():
                            break
                        self._process_track(camera_id, frame, row)
            self._cleanup()
            self._stop.wait(self.poll_interval)

    def _cleanup(self):
        now = time.monotonic()
        with self._lock:
            self._pending = {
                key: value
                for key, value in self._pending.items()
                if now - value["last_seen_mono"] <= self.confirm_window
            }
            self._local_bindings = {
                key: value
                for key, value in self._local_bindings.items()
                if now - value["last_seen_mono"] <= self.binding_ttl
            }
            self._global_bindings = {
                key: value
                for key, value in self._global_bindings.items()
                if now - value["last_seen_mono"] <= self.global_binding_ttl
            }
            expired = [
                token
                for token, value in self._enrollment_tokens.items()
                if now - value["created_mono"] > self.enrollment_token_ttl
            ]
            for token in expired:
                self._enrollment_tokens.pop(token, None)

    def capture_enrollment_sample(self, camera_id: str, track_id: int) -> dict:
        with self._lock:
            if not self._ready:
                raise RuntimeError(self._last_error or "face engine not ready")
        store = self.stores.get(camera_id)
        publisher = self.publishers.get(camera_id)
        if store is None or publisher is None:
            raise KeyError("camera not found")
        frame, _version = store.get()
        if frame is None:
            raise RuntimeError("camera frame not ready")
        row = next(
            (item for item in publisher.track_snapshot() if self._track_id(item) == int(track_id)),
            None,
        )
        if row is None:
            raise RuntimeError("selected track is not currently visible")
        crop = self._person_crop(frame.image, row.get("bbox"))
        sample = self._best_face(crop, enrollment=True)
        if sample is None:
            raise RuntimeError("no enrollment-quality face found; face the camera and move closer")

        with self._lock:
            recent = [
                value
                for value in self._enrollment_tokens.values()
                if value.get("camera_id") == camera_id and value.get("track_id") == int(track_id)
            ]
        if recent:
            best_duplicate = max(
                cosine_similarity(sample["embedding"], value["embedding"])
                for value in recent[-4:]
            )
            if best_duplicate > 0.9985:
                raise RuntimeError("sample is too similar to the previous frame; change head angle slightly")

        token = uuid.uuid4().hex
        with self._lock:
            self._enrollment_tokens[token] = {
                **sample,
                "created_mono": time.monotonic(),
                "camera_id": camera_id,
                "track_id": int(track_id),
            }
        return {
            "token": token,
            "quality": round(sample["quality"], 4),
            "det_score": round(sample["det_score"], 4),
            "blur": round(sample["blur"], 2),
            "face_size": [sample["face_width"], sample["face_height"]],
            "thumbnail_jpeg_b64": base64.b64encode(sample["jpeg"]).decode("ascii"),
        }

    def commit_enrollment(self, name: str, department: str, employee_id: str, tokens: list[str]) -> dict:
        unique_tokens = list(dict.fromkeys(str(token) for token in tokens if token))
        if len(unique_tokens) < self.enrollment_target:
            raise ValueError(f"{self.enrollment_target} accepted samples are required")
        now = time.monotonic()
        with self._lock:
            samples = []
            for token in unique_tokens[: self.enrollment_target]:
                value = self._enrollment_tokens.get(token)
                if value is None or now - value["created_mono"] > self.enrollment_token_ttl:
                    raise ValueError("an enrollment sample expired; capture again")
                samples.append(dict(value))
        person = self.gallery.enroll(name, department, employee_id, samples)
        with self._lock:
            for token in unique_tokens:
                self._enrollment_tokens.pop(token, None)
        return person

    def delete_person(self, person_id: str) -> bool:
        deleted = self.gallery.delete(person_id)
        if deleted:
            with self._lock:
                self._local_bindings = {
                    key: value
                    for key, value in self._local_bindings.items()
                    if value.get("person_id") != person_id
                }
                self._global_bindings = {
                    key: value
                    for key, value in self._global_bindings.items()
                    if value.get("person_id") != person_id
                }
        return deleted

    def metrics(self) -> dict:
        with self._lock:
            average = self._total_inference_ms / self._inferences if self._inferences else 0.0
            return {
                "enabled": self.enabled,
                "ready": self._ready,
                "provider": "CPUExecutionProvider",
                "model_pack": self.model_pack,
                "det_size": [self.det_size, self.det_size],
                "last_error": self._last_error,
                "people": len(self.gallery.list_people()),
                "inferences": self._inferences,
                "faces_detected": self._faces_detected,
                "matches": self._matches,
                "confirmed_bindings": self._confirmed,
                "quality_rejects": self._quality_rejects,
                "unknown_or_ambiguous": self._ambiguous_or_unknown,
                "active_local_bindings": len(self._local_bindings),
                "active_global_bindings": len(self._global_bindings),
                "pending_confirmations": len(self._pending),
                "enrollment_tokens": len(self._enrollment_tokens),
                "last_inference_ms": self._last_inference_ms,
                "average_inference_ms": average,
            }

    def snapshot(self) -> dict:
        with self._lock:
            bindings = []
            for gid, value in self._global_bindings.items():
                bindings.append({"scope": "global", "identity": gid, **{k: v for k, v in value.items() if k != "last_seen_mono"}})
            for (camera_id, track_id), value in self._local_bindings.items():
                bindings.append({"scope": "local", "camera_id": camera_id, "track_id": track_id, **{k: v for k, v in value.items() if k != "last_seen_mono"}})
        return {
            "metrics": self.metrics(),
            "people": self.gallery.list_people(),
            "bindings": bindings,
        }

    def avatar(self, person_id: str) -> bytes | None:
        return self.gallery.avatar(person_id)
