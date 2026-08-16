from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .app import app, core_cfg, face
from .face_service import _normalize


class FaceEnrollFilesRequest(BaseModel):
    name: str
    department: str = ""
    employee_id: str = ""
    images_jpeg_b64: list[str] = Field(default_factory=list)
    profile_index: int = 0


@app.post("/faces/enrollment/files")
def face_enrollment_files(request: FaceEnrollFilesRequest):
    """Enroll exactly ten user-selected images into the real InsightFace gallery."""
    if face is None:
        raise HTTPException(503, "face recognition is disabled")

    target = int(getattr(face, "enrollment_target", 10) or 10)
    if len(request.images_jpeg_b64) != target:
        raise HTTPException(400, f"exactly {target} images are required")
    if request.profile_index < 0 or request.profile_index >= target:
        raise HTTPException(400, "profile_index is out of range")

    with face._lock:
        if not face._ready:
            raise HTTPException(503, face._last_error or "face engine not ready")

    samples: list[dict] = []
    qualities: list[float] = []
    for index, encoded in enumerate(request.images_jpeg_b64):
        try:
            raw = base64.b64decode(str(encoded), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(400, f"image {index + 1} is not valid base64") from exc
        if not raw or len(raw) > 8 * 1024 * 1024:
            raise HTTPException(400, f"image {index + 1} is empty or too large")
        array = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise HTTPException(400, f"image {index + 1} could not be decoded")
        if image.shape[0] > 2048 or image.shape[1] > 2048:
            scale = min(2048.0 / image.shape[0], 2048.0 / image.shape[1])
            image = cv2.resize(
                image,
                (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        sample = face._best_face(image, enrollment=True)
        if sample is None:
            raise HTTPException(
                409,
                f"image {index + 1}: no enrollment-quality face found; use a clear single-face image",
            )
        samples.append(sample)
        qualities.append(float(sample.get("quality") or 0.0))

    embeddings = [_normalize(sample.get("embedding")) for sample in samples]
    if any(embedding is None for embedding in embeddings):
        raise HTTPException(409, "one or more enrollment embeddings are invalid")
    matrix = np.stack(embeddings, axis=0)
    centroid = _normalize(np.mean(matrix, axis=0))
    if centroid is None:
        raise HTTPException(409, "could not build enrollment face centroid")
    similarities = matrix @ centroid
    consistency_threshold = float(
        getattr(face, "enrollment_consistency_similarity", 0.35)
    )
    max_outliers = int(getattr(face, "enrollment_max_outliers", 1))
    outliers = int(np.sum(similarities < consistency_threshold))
    if outliers > max_outliers:
        raise HTTPException(
            409,
            "the ten images do not look like one consistent person; select ten images of the same face",
        )

    selected = samples[request.profile_index]
    ordered = [selected] + [
        sample for i, sample in enumerate(samples) if i != request.profile_index
    ]
    try:
        person = face.gallery.enroll(
            request.name,
            request.department,
            request.employee_id,
            ordered,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        **person,
        "accepted_samples": target,
        "profile_index": int(request.profile_index),
        "quality_min": round(min(qualities), 4),
        "quality_avg": round(sum(qualities) / len(qualities), 4),
        "consistency_min": round(float(np.min(similarities)), 4),
    }
