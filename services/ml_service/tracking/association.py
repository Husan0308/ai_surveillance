"""Vectorized CPU IoU and appearance matrices with greedy one-to-one matching."""
import time
import numpy as np

def iou_matrix(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)), np.float32)
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2]); br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0, br - tl); inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

def appearance_matrix(detection_embeddings, track_embeddings):
    if not len(detection_embeddings) or not len(track_embeddings): return np.empty((len(track_embeddings), len(detection_embeddings)), np.float32)
    detections = np.stack(detection_embeddings).astype(np.float32); tracks = np.stack(track_embeddings).astype(np.float32)
    detections /= np.maximum(np.linalg.norm(detections, axis=1, keepdims=True), 1e-12)
    tracks /= np.maximum(np.linalg.norm(tracks, axis=1, keepdims=True), 1e-12)
    return tracks @ detections.T

def greedy_match(scores, threshold):
    matches, used_rows, used_cols = [], set(), set()
    for flat in np.argsort(scores, axis=None)[::-1]:
        row, col = np.unravel_index(flat, scores.shape)
        if scores[row, col] < threshold: break
        if row not in used_rows and col not in used_cols:
            matches.append((int(row), int(col), float(scores[row, col])))
            used_rows.add(row); used_cols.add(col)
    return matches
