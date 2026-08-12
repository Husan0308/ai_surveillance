"""Vectorized association features and deterministic one-to-one matching."""
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

def motion_proximity_matrix(predicted_boxes,detection_boxes,max_normalized_distance=1.5):
    """Bounded center-distance evidence relative to predicted person-box size."""
    if not len(predicted_boxes) or not len(detection_boxes):return np.zeros((len(predicted_boxes),len(detection_boxes)),np.float32)
    a=np.asarray(predicted_boxes,np.float32);b=np.asarray(detection_boxes,np.float32)
    ac=(a[:,:2]+a[:,2:])/2;bc=(b[:,:2]+b[:,2:])/2
    distance=np.linalg.norm(ac[:,None]-bc[None,:],axis=2)
    scale=np.maximum(np.linalg.norm(a[:,2:]-a[:,:2],axis=1)[:,None],1.0)
    normalized=distance/scale
    return np.maximum(0.0,1.0-normalized/max(float(max_normalized_distance),1e-6)).astype(np.float32)

def association_features(predicted_boxes, detection_boxes, track_boxes=None,
                         track_velocities=None, detection_confidences=None,
                         max_normalized_distance=1.5):
    """Return bounded geometry/motion evidence in track x detection matrices."""
    rows, columns = len(predicted_boxes), len(detection_boxes)
    shape = (rows, columns)
    names=("iou","normalized_center_distance","center_score","height_ratio","size_ratio",
           "direction_cosine","direction_score","confidence","geometry_score","geometry_gate")
    if not rows or not columns:
        empty=np.zeros(shape,np.float32)
        return {name:empty.copy() for name in names}
    predicted=np.asarray(predicted_boxes,np.float32);detections=np.asarray(detection_boxes,np.float32)
    previous=np.asarray(track_boxes if track_boxes is not None else predicted_boxes,np.float32)
    velocity=np.asarray(track_velocities if track_velocities is not None else np.zeros((rows,2)),np.float32)[:,:2]
    confidence=np.asarray(detection_confidences if detection_confidences is not None else np.ones(columns),np.float32)
    pc=(predicted[:,:2]+predicted[:,2:])*.5;dc=(detections[:,:2]+detections[:,2:])*.5
    previous_center=(previous[:,:2]+previous[:,2:])*.5
    displacement=dc[None,:,:]-previous_center[:,None,:]
    predicted_wh=np.maximum(1.0,predicted[:,2:]-predicted[:,:2]);detection_wh=np.maximum(1.0,detections[:,2:]-detections[:,:2])
    # Keep the distance definition compatible with the proven production
    # matcher: normalize by the predicted box diagonal, not sqrt(area).
    # sqrt(area) is much smaller for tall person boxes and turned ordinary
    # observation gaps into impossible jumps.
    scale=np.maximum(1.0,np.linalg.norm(predicted_wh,axis=1))[:,None]
    distance=np.linalg.norm(pc[:,None,:]-dc[None,:,:],axis=2);normalized=distance/scale
    center_score=np.maximum(0.0,1.0-normalized/max(float(max_normalized_distance),1e-6))
    height_ratio=np.minimum(predicted_wh[:,None,1],detection_wh[None,:,1])/np.maximum(predicted_wh[:,None,1],detection_wh[None,:,1])
    area_a=predicted_wh[:,0]*predicted_wh[:,1];area_b=detection_wh[:,0]*detection_wh[:,1]
    size_ratio=np.minimum(area_a[:,None],area_b[None,:])/np.maximum(area_a[:,None],area_b[None,:])
    speed=np.linalg.norm(velocity,axis=1);travel=np.linalg.norm(displacement,axis=2)
    cosine=np.divide(np.sum(velocity[:,None,:]*displacement,axis=2),speed[:,None]*travel,
                     out=np.zeros(shape,np.float32),where=(speed[:,None]>8.0)&(travel>2.0))
    direction_score=np.where(speed[:,None]>8.0,(cosine+1.0)*.5,.5).astype(np.float32)
    overlap=iou_matrix(predicted,detections)
    confidence_matrix=np.broadcast_to(np.clip(confidence,0.0,1.0)[None,:],shape)
    # Preserve the established IoU-or-motion threshold semantics. The other
    # terms are bounded ranking penalties, never independent rescue evidence.
    # A noisy direction estimate must not reject a strong real overlap.
    base=np.maximum(overlap,center_score)
    quality=.82+.08*height_ratio+.05*size_ratio+.03*direction_score+.02*confidence_matrix
    geometry=(base*quality).astype(np.float32)
    # Appearance is considered only after this physical plausibility gate.
    # Distance and scale make a candidate impossible; velocity direction is a
    # ranking/ambiguity feature because real people can stop or reverse.
    gate=((normalized<=float(max_normalized_distance))&(height_ratio>=.35)&(size_ratio>=.16)).astype(np.float32)
    return {"iou":overlap,"normalized_center_distance":normalized.astype(np.float32),"center_score":center_score.astype(np.float32),
            "height_ratio":height_ratio.astype(np.float32),"size_ratio":size_ratio.astype(np.float32),
            "direction_cosine":cosine.astype(np.float32),"direction_score":direction_score,
            "confidence":confidence_matrix.astype(np.float32),"geometry_score":geometry,"geometry_gate":gate}
