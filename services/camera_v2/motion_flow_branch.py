from __future__ import annotations

"""Continuous low-resolution optical-flow branch for Pascal-safe Camera V2.

The branch is intentionally independent of RF-DETR scheduling:

    post-mux tee -> leaky queue -> 2x3 512x432 tiler -> BGRx appsink

Each camera occupies one 256x144 tile.  Only active person-track regions run
Lucas-Kanade optical flow.  No Gst/NVMM buffer is retained after the callback.
"""

import math
import time

import cv2
import numpy as np


MOTION_COLUMNS = 2
MOTION_ROWS = 3
MOTION_TILE_W = 256
MOTION_TILE_H = 144
MOTION_W = MOTION_TILE_W * MOTION_COLUMNS
MOTION_H = MOTION_TILE_H * MOTION_ROWS


def _robust_displacement(prev_gray, curr_gray, region):
    x1, y1, x2, y2 = [float(v) for v in region]
    h, w = prev_gray.shape[:2]
    x1 = max(0, min(w - 2, int(math.floor(x1))))
    y1 = max(0, min(h - 2, int(math.floor(y1))))
    x2 = max(x1 + 2, min(w, int(math.ceil(x2))))
    y2 = max(y1 + 2, min(h, int(math.ceil(y2))))

    rw = x2 - x1
    rh = y2 - y1
    if rw < 6 or rh < 8:
        return None

    # Prefer body interior instead of bbox edges/background.  If the interior is
    # texture-poor, a second wider attempt is made below.
    def feature_points(inset: float, quality: float, corners: int):
        ix = int(rw * inset)
        iy = int(rh * inset)
        ax1 = max(0, x1 + ix)
        ay1 = max(0, y1 + iy)
        ax2 = min(w, x2 - ix)
        ay2 = min(h, y2 - iy)
        if ax2 - ax1 < 5 or ay2 - ay1 < 5:
            ax1, ay1, ax2, ay2 = x1, y1, x2, y2
        mask = np.zeros_like(prev_gray, dtype=np.uint8)
        mask[ay1:ay2, ax1:ax2] = 255
        return cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=int(corners),
            qualityLevel=float(quality),
            minDistance=3.0,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )

    p0 = feature_points(0.12, 0.015, 24)
    if p0 is None or len(p0) < 4:
        p0 = feature_points(0.02, 0.008, 32)
    if p0 is None or len(p0) < 3:
        return None

    lk = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            14,
            0.025,
        ),
    )
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk)
    if p1 is None or st1 is None:
        return None
    p0_back, st2, _err2 = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **lk)
    if p0_back is None or st2 is None:
        return None

    a = p0.reshape(-1, 2)
    b = p1.reshape(-1, 2)
    back = p0_back.reshape(-1, 2)
    valid = (st1.reshape(-1) > 0) & (st2.reshape(-1) > 0)
    if err1 is not None:
        valid &= np.asarray(err1).reshape(-1) < 28.0
    fb = np.linalg.norm(a - back, axis=1)
    valid &= fb < 1.6
    if int(valid.sum()) < 3:
        return None

    delta = b[valid] - a[valid]
    dx0 = float(np.median(delta[:, 0]))
    dy0 = float(np.median(delta[:, 1]))
    residual = np.linalg.norm(delta - np.array([dx0, dy0], dtype=np.float32), axis=1)
    mad = float(np.median(residual)) if len(residual) else 0.0
    keep = residual <= max(0.9, 2.8 * mad + 0.45)
    if int(keep.sum()) < 3:
        return None

    delta = delta[keep]
    dx = float(np.median(delta[:, 0]))
    dy = float(np.median(delta[:, 1]))

    # Reject impossible one-frame jumps.  At 20 FPS an 18px shift in a 256px
    # tile is already a very fast person motion.
    if abs(dx) > 18.0 or abs(dy) > 18.0:
        return None

    good = int(len(delta))
    total = max(1, int(len(p0)))
    fb_good = fb[valid][keep]
    fb_med = float(np.median(fb_good)) if len(fb_good) else 9.0
    count_score = min(1.0, good / 8.0)
    survival = min(1.0, good / max(3.0, total * 0.55))
    fb_score = max(0.0, 1.0 - fb_med / 1.6)
    quality = 0.45 * count_score + 0.35 * survival + 0.20 * fb_score
    return dx, dy, float(quality), good


def _track_tile(owner, cid: str, prev_gray, curr_gray, now: float) -> int:
    tracker = owner.boxes
    flow_regions = getattr(tracker, "flow_regions", None)
    apply_flow = getattr(tracker, "apply_flow", None)
    if flow_regions is None or apply_flow is None:
        return 0

    scale_x = MOTION_TILE_W / float(tracker.width)
    scale_y = MOTION_TILE_H / float(tracker.height)
    updates = 0
    for row in flow_regions(cid, now):
        sx1, sy1, sx2, sy2 = row["box"]
        region = (
            sx1 * scale_x,
            sy1 * scale_y,
            sx2 * scale_x,
            sy2 * scale_y,
        )
        result = _robust_displacement(prev_gray, curr_gray, region)
        if result is None:
            continue
        dx_tile, dy_tile, quality, _good = result
        dx_source = dx_tile / max(1e-6, scale_x)
        dy_source = dy_tile / max(1e-6, scale_y)
        if apply_flow(
            cid,
            int(row["track_id"]),
            dx_source,
            dy_source,
            now,
            quality,
        ):
            updates += 1
    return updates


def _on_motion_sample(owner, sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return owner.Gst.FlowReturn.OK

    structure = sample.get_caps().get_structure(0)
    width = int(structure.get_value("width"))
    height = int(structure.get_value("height"))
    if (width, height) != (MOTION_W, MOTION_H):
        return owner.Gst.FlowReturn.OK

    buffer = sample.get_buffer()
    ok, mapped = buffer.map(owner.Gst.MapFlags.READ)
    if not ok:
        return owner.Gst.FlowReturn.OK

    now = time.monotonic()
    current_gray = {}
    try:
        tight_stride = width * 4
        mapped_size = int(getattr(mapped, "size", len(mapped.data)))
        if mapped_size < tight_stride * height:
            return owner.Gst.FlowReturn.OK
        row_stride = (
            mapped_size // height
            if height > 0 and mapped_size % height == 0
            else tight_stride
        )
        if row_stride < tight_stride:
            return owner.Gst.FlowReturn.OK

        raw = np.frombuffer(mapped.data, dtype=np.uint8, count=row_stride * height)
        rows = raw.reshape((height, row_stride))
        bgrx = rows[:, :tight_stride].reshape((height, width, 4))

        for cid, index in owner.camera_index.items():
            index = int(index)
            row = index // MOTION_COLUMNS
            col = index % MOTION_COLUMNS
            y1 = row * MOTION_TILE_H
            y2 = y1 + MOTION_TILE_H
            x1 = col * MOTION_TILE_W
            x2 = x1 + MOTION_TILE_W
            tile = bgrx[y1:y2, x1:x2, :3]
            if tile.shape[:2] != (MOTION_TILE_H, MOTION_TILE_W):
                continue
            # cvtColor creates an owned grayscale image before GstBuffer unmap.
            current_gray[cid] = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    finally:
        buffer.unmap(mapped)

    previous = getattr(owner, "_motion_prev_gray", {})
    total_updates = 0
    for cid, gray in current_gray.items():
        prev = previous.get(cid)
        if prev is not None and prev.shape == gray.shape:
            total_updates += _track_tile(owner, cid, prev, gray, now)
    owner._motion_prev_gray = current_gray
    owner.motion_frames = int(getattr(owner, "motion_frames", 0)) + 1
    owner.motion_updates = int(getattr(owner, "motion_updates", 0)) + total_updates
    return owner.Gst.FlowReturn.OK


def attach_motion_flow(owner) -> None:
    """Attach the non-blocking continuous motion branch before PLAYING."""
    if getattr(owner, "_motion_flow_attached", False):
        return
    tee = getattr(owner, "postmux_tee", None)
    request_src = getattr(owner, "_request_src_pad", None)
    if tee is None or request_src is None:
        # Non-Pascal runtimes keep their existing tracker contract.
        return

    q = owner._make("queue", "pascal_motion_branch")
    tiler = owner._make("nvmultistreamtiler", "pascal_motion_tiler")
    convert = owner._make("nvvideoconvert", "pascal_motion_convert")
    caps = owner._make("capsfilter", "pascal_motion_caps")
    sink = owner._make("appsink", "pascal_motion_sink")

    owner._queue_latest(owner, q, 1)
    owner._set_if(tiler, "rows", MOTION_ROWS)
    owner._set_if(tiler, "columns", MOTION_COLUMNS)
    owner._set_if(tiler, "width", MOTION_W)
    owner._set_if(tiler, "height", MOTION_H)
    owner._set_if(tiler, "gpu-id", owner.gpu_id)
    owner._set_if(tiler, "nvbuf-memory-type", 2)
    owner._set_if(tiler, "compute-hw", 1)
    owner._set_if(tiler, "interpolation-method", 2)
    if tiler.find_property("show-source") is not None:
        tiler.set_property("show-source", -1)

    owner._set_if(convert, "gpu-id", owner.gpu_id)
    owner._set_if(convert, "compute-hw", 1)
    caps.set_property(
        "caps",
        owner.Gst.Caps.from_string(
            f"video/x-raw,format=BGRx,width={MOTION_W},height={MOTION_H},pixel-aspect-ratio=1/1"
        ),
    )
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)
    sink.set_property("drop", True)
    sink.set_property("max-buffers", 1)
    owner._set_if(sink, "enable-last-sample", False)
    owner._set_if(sink, "wait-on-eos", False)

    for element in (q, tiler, convert, caps, sink):
        owner.pipeline.add(element)

    tee_motion = request_src(tee, "src_%u")
    if tee_motion.link(q.get_static_pad("sink")) != owner.Gst.PadLinkReturn.OK:
        raise RuntimeError("failed mux tee -> optical-flow queue")
    if not q.link(tiler):
        raise RuntimeError("failed optical-flow queue -> tiler")
    if not tiler.link(convert):
        raise RuntimeError("failed optical-flow tiler -> convert")
    if not convert.link(caps):
        raise RuntimeError("failed optical-flow convert -> caps")
    if not caps.link(sink):
        raise RuntimeError("failed optical-flow caps -> appsink")

    callback = lambda appsink: _on_motion_sample(owner, appsink)
    sink.connect("new-sample", callback)

    owner._motion_callback = callback
    owner._motion_prev_gray = {}
    owner.motion_frames = 0
    owner.motion_updates = 0
    owner.motion_queue = q
    owner.motion_tiler = tiler
    owner.motion_convert = convert
    owner.motion_caps = caps
    owner.motion_sink = sink
    owner._motion_flow_attached = True

    print(
        "CAMERA_MOTION_FLOW mode=continuous-lk wall=512x432 tile=256x144 "
        "queue=latest1 detector_independent=1",
        flush=True,
    )
