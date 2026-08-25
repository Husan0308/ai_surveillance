from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


SRC_W = 1280
SRC_H = 720
NET_W = 672
NET_H = 384

THRESHOLD = 0.32
PERSON_ID = 1
MAX_FRAMES = 400

ENGINE = Path(
    "artifacts/rfdetr_trt86/"
    "rfdetr-small_b1_384x672_fp32.engine"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def check(ret, name):
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"{name}: CUDA error {int(err)}")


def resize_half_pixel(rgb):
    # RF-DETR / torchvision resize(..., antialias=False) convention.
    src = rgb.astype(np.float32) / 255.0

    ys = (
        (np.arange(NET_H, dtype=np.float32) + 0.5)
        * (SRC_H / NET_H)
        - 0.5
    )
    xs = (
        (np.arange(NET_W, dtype=np.float32) + 0.5)
        * (SRC_W / NET_W)
        - 0.5
    )

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = ys - y0
    wx = xs - x0

    y0 = np.clip(y0, 0, SRC_H - 1)
    y1 = np.clip(y1, 0, SRC_H - 1)
    x0 = np.clip(x0, 0, SRC_W - 1)
    x1 = np.clip(x1, 0, SRC_W - 1)

    a = src[y0[:, None], x0[None, :]]
    b = src[y0[:, None], x1[None, :]]
    c = src[y1[:, None], x0[None, :]]
    d = src[y1[:, None], x1[None, :]]

    wx = wx[None, :, None]
    wy = wy[:, None, None]

    top = a + (b - a) * wx
    bot = c + (d - c) * wx
    out = top + (bot - top) * wy

    out = (out - MEAN) / STD
    out = out.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(out, dtype=np.float32)


def decode_persons(dets, labels):
    logits = labels[0, :, :-1]

    scores_all = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -88.0, 88.0))
    )

    flat = scores_all.reshape(-1)
    k = min(300, flat.size)

    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]

    scores = flat[idx]
    idx = idx[scores >= THRESHOLD]
    scores = scores[scores >= THRESHOLD]

    classes = scores_all.shape[1]
    qidx = idx // classes
    class_ids = idx % classes

    rows = []

    for score, q, cid in zip(scores, qidx, class_ids):
        if int(cid) != PERSON_ID:
            continue

        cx, cy, w, h = dets[0, q]

        x1 = (cx - w / 2.0) * SRC_W
        y1 = (cy - h / 2.0) * SRC_H
        x2 = (cx + w / 2.0) * SRC_W
        y2 = (cy + h / 2.0) * SRC_H

        chair = float(scores_all[q, 62]) if scores_all.shape[1] > 62 else 0.0
        couch = float(scores_all[q, 63]) if scores_all.shape[1] > 63 else 0.0
        table = float(scores_all[q, 67]) if scores_all.shape[1] > 67 else 0.0

        non_person = scores_all[q].copy()
        if non_person.shape[0] > PERSON_ID:
            non_person[PERSON_ID] = -1.0

        best_other_id = int(np.argmax(non_person))
        best_other = float(non_person[best_other_id])

        print(
            f"PERSON_RAW q={int(q)}"
            f" p={float(score):.3f}"
            f" chair={chair:.3f}"
            f" couch={couch:.3f}"
            f" table={table:.3f}"
            f" other_id={best_other_id}"
            f" other={best_other:.3f}"
            f" box=({float(x1):.0f},{float(y1):.0f},"
            f"{float(x2):.0f},{float(y2):.0f})",
            file=sys.stderr,
            flush=True,
        )

        rows.append([
            float(x1), float(y1),
            float(x2), float(y2),
            float(score),
        ])

    # Simple high-IoU dedupe.
    rows.sort(key=lambda r: r[4], reverse=True)
    kept = []

    for row in rows:
        duplicate = False

        for old in kept:
            ax1, ay1, ax2, ay2 = row[:4]
            bx1, by1, bx2, by2 = old[:4]

            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter = iw * ih

            aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
            bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)

            iou = inter / max(1e-9, aa + bb - inter)

            if iou >= 0.62:
                duplicate = True
                break

        if not duplicate:
            kept.append(row)

    return kept


def draw_box(img, row):
    x1, y1, x2, y2, _ = row

    x1 = max(0, min(SRC_W - 1, int(round(x1))))
    y1 = max(0, min(SRC_H - 1, int(round(y1))))
    x2 = max(0, min(SRC_W - 1, int(round(x2))))
    y2 = max(0, min(SRC_H - 1, int(round(y2))))

    if x2 <= x1 or y2 <= y1:
        return

    t = 3
    color = np.array([255, 196, 64], dtype=np.uint8)

    img[y1:min(y1+t, SRC_H), x1:x2+1] = color
    img[max(0, y2-t+1):y2+1, x1:x2+1] = color
    img[y1:y2+1, x1:min(x1+t, SRC_W)] = color
    img[y1:y2+1, max(0, x2-t+1):x2+1] = color


logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())

if engine is None:
    raise SystemExit("TRT_ENGINE_LOAD=FAIL")

context = engine.create_execution_context()

bindings = [0] * engine.num_bindings
ptrs = []
outputs = {}

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = tuple(context.get_binding_shape(i))
    dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
    nbytes = int(np.prod(shape)) * dtype.itemsize

    err, ptr = cudart.cudaMalloc(nbytes)
    check((err,), f"cudaMalloc {name}")

    bindings[i] = int(ptr)
    ptrs.append(ptr)

    if not engine.binding_is_input(i):
        outputs[name] = np.empty(shape, dtype=dtype)

err, stream = cudart.cudaStreamCreate()
check((err,), "cudaStreamCreate")

input_index = engine.get_binding_index("input")

frame_bytes = SRC_W * SRC_H * 3

times = []
frame_no = 0
started = time.perf_counter()

while frame_no < MAX_FRAMES:
    raw = sys.stdin.buffer.read(frame_bytes)

    if len(raw) != frame_bytes:
        break

    rgb = np.frombuffer(
        raw,
        dtype=np.uint8,
    ).reshape(SRC_H, SRC_W, 3).copy()

    t0 = time.perf_counter()
    inp = resize_half_pixel(rgb)
    t1 = time.perf_counter()

    check(
        cudart.cudaMemcpyAsync(
            ptrs[input_index],
            inp.ctypes.data,
            inp.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        ),
        "H2D",
    )

    infer0 = time.perf_counter()

    ok = context.execute_async_v2(
        bindings=bindings,
        stream_handle=int(stream),
    )

    if not ok:
        raise RuntimeError("TRT execute failed")

    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            continue

        name = engine.get_binding_name(i)
        arr = outputs[name]

        check(
            cudart.cudaMemcpyAsync(
                arr.ctypes.data,
                ptrs[i],
                arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            f"D2H {name}",
        )

    check(
        cudart.cudaStreamSynchronize(stream),
        "cudaStreamSynchronize",
    )

    infer1 = time.perf_counter()

    persons = decode_persons(
        outputs["dets"],
        outputs["labels"],
    )

    for row in persons:
        draw_box(rgb, row)

    sys.stdout.buffer.write(rgb.tobytes())
    sys.stdout.buffer.flush()

    total_ms = (time.perf_counter() - t0) * 1000.0
    prep_ms = (t1 - t0) * 1000.0
    infer_ms = (infer1 - infer0) * 1000.0

    times.append(total_ms)
    frame_no += 1

    if frame_no % 20 == 0:
        elapsed = time.perf_counter() - started
        fps = frame_no / max(elapsed, 1e-6)

        print(
            f"TRT_LIVE frame={frame_no}"
            f" persons={len(persons)}"
            f" prep_ms={prep_ms:.1f}"
            f" infer_ms={infer_ms:.1f}"
            f" total_ms={total_ms:.1f}"
            f" fps={fps:.2f}",
            file=sys.stderr,
            flush=True,
        )

cudart.cudaStreamDestroy(stream)

for ptr in ptrs:
    cudart.cudaFree(ptr)

if times:
    a = np.asarray(times, dtype=np.float32)
    print(
        "TRT_LIVE_RESULT"
        f" frames={frame_no}"
        f" mean_ms={a.mean():.2f}"
        f" p95_ms={np.percentile(a,95):.2f}"
        f" effective_fps={1000.0/a.mean():.2f}",
        file=sys.stderr,
        flush=True,
    )
