from __future__ import annotations

"""Compatibility entrypoint for the stable YOLO detector-truth mode.

Detection stays deliberately isolated: no temporal tracker, optical flow, ReID,
or identity logic is allowed until raw YOLO26m person boxes are visibly proven.

The important DeepStream detail here is that external detections are attached as
*detector results*, not as generic object metadata.  The native bridge therefore
sets ``NvDsFrameMeta.bInferDone`` and writes ``NvDsObjectMeta.rect_params`` before
``nvmultistreamtiler -> nvdsosd`` consumes the batch.
"""

import time

from . import stable_yolo_truth_backend as truth

stable_yolo_truth_worker = truth.stable_yolo_truth_worker


def _inject_detector_truth_probe(self, _pad, info):
    buffer = info.get_buffer()
    if buffer is None:
        return self.Gst.PadProbeReturn.OK

    now = time.monotonic()
    added = 0
    logged = getattr(self, "_truth_logged_versions", None)
    if logged is None:
        logged = {}
        self._truth_logged_versions = logged

    for cid, source_id in self.camera_index.items():
        rows = self.boxes.render(cid, now)

        # Apply every camera's latest detector result, including an empty result.
        # NativeMetaBridge.apply_detector_result() marks bInferDone=TRUE and then
        # attaches the Person NvDsObjectMeta rows with rect_params for nvdsosd.
        result = self.bridge.apply_detector_result(buffer, source_id, rows)
        if result > 0:
            added += result

        version = self.boxes.version(cid)
        if version > logged.get(cid, 0):
            logged[cid] = version
            age = self.boxes.age(cid, now)
            age_ms = -1.0 if age is None else age * 1000.0
            print(
                "YOLO_TRUTH_META "
                f"camera={cid} source_id={source_id} version={version} "
                f"raw_boxes={len(rows)} injected={result} age={age_ms:.1f}ms "
                "infer_done=1",
                flush=True,
            )

    with self.det_lock:
        self.meta_boxes += added
    return self.Gst.PadProbeReturn.OK


def install() -> None:
    # First install the tracker-free YOLO26m truth worker and raw box store.
    truth.install()

    # Then replace only the metadata injector with the DeepStream detector-result
    # path.  No tracking/smoothing/flow is introduced here.
    from . import detection

    detection.CameraDetectionV2._inject_boxes_probe = _inject_detector_truth_probe
    print(
        "CAMERA_YOLO_META mode=detector-result bInferDone=1 "
        "object_meta=Person rect_params=source-space tracker=OFF flow=OFF",
        flush=True,
    )


__all__ = ["install", "stable_yolo_truth_worker"]
