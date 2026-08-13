from __future__ import annotations

import math
import threading
import time

import cv2

from .visual_tracker import VisualTracker

try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)
except Exception:
    pass


def _box_center_distance(a, b):
    acx=(float(a.x1)+float(a.x2))*0.5;acy=(float(a.y1)+float(a.y2))*0.5
    bcx=(float(b.x1)+float(b.x2))*0.5;bcy=(float(b.y1)+float(b.y2))*0.5
    aw=max(1.0,float(a.x2)-float(a.x1));ah=max(1.0,float(a.y2)-float(a.y1));bw=max(1.0,float(b.x2)-float(b.x1));bh=max(1.0,float(b.y2)-float(b.y1))
    return math.hypot(acx-bcx,acy-bcy)/max(20.0,aw,ah,bw,bh)


class LatestJpegPublisher:
    """Publish one newest JPEG per camera with no presentation backlog."""

    def __init__(self, camera_id, store, fps=12, quality=82, max_width=960,
                 max_height=540, detections=None, overlay_max_age_ms=350,
                 tracker_config=None, identity_provider=None):
        self.camera_id = camera_id
        self.store = store
        self.interval = 1 / max(1.0, float(fps))
        self.quality = int(quality)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.detections = detections
        self.overlay_max_age_ms = max(0.0, float(overlay_max_age_ms))
        self.identity_provider = identity_provider

        cfg = dict(tracker_config or {})
        camera_zones = dict(cfg.get('camera_exclusion_zones') or {})
        camera_birth_zones = dict(cfg.get('camera_new_track_zones') or {})
        fragment_cameras = set(str(cid) for cid in (cfg.get('fragment_duplicate_cameras') or []))
        camera_start_conf = dict(cfg.get('camera_start_conf') or {})
        camera_low_conf = dict(cfg.get('camera_low_conf_confirm') or {})

        self.visual_tracker = VisualTracker(
            hold_ms=cfg.get('hold_ms', 800),
            memory_ms=cfg.get('memory_ms', 3000),
            prediction_ms=cfg.get('prediction_ms', 420),
            match_iou=cfg.get('match_iou', 0.12),
            reacquire_distance=cfg.get('reacquire_distance', 0.85),
            duplicate_iou=cfg.get('duplicate_iou', 0.68),
            duplicate_containment=cfg.get('duplicate_containment', 0.90),
            duplicate_center_distance=cfg.get('duplicate_center_distance', 0.20),
            fragment_duplicate=(camera_id in fragment_cameras),
            fragment_horizontal_overlap=cfg.get('fragment_horizontal_overlap', 0.78),
            fragment_x_center=cfg.get('fragment_x_center', 0.18),
            fragment_max_area_ratio=cfg.get('fragment_max_area_ratio', 0.55),
            fragment_min_vertical_overlap=cfg.get('fragment_min_vertical_overlap', 0.20),
            fragment_max_vertical_gap=cfg.get('fragment_max_vertical_gap', 0.06),
            low_conf_confirm=camera_low_conf.get(camera_id, cfg.get('low_conf_confirm', 0.08)),
            start_conf=camera_start_conf.get(camera_id, cfg.get('start_conf', 0.34)),
            new_track_min_conf=cfg.get('new_track_min_conf', 0.24),
            strong_confirm_hits=cfg.get('strong_confirm_hits', 2),
            weak_confirm_hits=cfg.get('weak_confirm_hits', 3),
            byte_high_conf=cfg.get('byte_high_conf', 0.24),
            byte_low_conf=cfg.get('byte_low_conf', 0.08),
            byte_second_match_iou=cfg.get('byte_second_match_iou', 0.04),
            byte_match_center=cfg.get('byte_match_center', 0.70),
            byte_second_match_center=cfg.get('byte_second_match_center', 0.50),
            low_match_max_age_ms=cfg.get('low_match_max_age_ms', 650),
            process_noise=cfg.get('process_noise', 0.85),
            measurement_noise=cfg.get('measurement_noise', 0.90),
            velocity_damping=cfg.get('velocity_damping', 0.96),
            size_velocity_damping=cfg.get('size_velocity_damping', 0.60),
            max_prediction_shift_boxes=cfg.get('max_prediction_shift_boxes', 0.55),
            max_prediction_size_ratio=cfg.get('max_prediction_size_ratio', 0.08),
            adaptive_error_low=cfg.get('adaptive_error_low', 0.08),
            adaptive_error_high=cfg.get('adaptive_error_high', 0.25),
            center_response_slow=cfg.get('center_response_slow', 0.42),
            center_response_fast=cfg.get('center_response_fast', 0.88),
            size_response=cfg.get('size_response', 0.30),
            snap_distance_boxes=cfg.get('snap_distance_boxes', 0.65),
            reversal_damping=cfg.get('reversal_damping', 0.15),
            new_track_zones=camera_birth_zones.get(camera_id, []),
            exclusion_zones=camera_zones.get(camera_id, []),
            exclusion_max_box_height=cfg.get('exclusion_max_box_height', 0.24),
            exclusion_overlap_threshold=cfg.get('exclusion_overlap_threshold', 0.35),
        )

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._jpeg = None
        self._version = 0
        self._published_monotonic = 0.0
        self._source_frame_id = 0
        self._last_source_capture_mono = 0.0
        self._started_mono = time.monotonic()
        self._last_encode_ms = 0.0
        self._last_resize_ms = 0.0
        self._last_overlay_ms = 0.0
        self._last_jpeg_ms = 0.0
        self._last_cycle_ms = 0.0
        self._last_publish_source_age_ms = None
        self.encoded = 0
        self.skipped_same_frame = 0
        self.stale_detection_rejects = 0
        self.future_detection_deferrals = 0
        self._current_detection_key = None
        self._current_detection_accepted = False
        self._current_future_counted = False
        self._current_stale_counted = False

    def start(self):
        self._thread = threading.Thread(target=self._run,name=f"core-v1-jpeg-{self.camera_id}",daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._condition:self._condition.notify_all()

    def join(self, timeout=3):
        if self._thread:self._thread.join(timeout)

    def latest(self):
        with self._lock:return self._jpeg,self._version

    def snapshot(self):
        with self._lock:return self._jpeg,self._version,self._published_monotonic,self._source_frame_id

    def metrics(self):
        now=time.monotonic()
        with self._lock:
            elapsed=max(0.001,now-self._started_mono)
            return {
                'encoded':self.encoded,
                'publish_rate':self.encoded/elapsed,
                'skipped_same_frame':self.skipped_same_frame,
                'stale_detection_rejects':self.stale_detection_rejects,
                'future_detection_deferrals':self.future_detection_deferrals,
                'last_encode_ms':self._last_encode_ms,
                'last_resize_ms':self._last_resize_ms,
                'last_overlay_ms':self._last_overlay_ms,
                'last_jpeg_ms':self._last_jpeg_ms,
                'last_cycle_ms':self._last_cycle_ms,
                'frame_budget_ms':self.interval*1000.0,
                'last_publish_source_age_ms':self._last_publish_source_age_ms,
                'last_published_age_ms':((now-self._published_monotonic)*1000.0) if self._published_monotonic else None,
                'source_frame_id':self._source_frame_id,
                'tracker':self.visual_tracker.metrics(),
            }

    def wait_newer(self,last_version:int,timeout:float=0.25):
        deadline=time.monotonic()+max(0.0,float(timeout))
        with self._condition:
            while self._version<=last_version and not self._stop.is_set():
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                self._condition.wait(remaining)
            return self._jpeg,self._version,self._published_monotonic,self._source_frame_id

    def _identity_for_box(self, box):
        if self.identity_provider is None:return None
        try:labels=self.identity_provider.labels(self.camera_id)
        except Exception:return None
        best=None;best_distance=1e9
        for item in labels:
            distance=_box_center_distance(box,item['box'])
            if distance<best_distance:
                best_distance=distance;best=item
        return best if best is not None and best_distance<=0.70 else None

    @staticmethod
    def _detection_key(result):
        return (
            int(result.frame_id),
            float(result.frame_captured_monotonic),
            float(getattr(result, 'produced_monotonic', 0.0)),
        )

    def _draw_detection(self,image,source_width,source_height,now,display_frame_id,display_frame_time):
        max_age_sec=(self.overlay_max_age_ms/1000.0) if self.overlay_max_age_ms>0 else None
        if self.detections is not None:
            result=self.detections.get(self.camera_id)
            if result is not None:
                key=self._detection_key(result)
                with self._lock:
                    if key!=self._current_detection_key:
                        self._current_detection_key=key
                        self._current_detection_accepted=False
                        self._current_future_counted=False
                        self._current_stale_counted=False
                result_time=float(result.frame_captured_monotonic)
                is_future=(
                    int(result.frame_id)>int(display_frame_id)
                    or result_time>float(display_frame_time)
                )
                if is_future:
                    # The latest detector result may race ahead after this display
                    # frame was fetched. Defer it; a newer display frame can still
                    # apply the same real observation on a later publish cycle.
                    with self._lock:
                        if not self._current_future_counted:
                            self.future_detection_deferrals+=1
                            self._current_future_counted=True
                else:
                    source_age=max(0.0,float(display_frame_time)-result_time)
                    if max_age_sec is None or source_age<=max_age_sec:
                        self.visual_tracker.update(result,now,source_width,source_height)
                        with self._lock:self._current_detection_accepted=True
                    else:
                        with self._lock:
                            if not self._current_detection_accepted and not self._current_stale_counted:
                                self.stale_detection_rejects+=1
                                self._current_stale_counted=True

        boxes=self.visual_tracker.visible(now,target_time=display_frame_time,max_observation_age_sec=max_age_sec)
        if not boxes:return image
        h,w=image.shape[:2]
        try:
            source_w=float(source_width);source_h=float(source_height)
        except (TypeError,ValueError,OverflowError):
            return image
        if not math.isfinite(source_w) or not math.isfinite(source_h) or source_w<=0.0 or source_h<=0.0:return image
        sx=w/source_w;sy=h/source_h
        for box in boxes:
            try:values=[float(box.x1),float(box.y1),float(box.x2),float(box.y2),float(box.confidence)]
            except (AttributeError,TypeError,ValueError,OverflowError):continue
            if not all(math.isfinite(value) for value in values):continue
            bx1,by1,bx2,by2,confidence=values
            bx1=max(0.0,min(source_w,bx1));bx2=max(0.0,min(source_w,bx2));by1=max(0.0,min(source_h,by1));by2=max(0.0,min(source_h,by2))
            if bx2<=bx1 or by2<=by1:continue
            x1=max(0,min(w-1,int(round(bx1*sx))));y1=max(0,min(h-1,int(round(by1*sy))));x2=max(0,min(w-1,int(round(bx2*sx))));y2=max(0,min(h-1,int(round(by2*sy))))
            if x2<=x1 or y2<=y1:continue
            identity=self._identity_for_box(box)
            label=identity['global_id'] if identity else f"person {confidence:.2f}"
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2,cv2.LINE_AA)
        return image

    def _run(self):
        last_frame_id=-1;next_at=time.monotonic()
        while not self._stop.is_set():
            now=time.monotonic()
            if now<next_at:self._stop.wait(next_at-now);continue
            next_at+=self.interval
            if next_at<now:
                missed=int((now-next_at)/self.interval)+1;next_at+=missed*self.interval
            frame,_=self.store.get()
            if frame is None:continue
            now=time.monotonic()
            if frame.frame_id==last_frame_id:
                with self._lock:self.skipped_same_frame+=1
                continue

            cycle_started=time.perf_counter();image=frame.image;source_h,source_w=image.shape[:2]
            resize_started=time.perf_counter();scale=min(1.0,self.max_width/max(1,source_w),self.max_height/max(1,source_h))
            if scale<1.0:image=cv2.resize(image,(max(1,round(source_w*scale)),max(1,round(source_h*scale))),interpolation=cv2.INTER_AREA)
            else:image=image.copy()
            resize_ms=(time.perf_counter()-resize_started)*1000.0

            overlay_started=time.perf_counter();image=self._draw_detection(image,source_w,source_h,now,frame.frame_id,frame.captured_monotonic);overlay_ms=(time.perf_counter()-overlay_started)*1000.0
            jpeg_started=time.perf_counter();ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,self.quality]);jpeg_ms=(time.perf_counter()-jpeg_started)*1000.0
            if not ok:continue

            published=time.monotonic();payload=encoded.tobytes();cycle_ms=(time.perf_counter()-cycle_started)*1000.0
            with self._condition:
                self._jpeg=payload;self._version+=1;self._published_monotonic=published;self._source_frame_id=frame.frame_id;self._last_source_capture_mono=float(frame.captured_monotonic);self._last_resize_ms=resize_ms;self._last_overlay_ms=overlay_ms;self._last_jpeg_ms=jpeg_ms;self._last_encode_ms=cycle_ms;self._last_cycle_ms=cycle_ms;self._last_publish_source_age_ms=max(0.0,(published-float(frame.captured_monotonic))*1000.0);self.encoded+=1;self._condition.notify_all()
            last_frame_id=frame.frame_id
