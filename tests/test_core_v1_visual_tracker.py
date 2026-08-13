from __future__ import annotations

import math
from statistics import pstdev
from types import SimpleNamespace
import unittest

from services.ml_service.core_v1.visual_tracker import VisualBox, VisualTracker


def _box(
    center_x: float,
    *,
    center_y: float = 180.0,
    width: float = 60.0,
    height: float = 100.0,
    confidence: float = 0.90,
) -> VisualBox:
    return VisualBox(
        center_x - width * 0.5,
        center_y - height * 0.5,
        center_x + width * 0.5,
        center_y + height * 0.5,
        confidence,
    )


def _center_x(box: VisualBox) -> float:
    return (box.x1 + box.x2) * 0.5


def _center_y(box: VisualBox) -> float:
    return (box.y1 + box.y2) * 0.5


def _width(box: VisualBox) -> float:
    return box.x2 - box.x1


def _height(box: VisualBox) -> float:
    return box.y2 - box.y1


def _result(frame_id: int, captured_at: float, boxes: list[VisualBox]):
    return SimpleNamespace(
        frame_id=frame_id,
        frame_captured_monotonic=captured_at,
        boxes=boxes,
    )


class VisualTrackerRegressionTests(unittest.TestCase):
    """Deterministic presentation-tracker regressions; no cameras or clocks."""

    def _tracker(self, **overrides) -> VisualTracker:
        config = {
            "hold_ms": 900,
            "memory_ms": 4000,
            "prediction_ms": 400,
            "match_iou": 0.05,
            "reacquire_distance": 1.20,
            "byte_high_conf": 0.50,
            "byte_low_conf": 0.10,
            "byte_match_center": 1.10,
            "byte_second_match_center": 0.75,
            "byte_second_match_iou": 0.02,
            "low_match_max_age_ms": 700,
            "start_conf": 0.70,
            "new_track_min_conf": 0.40,
            "strong_confirm_hits": 1,
            "weak_confirm_hits": 3,
            "process_noise": 0.80,
            "measurement_noise": 0.85,
            "velocity_damping": 0.985,
            "max_prediction_shift_boxes": 0.60,
            "max_prediction_size_ratio": 0.10,
            "adaptive_error_low": 0.08,
            "adaptive_error_high": 0.25,
            "center_response_slow": 0.35,
            "center_response_fast": 0.88,
            "size_response": 0.35,
            "snap_distance_boxes": 0.80,
            "reversal_damping": 0.12,
        }
        config.update(overrides)
        return VisualTracker(**config)

    def _update(
        self,
        tracker: VisualTracker,
        frame_id: int,
        captured_at: float,
        boxes: list[VisualBox],
        *,
        arrival_delay: float = 0.02,
    ) -> None:
        tracker.update(
            _result(frame_id, captured_at, boxes),
            now=captured_at + arrival_delay,
            source_width=640,
            source_height=360,
        )

    def _single_visible(
        self,
        tracker: VisualTracker,
        *,
        now: float,
        target_time: float,
        max_observation_age_sec: float | None = None,
    ) -> VisualBox:
        visible = tracker.visible(
            now=now,
            target_time=target_time,
            max_observation_age_sec=max_observation_age_sec,
        )
        self.assertEqual(len(visible), 1)
        return visible[0]

    def test_standing_jitter_is_smoothed_for_center_and_size(self) -> None:
        tracker = self._tracker()
        base_time = 100.0
        center_jitter = (-2.5, 1.8, -1.2, 2.2, -1.7, 1.3)
        vertical_jitter = (-2.0, 1.4, -0.8, 1.8, -1.5, 1.0)
        width_jitter = (-5.0, 3.5, -3.0, 4.5, -4.0, 2.5)
        height_jitter = (-7.0, 5.0, -4.0, 6.0, -5.5, 3.5)
        raw_centers: list[float] = []
        raw_vertical_centers: list[float] = []
        raw_widths: list[float] = []
        raw_heights: list[float] = []
        shown_centers: list[float] = []
        shown_vertical_centers: list[float] = []
        shown_widths: list[float] = []
        shown_heights: list[float] = []

        for index in range(60):
            captured_at = base_time + index * 0.1
            center = 220.0 + center_jitter[index % len(center_jitter)]
            center_y = 180.0 + vertical_jitter[index % len(vertical_jitter)]
            width = 64.0 + width_jitter[index % len(width_jitter)]
            height = 100.0 + height_jitter[index % len(height_jitter)]
            self._update(
                tracker,
                index + 1,
                captured_at,
                [_box(center, center_y=center_y, width=width, height=height)],
            )
            shown = self._single_visible(
                tracker,
                now=captured_at + 0.02,
                target_time=captured_at,
            )
            if index >= 15:
                raw_centers.append(center)
                raw_vertical_centers.append(center_y)
                raw_widths.append(width)
                raw_heights.append(height)
                shown_centers.append(_center_x(shown))
                shown_vertical_centers.append(_center_y(shown))
                shown_widths.append(_width(shown))
                shown_heights.append(_height(shown))

        self.assertLess(pstdev(shown_centers), pstdev(raw_centers) * 0.75)
        self.assertLess(
            pstdev(shown_vertical_centers), pstdev(raw_vertical_centers) * 0.75
        )
        self.assertLess(pstdev(shown_widths), pstdev(raw_widths) * 0.65)
        self.assertLess(pstdev(shown_heights), pstdev(raw_heights) * 0.65)

    def test_display_timestamp_prediction_reduces_fast_motion_lag(self) -> None:
        tracker = self._tracker()
        base_time = 200.0
        speed_px_per_sec = 140.0

        for index in range(9):
            captured_at = base_time + index * 0.1
            center = 100.0 + speed_px_per_sec * (captured_at - base_time)
            self._update(tracker, index + 1, captured_at, [_box(center)])

        observation_time = base_time + 0.8
        display_time = observation_time + 0.25
        truth_at_display = 100.0 + speed_px_per_sec * (display_time - base_time)
        stale = self._single_visible(
            tracker,
            now=display_time + 0.02,
            target_time=observation_time,
        )
        current = self._single_visible(
            tracker,
            now=display_time + 0.02,
            target_time=display_time,
        )
        stale_error = abs(_center_x(stale) - truth_at_display)
        current_error = abs(_center_x(current) - truth_at_display)

        self.assertGreater(_center_x(current), _center_x(stale) + 8.0)
        self.assertLess(current_error, stale_error * 0.60)

    def test_sudden_stop_has_bounded_overshoot(self) -> None:
        tracker = self._tracker()
        base_time = 300.0

        for index in range(8):
            captured_at = base_time + index * 0.1
            self._update(tracker, index + 1, captured_at, [_box(100.0 + index * 10.0)])

        stop_x = 170.0
        for offset in range(1, 4):
            captured_at = base_time + (7 + offset) * 0.1
            self._update(tracker, 8 + offset, captured_at, [_box(stop_x)])

        last_observation = base_time + 1.0
        predicted = self._single_visible(
            tracker,
            now=last_observation + 0.30,
            target_time=last_observation + 0.30,
        )
        self.assertLessEqual(abs(_center_x(predicted) - stop_x), _width(predicted) * 0.18)

    def test_partial_stop_sample_damps_old_velocity(self) -> None:
        tracker = self._tracker()
        base_time = 350.0
        for index in range(8):
            captured_at = base_time + index * 0.1
            self._update(
                tracker,
                index + 1,
                captured_at,
                [_box(100.0 + index * 12.0)],
            )

        partial_stop_time = base_time + 0.8
        partial_stop_x = 190.0
        self._update(tracker, 9, partial_stop_time, [_box(partial_stop_x)])
        predicted = self._single_visible(
            tracker,
            now=partial_stop_time + 0.32,
            target_time=partial_stop_time + 0.30,
        )
        self.assertLessEqual(
            abs(_center_x(predicted) - partial_stop_x),
            _width(predicted) * 0.20,
        )

    def test_direction_reversal_changes_prediction_and_increments_counter(self) -> None:
        tracker = self._tracker()
        base_time = 400.0
        centers = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 140.0, 130.0, 120.0]

        for index, center in enumerate(centers):
            captured_at = base_time + index * 0.1
            self._update(tracker, index + 1, captured_at, [_box(center)])

        observation_time = base_time + (len(centers) - 1) * 0.1
        at_observation = self._single_visible(
            tracker,
            now=observation_time + 0.02,
            target_time=observation_time,
        )
        predicted = self._single_visible(
            tracker,
            now=observation_time + 0.22,
            target_time=observation_time + 0.20,
        )

        self.assertLess(_center_x(predicted), _center_x(at_observation) - 2.0)
        self.assertGreaterEqual(tracker.metrics()["direction_reversals"], 1)

    def test_empty_results_cannot_extend_prediction_horizon_or_shift_cap(self) -> None:
        repeated_misses = self._tracker(
            hold_ms=1400,
            prediction_ms=300,
            max_prediction_shift_boxes=0.40,
        )
        direct_prediction = self._tracker(
            hold_ms=1400,
            prediction_ms=300,
            max_prediction_shift_boxes=0.40,
        )
        base_time = 500.0

        for index in range(8):
            captured_at = base_time + index * 0.1
            detection = [_box(100.0 + index * 9.0)]
            self._update(repeated_misses, index + 1, captured_at, detection)
            self._update(direct_prediction, index + 1, captured_at, detection)

        observation_time = base_time + 0.7
        at_observation = self._single_visible(
            repeated_misses,
            now=observation_time + 0.02,
            target_time=observation_time,
        )
        for miss in range(1, 10):
            captured_at = observation_time + miss * 0.1
            self._update(repeated_misses, 8 + miss, captured_at, [])

        display_time = observation_time + 0.9
        after_empty_results = self._single_visible(
            repeated_misses,
            now=display_time + 0.02,
            target_time=display_time,
        )
        without_empty_results = self._single_visible(
            direct_prediction,
            now=display_time + 0.02,
            target_time=display_time,
        )
        actual_shift = abs(_center_x(after_empty_results) - _center_x(at_observation))
        shift_cap = max(12.0, max(_width(at_observation), at_observation.y2 - at_observation.y1) * 0.40)

        self.assertLessEqual(actual_shift, shift_cap + 0.5)
        self.assertLessEqual(
            actual_shift,
            abs(_center_x(without_empty_results) - _center_x(at_observation)) + 1.0,
        )

    def test_short_miss_predicts_but_freshness_rejects_long_miss(self) -> None:
        tracker = self._tracker(hold_ms=750, prediction_ms=350)
        base_time = 600.0

        for index in range(7):
            captured_at = base_time + index * 0.1
            self._update(tracker, index + 1, captured_at, [_box(150.0 + index * 7.0)])

        observation_time = base_time + 0.6
        observed = self._single_visible(
            tracker,
            now=observation_time + 0.02,
            target_time=observation_time,
        )
        short_miss = self._single_visible(
            tracker,
            now=observation_time + 0.25,
            target_time=observation_time + 0.25,
            max_observation_age_sec=0.40,
        )
        self.assertGreater(_center_x(short_miss), _center_x(observed))

        stale = tracker.visible(
            now=observation_time + 0.55,
            target_time=observation_time + 0.55,
            max_observation_age_sec=0.40,
        )
        self.assertEqual(stale, [])
        self.assertGreaterEqual(tracker.metrics()["stale_prediction_rejects"], 1)

        beyond_hold = tracker.visible(
            now=observation_time + 0.80,
            target_time=observation_time + 0.80,
        )
        self.assertEqual(beyond_hold, [])

    def test_two_close_people_remain_two_visible_boxes(self) -> None:
        tracker = self._tracker()
        base_time = 700.0
        pairs = [(100.0, 150.0), (106.0, 146.0), (112.0, 144.0), (116.0, 146.0)]

        for index, (left, right) in enumerate(pairs):
            captured_at = base_time + index * 0.1
            self._update(
                tracker,
                index + 1,
                captured_at,
                [_box(left), _box(right)],
            )
            visible = tracker.visible(now=captured_at + 0.02, target_time=captured_at)
            self.assertEqual(len(visible), 2)
            shown_centers = sorted(_center_x(item) for item in visible)
            self.assertGreater(shown_centers[1] - shown_centers[0], 15.0)

    def test_detector_realistic_overlapping_people_birth_separately(self) -> None:
        tracker = self._tracker(strong_confirm_hits=2, weak_confirm_hits=3)
        base_time = 760.0

        self._update(tracker, 1, base_time, [_box(100.0), _box(120.0)])
        first = tracker.metrics()
        self.assertEqual(first["births"], 0)
        self.assertEqual(first["birth_candidates"], 2)

        self._update(
            tracker,
            2,
            base_time + 0.1,
            [_box(104.0), _box(124.0)],
        )
        visible = tracker.visible(now=base_time + 0.12, target_time=base_time + 0.1)
        self.assertEqual(tracker.metrics()["births"], 2)
        self.assertEqual(len(visible), 2)
        self.assertGreater(
            max(_center_x(box) for box in visible)
            - min(_center_x(box) for box in visible),
            12.0,
        )

        self._update(
            tracker,
            3,
            base_time + 0.2,
            [_box(108.0), _box(128.0)],
        )
        self.assertEqual(tracker.metrics()["births"], 2)
        self.assertEqual(
            len(tracker.visible(now=base_time + 0.22, target_time=base_time + 0.2)),
            2,
        )

    def test_near_identical_detections_still_birth_only_one_track(self) -> None:
        tracker = self._tracker(strong_confirm_hits=2, weak_confirm_hits=3)
        base_time = 770.0
        for index in range(3):
            captured_at = base_time + index * 0.1
            self._update(
                tracker,
                index + 1,
                captured_at,
                [_box(100.0), _box(108.0)],
            )

        self.assertEqual(tracker.metrics()["births"], 1)
        self.assertEqual(
            len(tracker.visible(now=base_time + 0.22, target_time=base_time + 0.2)),
            1,
        )

    def test_birth_confirmation_scales_with_detector_interval(self) -> None:
        tracker = self._tracker(strong_confirm_hits=2, weak_confirm_hits=3)
        base_time = 780.0
        self._update(tracker, 1, base_time, [_box(100.0)])
        self._update(tracker, 2, base_time + 0.30, [_box(160.0)])

        self.assertEqual(tracker.metrics()["births"], 1)
        observed = self._single_visible(
            tracker,
            now=base_time + 0.32,
            target_time=base_time + 0.30,
        )
        predicted = self._single_visible(
            tracker,
            now=base_time + 0.42,
            target_time=base_time + 0.40,
        )
        self.assertAlmostEqual(_center_x(observed), 160.0, delta=0.5)
        self.assertGreater(_center_x(predicted), _center_x(observed) + 8.0)

    def test_brief_single_detection_occlusion_keeps_two_tracks_visible(self) -> None:
        tracker = self._tracker()
        base_time = 790.0
        self._update(tracker, 1, base_time, [_box(80.0), _box(120.0)])
        self._update(tracker, 2, base_time + 0.1, [_box(90.0), _box(110.0)])
        self._update(tracker, 3, base_time + 0.2, [_box(100.0)])

        visible = tracker.visible(now=base_time + 0.22, target_time=base_time + 0.2)
        self.assertEqual(len(visible), 2)
        self.assertEqual(tracker.metrics()["tracks_in_memory"], 2)

    def test_weak_nearby_detection_cannot_hijack_static_track(self) -> None:
        tracker = self._tracker()
        base_time = 795.0
        self._update(tracker, 1, base_time, [_box(100.0)])
        self._update(tracker, 2, base_time + 0.1, [_box(100.0)])
        self._update(
            tracker,
            3,
            base_time + 0.2,
            [_box(130.0, confidence=0.11)],
        )

        metrics = tracker.metrics()
        self.assertEqual(metrics["low_matches"], 0)
        self.assertEqual(metrics["births"], 1)
        predicted = self._single_visible(
            tracker,
            now=base_time + 0.42,
            target_time=base_time + 0.40,
        )
        self.assertAlmostEqual(_center_x(predicted), 100.0, delta=3.0)

    def test_weak_detection_continues_plausible_motion(self) -> None:
        tracker = self._tracker()
        base_time = 797.0
        for index, center in enumerate((100.0, 106.0, 112.0)):
            self._update(
                tracker,
                index + 1,
                base_time + index * 0.1,
                [_box(center)],
            )
        self._update(
            tracker,
            4,
            base_time + 0.3,
            [_box(118.0, confidence=0.11)],
        )

        metrics = tracker.metrics()
        self.assertEqual(metrics["low_matches"], 1)
        self.assertEqual(metrics["births"], 1)
        shown = self._single_visible(
            tracker,
            now=base_time + 0.32,
            target_time=base_time + 0.3,
        )
        self.assertGreaterEqual(_center_x(shown), 112.0)
        self.assertLessEqual(_center_x(shown), 122.0)

    def test_invalid_detector_boxes_are_rejected_and_clipped(self) -> None:
        tracker = self._tracker()
        base_time = 799.0
        invalid = [
            VisualBox(float("nan"), 10.0, 30.0, 80.0, 0.9),
            VisualBox(10.0, 10.0, float("inf"), 80.0, 0.9),
            VisualBox(30.0, 10.0, 20.0, 80.0, 0.9),
            VisualBox(10.0, 20.0, 30.0, 20.0, 0.9),
            VisualBox(-30.0, 10.0, -10.0, 80.0, 0.9),
        ]
        self._update(tracker, 1, base_time, invalid)
        self.assertEqual(
            tracker.visible(now=base_time + 0.02, target_time=base_time),
            [],
        )
        self.assertEqual(tracker.metrics()["invalid_detections"], len(invalid))

        self._update(
            tracker,
            2,
            base_time + 0.1,
            [VisualBox(-10.0, 120.0, 30.0, 220.0, 2.0)],
        )
        shown = self._single_visible(
            tracker,
            now=base_time + 0.12,
            target_time=base_time + 0.1,
        )
        self.assertTrue(all(math.isfinite(value) for value in (shown.x1, shown.y1, shown.x2, shown.y2, shown.confidence)))
        self.assertGreaterEqual(shown.x1, 0.0)
        self.assertLessEqual(shown.x2, 640.0)
        self.assertLessEqual(shown.confidence, 1.0)

    def test_low_confidence_can_continue_but_never_birth_a_track(self) -> None:
        tracker = self._tracker(
            start_conf=0.70,
            new_track_min_conf=0.15,
            strong_confirm_hits=1,
            weak_confirm_hits=2,
        )
        base_time = 800.0
        self._update(tracker, 1, base_time, [_box(120.0, confidence=0.90)])
        self._update(tracker, 2, base_time + 0.1, [_box(124.0, confidence=0.25)])

        after_continuation = tracker.metrics()
        self.assertEqual(after_continuation["births"], 1)
        self.assertGreaterEqual(after_continuation["low_matches"], 1)

        for frame_id in range(3, 7):
            captured_at = base_time + (frame_id - 1) * 0.1
            self._update(
                tracker,
                frame_id,
                captured_at,
                [_box(350.0, confidence=0.25)],
            )

        metrics = tracker.metrics()
        self.assertEqual(metrics["births"], 1)
        self.assertEqual(metrics["active_tracks"], 1)

    def test_one_result_cannot_supply_multiple_birth_hits(self) -> None:
        tracker = self._tracker(
            strong_confirm_hits=2,
            weak_confirm_hits=3,
            duplicate_iou=0.95,
            duplicate_containment=0.99,
            duplicate_center_distance=0.05,
        )
        base_time = 900.0

        self._update(
            tracker,
            1,
            base_time,
            [
                _box(100.0, width=50.0, confidence=0.90),
                _box(125.0, width=50.0, confidence=0.88),
            ],
        )
        self.assertEqual(tracker.metrics()["births"], 0)

        self._update(
            tracker,
            2,
            base_time + 0.1,
            [_box(104.0, width=50.0, confidence=0.90)],
        )
        self.assertEqual(tracker.metrics()["births"], 1)
        self.assertEqual(
            len(tracker.visible(now=base_time + 0.12, target_time=base_time + 0.1)),
            1,
        )

    def test_far_error_snaps_center_and_reports_correction(self) -> None:
        tracker = self._tracker(
            byte_match_center=3.0,
            snap_distance_boxes=0.40,
        )
        base_time = 950.0
        self._update(tracker, 1, base_time, [_box(100.0)])
        self._update(tracker, 2, base_time + 0.1, [_box(160.0)])

        shown = self._single_visible(
            tracker,
            now=base_time + 0.12,
            target_time=base_time + 0.1,
        )
        metrics = tracker.metrics()
        self.assertAlmostEqual(_center_x(shown), 160.0, delta=1.0)
        self.assertEqual(metrics["corrections"], 1)
        self.assertEqual(metrics["snaps"], 1)

    def test_predicted_size_change_is_bounded(self) -> None:
        tracker = self._tracker(max_prediction_size_ratio=0.08)
        base_time = 975.0
        for index, box_width in enumerate((60.0, 64.0, 68.0, 72.0, 76.0)):
            captured_at = base_time + index * 0.1
            self._update(
                tracker,
                index + 1,
                captured_at,
                [_box(180.0, width=box_width)],
            )

        observation_time = base_time + 0.4
        observed = self._single_visible(
            tracker,
            now=observation_time + 0.02,
            target_time=observation_time,
        )
        predicted = self._single_visible(
            tracker,
            now=observation_time + 0.42,
            target_time=observation_time + 0.40,
        )
        self.assertLessEqual(
            abs(_width(predicted) - _width(observed)),
            _width(observed) * 0.08 + 0.1,
        )

    def test_low_confidence_occlusion_keeps_two_existing_people(self) -> None:
        tracker = self._tracker()
        base_time = 990.0
        self._update(
            tracker,
            1,
            base_time,
            [_box(120.0), _box(220.0)],
        )
        self._update(
            tracker,
            2,
            base_time + 0.1,
            [_box(126.0), _box(214.0)],
        )
        self._update(
            tracker,
            3,
            base_time + 0.2,
            [_box(132.0), _box(208.0, confidence=0.25)],
        )

        visible = tracker.visible(
            now=base_time + 0.22,
            target_time=base_time + 0.2,
        )
        self.assertEqual(len(visible), 2)
        self.assertGreaterEqual(tracker.metrics()["low_matches"], 1)

    def test_confidence_boundary_cannot_change_hijack_authority(self) -> None:
        for confidence in (0.239, 0.240):
            with self.subTest(confidence=confidence):
                tracker = VisualTracker()
                self._update(tracker, 1, 1100.0, [_box(200.0)])
                self._update(tracker, 2, 1100.1, [_box(200.0)])
                before = tracker.metrics()
                self._update(
                    tracker,
                    3,
                    1100.2,
                    [_box(244.0, confidence=confidence)],
                )
                metrics = tracker.metrics()
                self.assertEqual(metrics["high_matches"], before["high_matches"])
                self.assertEqual(metrics["low_matches"], before["low_matches"])
                shown = self._single_visible(
                    tracker,
                    now=1100.42,
                    target_time=1100.4,
                )
                self.assertAlmostEqual(_center_x(shown), 200.0, delta=3.0)

    def test_start_confidence_boundary_has_continuous_far_damping(self) -> None:
        future_centers = []
        for confidence in (0.339, 0.340):
            tracker = VisualTracker()
            base_time = 1105.0
            self._update(tracker, 1, base_time, [_box(200.0)])
            self._update(tracker, 2, base_time + 0.1, [_box(200.0)])
            self._update(
                tracker,
                3,
                base_time + 0.2,
                [_box(244.0, confidence=confidence)],
            )
            shown = self._single_visible(
                tracker,
                now=base_time + 0.42,
                target_time=base_time + 0.4,
            )
            future_centers.append(_center_x(shown))
            self.assertAlmostEqual(_center_x(shown), 244.0, delta=2.0)

        self.assertLess(abs(future_centers[1] - future_centers[0]), 1.0)

    def test_borderline_high_distractor_cannot_preempt_better_low_match(self) -> None:
        tracker = VisualTracker()
        base_time = 1110.0
        for index in range(3):
            self._update(
                tracker,
                index + 1,
                base_time + index * 0.1,
                [_box(100.0)],
            )
        before = tracker.metrics()
        self._update(
            tracker,
            4,
            base_time + 0.3,
            [
                _box(170.0, confidence=0.240),
                _box(101.0, confidence=0.239),
            ],
        )
        shown = self._single_visible(
            tracker,
            now=base_time + 0.32,
            target_time=base_time + 0.3,
        )
        metrics = tracker.metrics()
        self.assertEqual(metrics["low_matches"], before["low_matches"] + 1)
        self.assertEqual(metrics["high_matches"], before["high_matches"])
        self.assertLessEqual(_center_x(shown), 105.0)
        self.assertEqual(metrics["births"], 1)

    def test_moderate_far_correction_has_no_fly_ahead_cliff(self) -> None:
        future_offsets = []
        for displacement in range(20, 41):
            tracker = VisualTracker()
            base_time = 1112.0
            self._update(tracker, 1, base_time, [_box(200.0)])
            self._update(tracker, 2, base_time + 0.1, [_box(200.0)])
            self._update(
                tracker,
                3,
                base_time + 0.2,
                [_box(200.0 + displacement)],
            )
            predicted = self._single_visible(
                tracker,
                now=base_time + 0.42,
                target_time=base_time + 0.4,
            )
            future_offsets.append(_center_x(predicted) - 200.0)
            self.assertLessEqual(
                _center_x(predicted) - (200.0 + displacement),
                20.0,
            )

        adjacent_changes = [
            abs(right - left)
            for left, right in zip(future_offsets, future_offsets[1:])
        ]
        self.assertLess(max(adjacent_changes), 3.0)

    def test_far_recovery_radius_is_continuous_at_start_confidence(self) -> None:
        final_centers = []
        for confidence in (0.3399, 0.3400):
            tracker = VisualTracker()
            base_time = 1115.0
            self._update(tracker, 1, base_time, [_box(200.0)])
            self._update(tracker, 2, base_time + 0.1, [_box(200.0)])
            for offset in range(2, 5):
                self._update(
                    tracker,
                    offset + 1,
                    base_time + offset * 0.1,
                    [_box(255.0, confidence=confidence)],
                )
            visible = tracker.visible(
                now=base_time + 0.42,
                target_time=base_time + 0.4,
            )
            self.assertEqual(tracker.metrics()["births"], 1)
            self.assertEqual(tracker.metrics()["tracks_in_memory"], 1)
            self.assertEqual(len(visible), 1)
            final_centers.append(_center_x(visible[0]))

        self.assertLess(abs(final_centers[1] - final_centers[0]), 1.0)

    def test_strong_far_observation_reinitializes_without_duplicate_or_fly_ahead(self) -> None:
        tracker = VisualTracker()
        base_time = 1120.0
        self._update(tracker, 1, base_time, [_box(100.0)])
        self._update(tracker, 2, base_time + 0.1, [_box(100.0)])
        self._update(tracker, 3, base_time + 0.2, [_box(171.0)])

        observed = self._single_visible(
            tracker,
            now=base_time + 0.22,
            target_time=base_time + 0.2,
        )
        predicted = self._single_visible(
            tracker,
            now=base_time + 0.42,
            target_time=base_time + 0.4,
        )
        metrics = tracker.metrics()
        self.assertAlmostEqual(_center_x(observed), 171.0, delta=1.0)
        self.assertAlmostEqual(_center_x(predicted), 171.0, delta=2.0)
        self.assertEqual(metrics["births"], 1)
        self.assertEqual(metrics["tracks_in_memory"], 1)
        self.assertEqual(metrics["snaps"], 1)

    def test_irregular_fast_motion_uses_observation_prediction_for_association(self) -> None:
        tracker = VisualTracker()
        base_time = 1130.0
        offsets = (0.0, 0.28, 0.36, 0.68, 0.78, 1.12, 1.20, 1.49)
        speed = 340.0
        for index, offset in enumerate(offsets):
            center = 50.0 + speed * offset
            self._update(
                tracker,
                index + 1,
                base_time + offset,
                [_box(center)],
            )

        metrics = tracker.metrics()
        visible = tracker.visible(
            now=base_time + offsets[-1] + 0.02,
            target_time=base_time + offsets[-1],
        )
        self.assertEqual(metrics["births"], 1)
        self.assertEqual(metrics["tracks_in_memory"], 1)
        self.assertGreaterEqual(metrics["high_matches"], len(offsets) - 2)
        self.assertEqual(len(visible), 1)

    def test_weak_observation_cannot_poison_next_strong_velocity_anchor(self) -> None:
        future_shifts = []
        for weak_center in range(139, 146):
            tracker = VisualTracker()
            base_time = 1140.0
            for index, center in enumerate((100.0, 110.0, 120.0)):
                self._update(
                    tracker,
                    index + 1,
                    base_time + index * 0.1,
                    [_box(center)],
                )
            self._update(
                tracker,
                4,
                base_time + 0.3,
                [_box(float(weak_center), confidence=0.23)],
            )
            self._update(tracker, 5, base_time + 0.4, [_box(140.0)])
            observed = self._single_visible(
                tracker,
                now=base_time + 0.42,
                target_time=base_time + 0.4,
            )
            predicted = self._single_visible(
                tracker,
                now=base_time + 0.62,
                target_time=base_time + 0.6,
            )
            future_shifts.append(_center_x(predicted) - _center_x(observed))
            self.assertEqual(tracker.metrics()["low_matches"], 1)

        self.assertGreater(min(future_shifts), 12.0)
        self.assertLess(max(future_shifts) - min(future_shifts), 4.0)

    def test_size_outlier_is_rejected_or_bounded_by_confidence(self) -> None:
        for confidence in (0.08, 0.12, 0.239, 0.240):
            with self.subTest(confidence=confidence):
                tracker = VisualTracker()
                self._update(tracker, 1, 1150.0, [_box(200.0)])
                self._update(tracker, 2, 1150.1, [_box(200.0)])
                self._update(
                    tracker,
                    3,
                    1150.2,
                    [
                        _box(
                            200.0,
                            width=120.0,
                            height=200.0,
                            confidence=confidence,
                        )
                    ],
                )
                shown = self._single_visible(
                    tracker,
                    now=1150.22,
                    target_time=1150.2,
                )
                self.assertLess(_width(shown), 63.0)
                self.assertLess(_height(shown), 105.0)

        tracker = VisualTracker()
        self._update(tracker, 1, 1151.0, [_box(200.0)])
        self._update(tracker, 2, 1151.1, [_box(200.0)])
        self._update(
            tracker,
            3,
            1151.2,
            [_box(200.0, width=600.0, height=340.0)],
        )
        shown = self._single_visible(
            tracker,
            now=1151.22,
            target_time=1151.2,
        )
        self.assertAlmostEqual(_width(shown), 60.0, delta=1.0)
        self.assertAlmostEqual(_height(shown), 100.0, delta=1.0)
        self.assertEqual(tracker.metrics()["births"], 1)

    def test_slowdown_damping_has_no_threshold_cliff(self) -> None:
        future_shifts = []
        for final_stride in (5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6):
            tracker = VisualTracker()
            base_time = 1160.0
            for index in range(8):
                self._update(
                    tracker,
                    index + 1,
                    base_time + index * 0.1,
                    [_box(100.0 + index * 12.0)],
                )
            observation = base_time + 0.8
            self._update(
                tracker,
                9,
                observation,
                [_box(184.0 + final_stride)],
            )
            observed = self._single_visible(
                tracker,
                now=observation + 0.02,
                target_time=observation,
            )
            predicted = self._single_visible(
                tracker,
                now=observation + 0.22,
                target_time=observation + 0.2,
            )
            future_shifts.append(_center_x(predicted) - _center_x(observed))

        adjacent_changes = [
            abs(right - left)
            for left, right in zip(future_shifts, future_shifts[1:])
        ]
        self.assertTrue(
            all(right >= left for left, right in zip(future_shifts, future_shifts[1:]))
        )
        self.assertLess(max(adjacent_changes), 2.5)

    def test_stale_reacquisition_fragment_is_suppressed_without_merging_people(self) -> None:
        tracker = VisualTracker()
        base_time = 1170.0
        self._update(tracker, 1, base_time, [_box(100.0)])
        self._update(tracker, 2, base_time + 0.1, [_box(100.0)])
        for frame_id in range(3, 10):
            self._update(
                tracker,
                frame_id,
                base_time + (frame_id - 1) * 0.1,
                [],
            )
        self._update(tracker, 10, base_time + 0.9, [_box(250.0)])
        self._update(tracker, 11, base_time + 1.0, [_box(250.0)])
        for frame_id, center in enumerate(
            (220.0, 190.0, 160.0, 135.0, 125.0, 125.0),
            start=12,
        ):
            self._update(
                tracker,
                frame_id,
                base_time + (frame_id - 1) * 0.1,
                [_box(center)],
            )

        self.assertEqual(tracker.metrics()["births"], 2)
        self.assertEqual(tracker.metrics()["tracks_in_memory"], 2)
        self._update(tracker, 18, base_time + 1.7, [_box(105.0)])
        self.assertEqual(
            sum(track.reacquire_pending for track in tracker._tracks.values()),
            1,
        )
        visible = tracker.visible(
            now=base_time + 1.72,
            target_time=base_time + 1.7,
        )
        self.assertEqual(len(visible), 1)
        self.assertAlmostEqual(_center_x(visible[0]), 105.0, delta=8.0)
        self.assertEqual(
            len(
                tracker.visible(
                    now=base_time + 1.77,
                    target_time=base_time + 1.75,
                )
            ),
            1,
        )

        self._update(
            tracker,
            19,
            base_time + 1.8,
            [_box(105.0), _box(125.0)],
        )
        both_current = tracker.visible(
            now=base_time + 1.82,
            target_time=base_time + 1.8,
        )
        self.assertEqual(len(both_current), 2)

    def test_metrics_expose_required_tuning_signals(self) -> None:
        metrics = self._tracker().metrics()
        required = {
            "algorithm",
            "active_tracks",
            "high_matches",
            "low_matches",
            "prediction_renders",
            "average_observation_age_ms",
            "average_prediction_horizon_ms",
            "corrections",
            "snaps",
            "direction_reversals",
            "stale_prediction_rejects",
        }
        self.assertTrue(required.issubset(metrics), required - set(metrics))
        self.assertIsInstance(metrics["algorithm"], str)
        self.assertEqual(metrics["active_tracks"], 0)


if __name__ == "__main__":
    unittest.main()
