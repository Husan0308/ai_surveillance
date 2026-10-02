import unittest

from scripts.dev_room_mv3dt.diagnose_canonical_retention import (
    config_values, iou, shadow_sample, spans,
)


class CanonicalRetentionDiagnosticsTest(unittest.TestCase):
    def test_comments_cannot_satisfy_effective_parameter_assertions(self):
        values = config_values("# earlyTerminationAge: 2\n%YAML:1.0\nTargetManagement:\n  earlyTerminationAge: 1\n")
        self.assertEqual(values["TargetManagement"]["earlyTerminationAge"], 1)

    def test_shadow_is_current_sample_and_private_reason_stays_unavailable(self):
        row = shadow_sample("274 9 0 0 0 10 20 40 80 0 0 0 0 0 0 0 .02 2 0", "CAM-04")
        self.assertEqual(row["frame"], 274)
        self.assertEqual(row["state"], "INACTIVE")
        self.assertEqual(row["bbox"], [10, 20, 30, 60])
        self.assertIsNone(row["shadow_age"])
        self.assertIsNone(row["termination_reason"])

    def test_adjacent_different_person_deficits_remain_in_formal_window(self):
        # Do not shorten the formal gate just because the missing person changes.
        self.assertEqual(spans(range(274, 300))[0]["frames"], 26)
        self.assertEqual(spans([2, 4, 5]), [
            {"start": 2, "end": 2, "frames": 1, "seconds": .05},
            {"start": 4, "end": 5, "frames": 2, "seconds": .1}])

    def test_shadow_model_geometry_does_not_masquerade_as_detector_box(self):
        self.assertLess(iou([1081, 466, 178, 253], [1129, 641, 36, 60]), .1)


if __name__ == "__main__":
    unittest.main()
