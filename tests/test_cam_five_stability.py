import copy
import unittest

from scripts.cam_five_validation.check_stability import assess, parse_rows


class FiveCameraStabilityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.text = (
            "GROUP GRAPH nvstreammux(batch=5, inference=0\n"
            "CAM-01 END input=13200 hardware_decoder=1 errors=0 warnings=0\n"
            "CAM-02 END input=13200 hardware_decoder=1 errors=0 warnings=0\n"
            "CAM-03 END input=13200 hardware_decoder=1 errors=0 warnings=0\n"
            "CAM-04 END input=13200 hardware_decoder=1 errors=0 warnings=0\n"
            "CAM-05 END input=13200 hardware_decoder=1 errors=0 warnings=0\n"
            "GROUP END output=13200 fatal=0 shared_errors=0 shared_warnings=0"
        )
        self.sources = {"CAM-01": [], "CAM-02": [], "CAM-03": [], "CAM-04": [], "CAM-05": []}
        self.group = []
        self.gpu = []
        for t in range(30, 661, 5):
            for cid in self.sources:
                self.sources[cid].append(
                    dict(
                        elapsed=t,
                        input=t * 20,
                        fps=20,
                        age=0.05,
                        pts_ns=t * 1000000000,
                        pts_backwards=0,
                        pts_duplicates=0,
                        max_gap_ms=350,
                        queue=0,
                        queue_ms=0,
                        rtp_lost=0,
                        rtp_late=0,
                        errors=0,
                        warnings=0,
                        decoder=1,
                    )
                )
            self.group.append(
                dict(
                    elapsed=t,
                    output=t * 20,
                    fps=20,
                    age=0.05,
                    pts_ns=t * 1000000000,
                    pts_backwards=0,
                    pts_duplicates=0,
                    max_gap_ms=350,
                    rss_mib=320,
                    cpu_pct=8,
                    dropped=0,
                    shared_errors=0,
                    shared_warnings=0,
                )
            )
            self.gpu.append(
                dict(
                    elapsed=t,
                    memory_used_mib=400,
                    gpu_pct=1,
                    decoder_pct=8,
                    encoder_pct=5,
                )
            )

    def test_healthy_evidence(self):
        self.assertEqual(assess(self.sources, self.group, self.text, self.gpu)["status"], "PASS")

    def test_missing_camera_rejected(self):
        data = {"CAM-01": self.sources["CAM-01"], "CAM-02": self.sources["CAM-02"], "CAM-03": self.sources["CAM-03"], "CAM-04": self.sources["CAM-04"]}
        self.assertEqual(assess(data, self.group, self.text, self.gpu)["status"], "BLOCKED")

    def test_frozen_source_rejected(self):
        data = copy.deepcopy(self.sources)
        data["CAM-02"][70]["input"] = data["CAM-02"][69]["input"]
        self.assertEqual(assess(data, self.group, self.text, self.gpu)["status"], "BLOCKED")

    def test_source_loss_or_queue_rejected(self):
        for key, value in (("rtp_lost", 1), ("queue", 12), ("errors", 1), ("pts_backwards", 1)):
            data = copy.deepcopy(self.sources)
            data["CAM-02"][-1][key] = value
            with self.subTest(key=key):
                self.assertEqual(assess(data, self.group, self.text, self.gpu)["status"], "BLOCKED")


    def test_group_batch_rate_can_exceed_single_source_rate(self):
        group = copy.deepcopy(self.group)
        for row in group:
            row["fps"] = 24
        self.assertEqual(assess(self.sources, group, self.text, self.gpu)["status"], "PASS")

    def test_signed_age_is_parsed(self):
        rows = parse_rows(
            "GROUP STATS elapsed=35.000 output=700 fps=20.000 age=-0.001 max_gap_ms=300.000\n",
            "GROUP STATS ",
        )
        self.assertEqual(rows[0]["age"], -0.001)

    def test_missing_pair_age_is_blocked_not_exception(self):
        group = copy.deepcopy(self.group)
        del group[70]["age"]
        report = assess(self.sources, group, self.text, self.gpu)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("malformed telemetry" in failure for failure in report["failures"]))

    def test_group_stall_rejected(self):
        group = copy.deepcopy(self.group)
        group[70]["output"] = group[69]["output"]
        self.assertEqual(assess(self.sources, group, self.text, self.gpu)["status"], "BLOCKED")

    def test_gpu_decoder_loss_rejected(self):
        gpu = copy.deepcopy(self.gpu)
        for row in gpu:
            row["decoder_pct"] = 0
        self.assertEqual(assess(self.sources, self.group, self.text, gpu)["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
