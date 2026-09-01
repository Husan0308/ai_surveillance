#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REBIND = "self.pair_worker.set_scores_published_callback(self.match_worker.notify)"


class Step7RuntimeWiringTests(unittest.TestCase):
    def _assert_rebound_after_last_matcher_replacement(self, relative: str) -> None:
        text = (ROOT / relative).read_text(encoding="utf-8")
        replacement = text.rfind("self.match_worker =")
        rebind = text.rfind(REBIND)
        self.assertGreaterEqual(replacement, 0, relative)
        self.assertGreater(rebind, replacement, relative)

    def test_step5_rebinds_pair_producer_to_started_matcher(self) -> None:
        self._assert_rebound_after_last_matcher_replacement(
            "services/camera_v11/step5_global_shadow_runtime_v1.py"
        )

    def test_step6_rebinds_pair_producer_to_final_started_matcher(self) -> None:
        self._assert_rebound_after_last_matcher_replacement(
            "services/camera_v11/step6_global_shadow_runtime_v1.py"
        )

    def test_pair_checker_defaults_strict_but_supports_devs_scope(self) -> None:
        text = (
            ROOT / "scripts/check_camera_v11_step4_reid_pair_scorer_v1_log.py"
        ).read_text(encoding="utf-8")
        self.assertIn('V11_STEP4_PAIR_REQUIRE_DIFFERENT_ROOM", "1"', text)
        self.assertIn("if require_different_room and different_room <= 0:", text)

    def test_step7_wrapper_explicitly_selects_devs_only_pair_scope(self) -> None:
        text = (
            ROOT / "scripts/run_camera_v11_step7_cam01_cam04_one_person_acceptance_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("export V11_STEP4_PAIR_REQUIRE_DIFFERENT_ROOM=0", text)

    def test_frozen_step123_guard(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_camera_v11_frozen_step123_guard.py")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("V11_FROZEN_STEP123_GUARD RESULT=PASS", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
