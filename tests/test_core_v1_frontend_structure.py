from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "services" / "frontend" / "core_v1"


class FrontendStructureTests(unittest.TestCase):
    def test_dashboard_is_valid_python_without_ui_monkey_patch(self):
        dashboard_path = FRONTEND / "dashboard.py"
        source = dashboard_path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(dashboard_path))

        main_source = (FRONTEND / "main.py").read_text(encoding="utf-8")
        ast.parse(main_source, filename=str(FRONTEND / "main.py"))
        self.assertNotIn("ui_polish", main_source)
        self.assertNotIn("install(dashboard)", main_source)
        self.assertFalse((FRONTEND / "ui_polish.py").exists())

    def test_camera_view_is_integrated_16_by_9_contain_layout(self):
        source = (FRONTEND / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("fit = min(tw / iw, th / ih)", source)
        self.assertIn("tile_h = int(round(tile_w * 9.0 / 16.0))", source)
        self.assertIn("QLinearGradient", source)
        self.assertIn("_people_text", source)
        self.assertNotIn("cameraHeader", source)
        self.assertNotIn("cameraFooter", source)

    def test_events_use_room_sessions_not_local_track_seen_cache(self):
        source = (FRONTEND / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn('"room_sessions": {}', source)
        self.assertIn('self._json(connection, "/room-sessions")', source)
        self.assertNotIn("self._seen", source)
        self.assertNotIn("def _observe(self, reid_payload)", source)
        self.assertIn('"Entered"', source)
        self.assertIn("ROOM_TITLES", source)


if __name__ == "__main__":
    unittest.main()
