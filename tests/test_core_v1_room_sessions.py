from __future__ import annotations

import unittest

from services.ml_service.core_v1.room_sessions import RoomVisitSessionManager


CAMERA_ROOMS = {
    "CAM-01": "ROOM-1",
    "CAM-04": "ROOM-1",
    "CAM-02": "ROOM-2",
    "CAM-05": "ROOM-2",
    "CAM-03": "ROOM-3",
    "CAM-06": "ROOM-3",
}


def reid_state(*items):
    cameras = {}
    for camera_id, global_id, last_seen in items:
        cameras.setdefault(camera_id, []).append({
            "global_id": global_id,
            "last_seen": float(last_seen),
        })
    return {"cameras": cameras}


class RoomVisitSessionManagerTests(unittest.TestCase):
    def make_manager(self):
        return RoomVisitSessionManager(
            {
                "enabled": True,
                "enter_confirm_sec": 0.5,
                "inactive_timeout_sec": 2.0,
                "pending_timeout_sec": 1.5,
                "max_events": 50,
                "max_recent_sessions": 50,
            },
            camera_rooms=CAMERA_ROOMS,
        )

    def test_same_room_pair_creates_one_visit(self):
        manager = self.make_manager()
        manager.update(reid_state(("CAM-01", "Unknown_001", 10.0)), now=10.0)
        manager.update(
            reid_state(
                ("CAM-01", "Unknown_001", 10.6),
                ("CAM-04", "Unknown_001", 10.6),
            ),
            now=10.6,
        )
        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["events"]), 1)
        self.assertEqual(len(snapshot["active_sessions"]), 1)
        self.assertEqual(snapshot["events"][0]["room_id"], "ROOM-1")
        self.assertEqual(snapshot["events"][0]["global_id"], "Unknown_001")
        self.assertEqual(snapshot["active_sessions"][0]["cameras"], ["CAM-01", "CAM-04"])

    def test_repeated_same_room_observations_are_suppressed(self):
        manager = self.make_manager()
        manager.update(reid_state(("CAM-01", "Unknown_001", 20.0)), now=20.0)
        manager.update(reid_state(("CAM-01", "Unknown_001", 20.6)), now=20.6)
        manager.update(reid_state(("CAM-04", "Unknown_001", 21.0)), now=21.0)
        manager.update(
            reid_state(
                ("CAM-01", "Unknown_001", 21.4),
                ("CAM-04", "Unknown_001", 21.4),
            ),
            now=21.4,
        )
        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["events"]), 1)
        self.assertEqual(len(snapshot["active_sessions"]), 1)
        self.assertGreaterEqual(snapshot["metrics"]["suppressed_observations"], 2)

    def test_room_transition_closes_old_and_emits_new_enter(self):
        manager = self.make_manager()
        manager.update(reid_state(("CAM-01", "Unknown_007", 30.0)), now=30.0)
        manager.update(reid_state(("CAM-01", "Unknown_007", 30.6)), now=30.6)

        manager.update(reid_state(("CAM-02", "Unknown_007", 31.0)), now=31.0)
        manager.update(reid_state(("CAM-02", "Unknown_007", 31.6)), now=31.6)

        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["events"]), 2)
        self.assertEqual(snapshot["events"][0]["room_id"], "ROOM-2")
        self.assertEqual(snapshot["events"][1]["room_id"], "ROOM-1")
        self.assertEqual(len(snapshot["active_sessions"]), 1)
        self.assertEqual(snapshot["active_sessions"][0]["room_id"], "ROOM-2")
        self.assertEqual(len(snapshot["recent_sessions"]), 1)
        self.assertEqual(snapshot["recent_sessions"][0]["room_id"], "ROOM-1")
        self.assertEqual(snapshot["metrics"]["room_changes"], 1)

    def test_inactive_visit_closes_without_extra_event(self):
        manager = self.make_manager()
        manager.update(reid_state(("CAM-03", "Unknown_003", 40.0)), now=40.0)
        manager.update(reid_state(("CAM-03", "Unknown_003", 40.6)), now=40.6)
        manager.update(reid_state(), now=42.7)

        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["events"]), 1)
        self.assertEqual(len(snapshot["active_sessions"]), 0)
        self.assertEqual(len(snapshot["recent_sessions"]), 1)
        self.assertEqual(snapshot["metrics"]["closed"], 1)

    def test_two_people_in_same_room_remain_separate_sessions(self):
        manager = self.make_manager()
        manager.update(
            reid_state(
                ("CAM-01", "Unknown_001", 50.0),
                ("CAM-04", "Unknown_002", 50.0),
            ),
            now=50.0,
        )
        manager.update(
            reid_state(
                ("CAM-01", "Unknown_001", 50.6),
                ("CAM-04", "Unknown_002", 50.6),
            ),
            now=50.6,
        )

        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["events"]), 2)
        self.assertEqual(len(snapshot["active_sessions"]), 2)
        self.assertEqual(
            {item["global_id"] for item in snapshot["active_sessions"]},
            {"Unknown_001", "Unknown_002"},
        )


if __name__ == "__main__":
    unittest.main()
