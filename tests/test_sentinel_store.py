from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import services.camera_v2.sentinel_store as store_module


class SentinelStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._originals = (
            store_module.DATA_DIR,
            store_module.DB_PATH,
            store_module.PEOPLE_DIR,
            store_module.EVENTS_DIR,
        )
        store_module.DATA_DIR = root / "sentinel"
        store_module.DB_PATH = store_module.DATA_DIR / "sentinel.db"
        store_module.PEOPLE_DIR = store_module.DATA_DIR / "people"
        store_module.EVENTS_DIR = store_module.DATA_DIR / "events"
        self.store = store_module.SentinelStore()

    def tearDown(self) -> None:
        (
            store_module.DATA_DIR,
            store_module.DB_PATH,
            store_module.PEOPLE_DIR,
            store_module.EVENTS_DIR,
        ) = self._originals
        self.tmp.cleanup()

    def _images(self) -> list[str]:
        source_dir = Path(self.tmp.name) / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(10):
            path = source_dir / f"face_{index:02d}.jpg"
            path.write_bytes(f"image-{index}".encode())
            paths.append(str(path))
        return paths

    def test_enroll_persists_profile_and_deactivate_hides_it(self) -> None:
        person = self.store.enroll_person(
            name="Test Worker",
            role="Engineer",
            department="Vision",
            notes="test",
            image_paths=self._images(),
            profile_index=3,
        )
        self.assertTrue(person["id"].startswith("P-"))
        self.assertEqual(person["name"], "Test Worker")
        self.assertTrue(Path(person["profile_photo"]).is_file())
        person_dir = Path(person["profile_photo"]).parent
        self.assertEqual(len(list(person_dir.glob("face_*"))), 10)
        self.assertEqual(len(self.store.list_people()), 1)

        self.store.deactivate_person(person["id"])
        self.assertEqual(self.store.list_people(), [])
        self.assertIsNotNone(self.store.get_person(person["id"]))

    def test_event_dedup_suppresses_repeat_and_keeps_snapshot(self) -> None:
        first, inserted = self.store.record_event_once(
            event_type="entry",
            local_id="CAM-01:7",
            camera_id="CAM-01",
            room="Entrance",
            snapshot_bytes=b"jpeg-test",
            dedup_seconds=15,
            created_at=100.0,
        )
        self.assertTrue(inserted)
        self.assertTrue(Path(first["snapshot_path"]).is_file())

        duplicate, inserted = self.store.record_event_once(
            event_type="entry",
            local_id="CAM-01:7",
            camera_id="CAM-01",
            room="Entrance",
            snapshot_bytes=b"different",
            dedup_seconds=15,
            created_at=110.0,
        )
        self.assertFalse(inserted)
        self.assertEqual(duplicate["id"], first["id"])
        self.assertEqual(len(self.store.list_events()), 1)

        _, inserted = self.store.record_event_once(
            event_type="entry",
            local_id="CAM-01:7",
            camera_id="CAM-01",
            room="Entrance",
            dedup_seconds=15,
            created_at=116.0,
        )
        self.assertTrue(inserted)
        self.assertEqual(len(self.store.list_events()), 2)


if __name__ == "__main__":
    unittest.main()
