from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "services/frontend/core_v1/operator_dashboard_people_roster.py"


class PeopleRosterUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = UI.read_text(encoding="utf-8")

    def test_roster_is_exactly_ten_slots(self):
        self.assertIn("MAX_WORKERS = 10", self.source)
        self.assertIn("for slot in range(MAX_WORKERS)", self.source)
        self.assertIn("slot // 5, slot % 5", self.source)
        self.assertIn("maximum 10 profiles", self.source)

    def test_persons_page_has_integrated_enrollment_and_worker_profile(self):
        for required in (
            "class WorkerRosterPage",
            "class WorkerCard",
            "class WorkerEnrollmentDialog",
            "WORKER PROFILE",
            "WORKER ROSTER",
            "＋ Add Worker",
            "🗑 Delete Worker",
        ):
            self.assertIn(required, self.source)

    def test_worker_name_and_job_are_free_text_inputs(self):
        self.assertIn("self.name = QLineEdit()", self.source)
        self.assertIn("self.role = QLineEdit()", self.source)
        self.assertIn("Full Name  *", self.source)
        self.assertIn("Job / Role  *", self.source)
        self.assertIn('"department": role', self.source)
        self.assertNotIn('self.dept.addItems(["Security"', self.source)

    def test_enrollment_keeps_ten_quality_gated_samples(self):
        self.assertIn("ENROLLMENT_SAMPLES = 10", self.source)
        self.assertIn("Face samples", self.source)
        self.assertIn("Waiting for a better face", self.source)
        self.assertIn("Follow the angle prompt", self.source)
        self.assertIn("/faces/enrollment/sample/", self.source)
        self.assertIn('"/faces/enrollment/commit"', self.source)

    def test_worker_profile_uses_real_face_db_and_track_state(self):
        for required in (
            "person.get(\"has_avatar\")",
            "person.get('recognitions')",
            "person.get('last_seen')",
            "person.get('created_at')",
            "row.get(\"person_id\")",
            'face._json_request("DELETE", f"/faces/people/{person_id}"',
        ):
            self.assertIn(required, self.source)

    def test_existing_cuda_face_stack_is_wrapped_not_reimplemented(self):
        self.assertIn("from . import operator_dashboard_face as face", self.source)
        self.assertIn("from . import operator_dashboard_face_cuda as cuda", self.source)
        self.assertIn("face.PersonManagementPage = WorkerRosterPage", self.source)
        self.assertIn("face.EnrollmentPage = EnrollmentLauncherPage", self.source)
        self.assertIn("return cuda.run()", self.source)


if __name__ == "__main__":
    unittest.main()
