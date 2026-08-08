import ast,unittest
import os,tempfile
from pathlib import Path
from shared.schemas.messages import EnrollmentStartCommand,MLSettingsChangedCommand
from shared.settings import ServiceSettings
class FunctionalBoundaryTests(unittest.TestCase):
    def test_frontend_has_no_service_manager_or_direct_layers(self):
        text="\n".join(path.read_text() for path in Path("services/frontend").rglob("*.py"))
        for forbidden in ("sys.sm","service_manager","from backend","import backend","services.ml_service","api_service.repositories"):
            self.assertNotIn(forbidden,text)
    def test_message_validation(self):
        command=EnrollmentStartCommand(name="Husan",camera_id="CAM-01")
        self.assertEqual(command.type,"enrollment.start")
        with self.assertRaises(Exception):EnrollmentStartCommand(name="",camera_id="bad id")
        with self.assertRaises(Exception):MLSettingsChangedCommand(settings={"frame":b"bad"})
    def test_sqlite_initializer_exists(self):self.assertTrue(Path("services/api_service/database.py").exists())
    def test_database_path_is_absolute_and_cwd_independent(self):
        original=os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:path=ServiceSettings.from_env().database_path
            finally:os.chdir(original)
        self.assertTrue(Path(path).is_absolute());self.assertEqual(Path(path).name,"surveillance.db")
