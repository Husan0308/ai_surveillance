import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from services.ml_service.snapshots.manager import UnknownSnapshotManager

class SnapshotRaceTests(unittest.TestCase):
 def test_disappearing_snapshot_is_ignored_individually(self):
  with tempfile.TemporaryDirectory() as root:
   manager=UnknownSnapshotManager(root);present=Path(root)/"present.jpg";gone=Path(root)/"gone.jpg";present.write_bytes(b"1234");gone.write_bytes(b"123456")
   real_stat=Path.stat
   def racing_stat(path,*args,**kwargs):
    if Path(path)==gone:
     gone.unlink(missing_ok=True);raise FileNotFoundError(str(gone))
    return real_stat(path,*args,**kwargs)
   with patch.object(Path,"stat",racing_stat):self.assertEqual(manager.disk_usage(),4)
   manager.close()

if __name__=="__main__":unittest.main()
