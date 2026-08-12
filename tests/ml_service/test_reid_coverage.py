import unittest
from services.ml_service.identity.coverage import ReIDTaskCoverage
class ReIDTaskCoverageTests(unittest.TestCase):
 def test_execution_and_decision(self):
  c=ReIDTaskCoverage();c.update("CAM-01",7,reid_eligible=True);c.submitted("CAM-01",7);c.completed("CAM-01",7,.72,"quality_ok",True,True);s=c.snapshot({("CAM-01","7"):{"candidate_count":2,"decision":"PENDING"}});self.assertEqual((s["eligible"],s["submitted"]),(1,1));self.assertEqual(s["tracks"][0]["independent_evidence_count"],1)
 def test_low_quality_stays_retryable(self):
  c=ReIDTaskCoverage();c.update("CAM-05",3,reid_eligible=True);c.submitted("CAM-05",3);c.completed("CAM-05",3,.31,"crop_too_blurry",False,True);item=c.snapshot()["tracks"][0];self.assertEqual(item["independent_evidence_count"],0);self.assertEqual(item["decision"],"RETRY")
if __name__=="__main__":unittest.main()
