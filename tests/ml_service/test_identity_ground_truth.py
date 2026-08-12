import unittest
from tools.identity_ground_truth import evaluate

class IdentityGroundTruthTests(unittest.TestCase):
    def test_detects_false_split_and_false_merge(self):
        report=evaluate([
            {"subject":"A","camera_id":"C1","global_id":"UNK 1"},
            {"subject":"A","camera_id":"C2","global_id":"UNK 2"},
            {"subject":"B","camera_id":"C3","global_id":"UNK 1"},
        ])
        self.assertEqual(report["false_splits"],{"A":["UNK 1","UNK 2"]})
        self.assertEqual(report["false_merges"],{"UNK 1":["A","B"]});self.assertFalse(report["pass"])

    def test_clean_transition_passes(self):
        report=evaluate([
            {"subject":"A","camera_id":"C1","global_id":"UNK 1"},
            {"subject":"A","camera_id":"C2","global_id":"UNK 1"},
            {"subject":"B","camera_id":"C1","global_id":"UNK 2"},
        ])
        self.assertTrue(report["pass"]);self.assertEqual(report["camera_transitions"],1)

if __name__=="__main__":unittest.main()
