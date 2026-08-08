import unittest
from services.ml_service.identity.reid_extractor import ReIDExtractor

class SharedAppearance:
    def __init__(self,model):self.model=model
    def extract_batch(self,crops):return [None]*len(crops),{}

class ReIDOwnershipTests(unittest.TestCase):
    def test_tracking_and_identity_can_share_exact_model_instance(self):
        model=object();appearance=SharedAppearance(model)
        identity_reid=ReIDExtractor(appearance)
        self.assertIs(appearance.model,model)
        self.assertEqual(identity_reid.model_identity,id(model))

if __name__=="__main__":unittest.main()
