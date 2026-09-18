import copy
import unittest
from scripts.cam01_validation.check_stability import assess


class StabilityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.text='memory:NVMM nvstreammux(batch=1,\nCAM-01 END hardware_decoder=1 fatal=0'
        self.rows=[];self.gpu=[]
        for t in range(30, 661, 5):
            self.rows.append(dict(elapsed=t,decoded=t*20,output=t*20,pts_ns=t*1000000000,
                fps=20,age=0.05,max_gap_ms=350,queue=0,queue_ms=0,rss_mib=280,cpu_pct=5,
                errors=0,warnings=0,recoveries=0,dropped=0,pts_backwards=0,pts_duplicates=0,
                rtp_lost=0,rtp_late=0))
            self.gpu.append(dict(elapsed=t,memory_used_mib=321,gpu_pct=1,decoder_pct=4,encoder_pct=4))

    def test_complete_healthy_evidence(self):
        self.assertEqual(assess(self.rows,self.text,self.gpu)['status'],'PASS')

    def test_first_frame_is_not_stability(self):
        self.assertEqual(assess(self.rows[:2],self.text,self.gpu[:2])['status'],'BLOCKED')

    def test_frozen_counter_rejected_even_if_fps_claims_success(self):
        self.rows[70]['output']=self.rows[69]['output']
        self.assertEqual(assess(self.rows,self.text,self.gpu)['status'],'BLOCKED')

    def test_missing_telemetry_cannot_pass(self):
        del self.rows[30:45]
        self.assertEqual(assess(self.rows,self.text,self.gpu)['status'],'BLOCKED')

    def test_growth_drops_errors_and_missing_end_rejected(self):
        for key, value in [('rss_mib',400),('cpu_pct',50),('errors',1),('dropped',10),('pts_duplicates',1)]:
            rows=copy.deepcopy(self.rows)
            for row in rows[-12:]:row[key]=value
            with self.subTest(key=key):
                self.assertEqual(assess(rows,self.text,self.gpu)['status'],'BLOCKED')
        self.assertEqual(assess(self.rows,'memory:NVMM nvstreammux(batch=1,',self.gpu)['status'],'BLOCKED')

    def test_gpu_loss_or_growth_rejected(self):
        for key, value in [('memory_used_mib',1000),('decoder_pct',0)]:
            gpu=copy.deepcopy(self.gpu)
            for row in gpu[-12:] if key=='memory_used_mib' else gpu:row[key]=value
            self.assertEqual(assess(self.rows,self.text,gpu)['status'],'BLOCKED')


if __name__=='__main__':unittest.main()
