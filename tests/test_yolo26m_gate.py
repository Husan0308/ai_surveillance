import unittest
from scripts.yolo26m_person.check_gate import assess, analyze_overlaps


class DetectorGateTests(unittest.TestCase):
    def setUp(self):
        lines=['GROUP DETECTION_GRAPH mux->nvinfer(YOLO26m,FP16,batch=6,interval=0)->DeepStream-NMS(iou=0.70,conf=0.25) inference=1']
        for t in range(5,61,5):
            for i in range(1,7):
                lines.append(f'CAM-{i:02d} STATS elapsed={t} input={t*20} fps=20 age=0.02 pts_ns={t*1000000000} pts_backwards=0 pts_duplicates=0 queue=0 queue_ms=0 rtp_lost=0 rtp_late=0 errors=0 warnings=0 decoder=1')
                lines.append(f'CAM-{i:02d} DETECT frames={t*20} detections={t*2} persons={t*2} errors=0 last_detection_unix_ms=1000')
            lines.append(f'GROUP STATS elapsed={t} output={t*22} pts_ns={t*1000000000} rss_mib=1800 cpu_pct=100 dropped=0 shared_errors=0 shared_warnings=0 pts_backwards=0 pts_duplicates=0')
            lines.append(f'YOLO STATS persons={t*12} errors=0 parser_errors=0 parser_rejected=0')
        lines.extend(f'CAM-{i:02d} END hardware_decoder=1 errors=0 warnings=0' for i in range(1,7))
        lines.append('GROUP END fatal=0 shared_errors=0 shared_warnings=0')
        self.text='\n'.join(lines)

    def test_healthy_real_run_evidence_contract(self):
        self.assertEqual(assess(self.text)['status'],'PASS')

    def test_slow_source_cannot_pass_with_high_detection_counts(self):
        self.assertEqual(assess(self.text.replace('fps=20','fps=12'))['status'],'BLOCKED')

    def test_missing_camera_cannot_pass(self):
        self.assertEqual(assess('\n'.join(l for l in self.text.splitlines() if not l.startswith('CAM-06')))['status'],'BLOCKED')

    def test_parser_error_and_nonperson_count_cannot_pass(self):
        for old,new in [('parser_errors=0','parser_errors=1'),('detections=120 persons=120','detections=121 persons=120'),('rtp_lost=0','rtp_lost=1'),('queue=0','queue=12')]:
            with self.subTest(new=new):self.assertEqual(assess(self.text.replace(old,new))['status'],'BLOCKED')


    def test_overlap_analyzer_blocks_duplicate_bbox_above_nms_threshold(self):
        records = [
            {'source_id': 3, 'frame': 400, 'box': [100.0, 100.0, 200.0, 300.0],
             'confidence': 0.55, 'class_id': 0},
            {'source_id': 3, 'frame': 400, 'box': [102.0, 101.0, 198.0, 299.0],
             'confidence': 0.44, 'class_id': 0},
        ]
        report = analyze_overlaps(records, nms_threshold=0.70)
        self.assertGreater(report['pairs_over_nms_threshold'], 0)
        self.assertGreaterEqual(report['pairs_iou_gte_095'], 1)
        self.assertGreater(report['max_person_iou'], 0.95)

    def test_overlap_analyzer_allows_separate_people(self):
        records = [
            {'source_id': 0, 'frame': 10, 'box': [0.0, 0.0, 100.0, 200.0],
             'confidence': 0.80, 'class_id': 0},
            {'source_id': 0, 'frame': 10, 'box': [130.0, 0.0, 100.0, 200.0],
             'confidence': 0.75, 'class_id': 0},
        ]
        report = analyze_overlaps(records, nms_threshold=0.70)
        self.assertEqual(report['pairs_over_nms_threshold'], 0)
        self.assertEqual(report['pairs_iou_gte_090'], 0)
        self.assertEqual(report['pairs_iou_gte_095'], 0)

    def test_missing_nms_graph_evidence_is_blocked(self):
        self.assertEqual(
            assess(self.text.replace('->DeepStream-NMS(iou=0.70,conf=0.25)', ''))['status'],
            'BLOCKED',
        )

    def test_short_run_and_dirty_exit_cannot_pass(self):
        for text in [self.text.replace('elapsed=60','elapsed=50'),self.text.replace('fatal=0','fatal=1')]:
            self.assertEqual(assess(text)['status'],'BLOCKED')


if __name__=='__main__':unittest.main()
