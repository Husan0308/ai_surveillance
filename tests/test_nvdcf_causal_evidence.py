from collections import defaultdict

from scripts.dev_room_mv3dt.report_nvdcf_causal_evidence import summarize_window, xywh


def test_adjacent_losses_of_different_targets_do_not_shorten_gate():
    audit = defaultdict(dict)
    details = []
    for frame in range(274, 300):
        audit['CAM-04', frame] = {'pgie': [{'confidence': .91}, {'confidence': .90}],
                                 'tracker': [{'confidence': .91}]}
        details.append({'proposal': {'camera': 'CAM-04', 'frame': frame},
                        'current_frame_lifecycle': [{'native': '9' if frame < 282 else '3'}]})
    result = summarize_window('CAM-04', {'start': 274, 'end': 299, 'frames': 26}, details, audit)
    assert result['duration_frames'] == len(result['frames']) == 26
    assert all(row['detector_object_count'] == 2 and row['tracker_object_count'] == 1
               for row in result['frames'])
    assert result['suppression'] == 'NOT_EXPORTED_BY_SDK'


def test_identity_boxes_are_xyxy_and_not_tracker_xywh():
    assert xywh([10, 20, 40, 80]) == [10, 20, 30, 60]
