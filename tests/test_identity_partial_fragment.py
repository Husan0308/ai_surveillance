import numpy as np
import pytest

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager, partial_bbox_continuity


def setup_handoff():
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.array([1., 0.], np.float32)
    previous = dict(camera_id='CAM-01', native_track_id=1, frame=188,
                    bbox=[822.375, 423.75, 926.625, 585.75],
                    world=[18.09858, -32.11613])
    gid = manager.confirm_new(previous, vector, 'test_initial')
    peer = dict(camera_id='CAM-04', native_track_id=3, frame=191,
                bbox=[1081.125, 466.875, 1261.875, 700.125], world=[21.01644, -30.12671])
    manager.accept_existing(gid, peer, vector, {'reason': 'test_peer'})
    current = dict(camera_id='CAM-01', native_track_id=17, frame=191,
                   bbox=[821.625, 424.125, 927.375, 712.875], world=[23.58985, -25.91898],
                   pending_good_crops=2)
    return manager, gid, vector, previous, peer, current


def test_retained_partial_box_projection_jump_uses_existing_peer_gates():
    manager, gid, vector, _, peer, current = setup_handoff()
    result, evidence = manager.resolve_existing(current, vector, [{**peer, 'global_person_id_num': gid}])
    assert result == gid
    assert evidence['reason'] == 'cross_camera_reid_geometry'
    assert manager.next_number == 2


@pytest.mark.parametrize('missing', ['peer', 'current_camera_gallery', 'peer_gallery', 'two_crops', 'time', 'world', 'image'])
def test_independent_support_is_required(missing):
    manager, gid, vector, _, peer, current = setup_handoff()
    assignments = [{**peer, 'global_person_id_num': gid}]
    if missing == 'peer':
        assignments = []
    elif missing in ('current_camera_gallery', 'peer_gallery'):
        camera = 'CAM-01' if missing == 'current_camera_gallery' else 'CAM-04'
        manager.identities[gid].galleries[camera].clear()
    elif missing == 'two_crops':
        current['pending_good_crops'] = 1
    elif missing == 'time':
        assignments[0]['frame'] = 189
    elif missing == 'world':
        assignments[0]['world'] = [100., 100.]
    elif missing == 'image':
        current['bbox'] = [600., 400., 710., 690.]
    result, _ = manager.resolve_existing(current, vector, assignments)
    assert result is None
    assert manager.next_number == 2


def test_distinct_current_same_camera_person_cannot_be_overridden():
    manager, gid, vector, previous, peer, current = setup_handoff()
    assignments = [{**peer, 'global_person_id_num': gid},
                   {**previous, 'frame': 191, 'bbox': [600., 400., 710., 690.],
                    'global_person_id_num': gid}]
    result, _ = manager.resolve_existing(current, vector, assignments)
    assert result is None


def test_coexisting_nested_fragment_requires_a_current_independent_peer():
    manager, gid, vector, previous, peer, current = setup_handoff()
    assignments = [{**peer, 'global_person_id_num': gid},
                   {**previous, 'frame': 191, 'global_person_id_num': gid}]
    result, evidence = manager.resolve_existing(current, vector, assignments)
    assert result == gid
    assert evidence['same_camera_fragment']
    result, _ = manager.resolve_existing(current, vector, assignments[1:])
    assert result is None


def test_head_fragment_at_2099_then_coexisting_torso_at_2103():
    manager, gid, vector, _, peer, current = setup_handoff()
    previous = dict(camera_id='CAM-01', native_track_id=128, frame=2097,
                    bbox=[1426.125, 481.5, 1615.875, 790.5], world=[15.99041, -9.56015])
    manager.accept_existing(gid, previous, None, {'reason': 'test_fragment'})
    peer.update(frame=2099, world=[12.62782, -12.44418])
    current.update(frame=2099, native_track_id=130, bbox=[1540.5, 474., 1615.5, 622.5],
                   world=[6.74031, -11.70693])
    assignments = [{**peer, 'global_person_id_num': gid}]
    result, _ = manager.resolve_existing(current, vector, assignments)
    assert result == gid
    manager.accept_existing(gid, current, vector, {'reason': 'test_partial'})
    previous = {**current, 'frame': 2103, 'bbox': [1543.5, 470.25, 1606.5, 620.25],
                'world': [6.65711, -11.84135], 'global_person_id_num': gid}
    peer.update(frame=2103, world=[12.06312, -11.52708])
    current.update(frame=2103, native_track_id=131, bbox=[1469.625, 469.5, 1608.375, 688.5],
                   world=[11.06562, -11.15523])
    result, evidence = manager.resolve_existing(current, vector,
        [{**peer, 'global_person_id_num': gid}, previous])
    assert result == gid
    assert evidence['same_camera_fragment']


def test_short_black_shirt_fragment_does_not_require_lower_reid_thresholds():
    manager, gid, _, _, peer, current = setup_handoff()
    previous = dict(camera_id='CAM-01', native_track_id=148, frame=2247,
                    bbox=[1551.75, 489.75, 1652.25, 627.75], world=[6.62567, -10.98211])
    manager.accept_existing(gid, previous, None, {'reason': 'test_fragment'})
    peer.update(frame=2250, world=[13.24937, -6.91557])
    current.update(frame=2250, bbox=[1514.625, 490.5, 1653.375, 693.], world=[10.49705, -10.04592])
    vector = np.array([.62, np.sqrt(1 - .62 ** 2)], np.float32)
    result, evidence = manager.resolve_existing(current, vector, [{**peer, 'global_person_id_num': gid}])
    assert result == gid
    assert evidence['same_camera_fragment']


def test_non_nested_or_invalid_boxes_do_not_supply_partial_body_evidence():
    assert partial_bbox_continuity([10, 20, 60, 100], [9, 19, 61, 160])
    assert not partial_bbox_continuity([10, 20, 60, 100], [20, 40, 70, 120])
    assert not partial_bbox_continuity([10, 20, 10, 100], [9, 19, 61, 160])
    assert not partial_bbox_continuity([10, 20, 60, 100], [9, float('nan'), 61, 160])
