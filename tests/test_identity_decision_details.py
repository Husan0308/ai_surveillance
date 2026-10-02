from collections import deque

import numpy as np

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager


def observation(native, frame, world, bbox):
    return dict(camera_id='CAM-01', native_track_id=native, frame=frame,
                world=world, bbox=bbox)


def freeze(value):
    if isinstance(value, dict):
        return sorted([(repr(key), freeze(item)) for key, item in value.items()])
    if isinstance(value, (list, tuple, deque)):
        return [freeze(item) for item in value]
    if isinstance(value, set):
        return sorted(map(repr, value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, '__dict__'):
        return freeze(vars(value))
    return value


def test_detailed_trace_reports_actual_conflict_without_changing_decisions(monkeypatch):
    monkeypatch.delenv('MV3DT_IDENTITY_DECISION_DETAILS', raising=False)
    manager = GlobalIdentityManager(np.empty((0, 2), np.float32))
    vector = np.array([1., 0.], np.float32)
    previous = observation(1, 100, [0., 0.], [10., 10., 100., 200.])
    manager.confirm_new(previous, vector, 'test')
    candidate = observation(11, 103, [6., 0.], [10., 10., 100., 320.])
    before = freeze(manager.__dict__)
    ordinary = manager.candidate_trace(candidate, vector, [])
    assert freeze(manager.__dict__) == before
    assert ordinary[0]['exact_rejection_reason'] == 'same_camera_simultaneous'
    assert 'diagnostic_context' not in ordinary[0]
    manager.diagnostic_details = True
    before = freeze(manager.__dict__)
    detailed = manager.candidate_trace(candidate, vector, [])
    assert freeze(manager.__dict__) == before
    context = detailed[0].pop('diagnostic_context')
    assert detailed == ordinary
    assert context['retained_same_camera']['native_track_id'] == 1
    assert context['retained_frame_gap'] == 3
    assert context['retained_source_time_delta_ms'] == 150.
    assert context['retained_world_distance'] == 6.
    assert context['current_assignments'] == []
    # Diagnostic values must not alias and mutate retained observations.
    context['retained_same_camera']['world'][0] = 999
    assert manager.identities[1].last_by_camera['CAM-01']['world'][0] == 0.


def test_details_can_be_enabled_and_disabled_without_a_policy_change(monkeypatch):
    monkeypatch.setenv('MV3DT_IDENTITY_DECISION_DETAILS', '1')
    assert GlobalIdentityManager(np.empty((0, 2))).diagnostic_details
    assert not GlobalIdentityManager(np.empty((0, 2)), diagnostic_details=False).diagnostic_details
