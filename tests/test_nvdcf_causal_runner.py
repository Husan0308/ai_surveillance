import pytest

from scripts.dev_room_mv3dt.run_nvdcf_causal_experiment import verify_prefix


def test_stream_copy_future_reference_frame_is_rejected_even_with_equal_counts():
    source = ['0, 517, md5-a', '0, 518, md5-b', '0, 519, md5-c']
    truncated = ['0, 517, md5-a', '0, 518, md5-b', '0, 522, md5-d']
    with pytest.raises(ValueError, match='decoded frames or timestamps'):
        verify_prefix(source, truncated, 3)


def test_missing_or_retimed_frames_cannot_satisfy_prefix_provenance():
    with pytest.raises(ValueError):
        verify_prefix(['0, 1, md5-a'], [], 1)
    with pytest.raises(ValueError):
        verify_prefix(['0, 1, md5-a'], ['0, 2, md5-a'], 1)
    verify_prefix(['0, 1, md5-a'], ['0, 1, md5-a'], 1)
