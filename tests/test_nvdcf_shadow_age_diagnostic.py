from __future__ import annotations

from scripts.build_nvdcf_shadow_age_diagnostic import instrument_shadow
from scripts.run_local_nvdcf_parity_replay import V13_CONFIG, make_diagnostic_profile


def test_shadow_logger_reads_only_public_nvds_shadow_metadata():
    source = instrument_shadow(
        (V13_CONFIG.parents[2] / "services/mv3dt_room/native/bbox_correction.c").read_text()
    )
    assert "NVDS_TRACKER_SHADOW_LIST_META" in source
    assert "sample->age" in source
    assert "sample->trackerState" in source
    assert "sample->confidence" in source
    assert "sample->visibility" in source
    assert "sample->tBbox" in source
    assert "PN263_SHADOW_META_LOG" in source


def test_diagnostic_v13_overlay_does_not_edit_frozen_tracker_file():
    original = V13_CONFIG.read_bytes()
    diagnostic = make_diagnostic_profile(V13_CONFIG, None, True)
    assert diagnostic.replace(b"outputShadowTracks: 1", b"outputShadowTracks: 0") == original
