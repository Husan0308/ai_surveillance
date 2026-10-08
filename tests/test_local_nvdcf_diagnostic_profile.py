from __future__ import annotations

from pathlib import Path

from scripts.run_local_nvdcf_parity_replay import (
    LOCAL_CONFIG,
    V13_CONFIG,
    enable_shadow_dump,
    make_diagnostic_profile,
)


def test_regular_bbox_kf_changes_only_state_estimator_type():
    source = LOCAL_CONFIG.read_bytes().replace(b"stateEstimatorType: 2", b"stateEstimatorType: 1")
    result = make_diagnostic_profile(LOCAL_CONFIG, 2, False)
    assert source.count(b"stateEstimatorType: 1") == 1
    assert result.count(b"stateEstimatorType: 2") == 1
    assert result.replace(b"stateEstimatorType: 2", b"stateEstimatorType: 1") == source


def test_shadow_diagnostic_only_enables_shadow_output():
    source = LOCAL_CONFIG.read_bytes()
    result = make_diagnostic_profile(LOCAL_CONFIG, None, True)
    assert result == source


def test_type2_shadow_candidate_combines_only_the_two_diagnostic_deltas():
    source = LOCAL_CONFIG.read_bytes().replace(b"stateEstimatorType: 2", b"stateEstimatorType: 1")
    source = source.replace(b"outputShadowTracks: 1", b"outputShadowTracks: 0")
    result = make_diagnostic_profile(LOCAL_CONFIG, 2, True)
    assert result.count(b"stateEstimatorType: 2") == 1
    assert result.count(b"outputShadowTracks: 1") == 1
    normalized = result.replace(b"stateEstimatorType: 2", b"stateEstimatorType: 1")
    normalized = normalized.replace(b"outputShadowTracks: 1", b"outputShadowTracks: 0")
    assert normalized == source


def test_shadow_dump_uses_the_mounted_probe_directory(tmp_path: Path):
    config = tmp_path / "config_deepstream.txt"
    config.write_text("[application]\nenable-perf-measurement=1\n[source0]\ntype=3\n")
    enable_shadow_dump(config)
    assert "shadow-track-output-dir=/workspace/experiments/logs/probe/shadow" in config.read_text()


def test_v13_shadow_probe_changes_only_staged_shadow_output_flag():
    source = V13_CONFIG.read_bytes()
    result = make_diagnostic_profile(V13_CONFIG, None, True)
    assert result.replace(b"outputShadowTracks: 1", b"outputShadowTracks: 0") == source
