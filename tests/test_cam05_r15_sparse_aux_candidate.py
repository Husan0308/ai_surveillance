from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_cam05_r15_sparse_aux_candidate.py"
SPEC = importlib.util.spec_from_file_location("cam05_r15_sparse_aux", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parent_profile() -> dict:
    return {
        "primary_engine_sha256": MODULE.PRIMARY_ENGINE_SHA256,
        "primary_parser_sha256": MODULE.PRIMARY_PARSER_SHA256,
        "normal_nvdcf_input_threshold": 0.25,
        "aux_input": [896, 896],
        "aux_interval": 4,
        "aux_join_mode": "exact-key-sidecar-v1",
        "metamux_pts_tolerance_us": 0,
    }


def test_interval_change_is_exact_and_preserves_other_config() -> None:
    text = "batch-size=6\ninterval=4\ngie-unique-id=3\n"
    assert MODULE.set_aux_interval(text, 19) == "batch-size=6\ninterval=19\ngie-unique-id=3\n"


@pytest.mark.parametrize("text", ["interval=4\ninterval=4\n", "interval=3\n"])
def test_interval_change_rejects_unexpected_parent_config(text: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.set_aux_interval(text, 19)


@pytest.mark.parametrize("interval", [-1, 30])
def test_interval_change_rejects_out_of_contract_values(interval: int) -> None:
    with pytest.raises(ValueError, match="0..29"):
        MODULE.set_aux_interval("interval=4\n", interval)


def test_profile_preserves_primary_join_and_tracker_contract() -> None:
    profile = MODULE.updated_profile(parent_profile(), 19)
    assert profile["aux_interval"] == 19
    assert profile["primary_engine_sha256"] == MODULE.PRIMARY_ENGINE_SHA256
    assert profile["primary_parser_sha256"] == MODULE.PRIMARY_PARSER_SHA256
    assert profile["normal_nvdcf_input_threshold"] == 0.25
    assert profile["aux_join_mode"] == "exact-key-sidecar-v1"
    assert profile["metamux_pts_tolerance_us"] == 0


def test_profile_rejects_mutated_frozen_primary_contract() -> None:
    bad = parent_profile()
    bad["normal_nvdcf_input_threshold"] = 0.20
    with pytest.raises(ValueError, match="frozen profile"):
        MODULE.updated_profile(bad, 19)
