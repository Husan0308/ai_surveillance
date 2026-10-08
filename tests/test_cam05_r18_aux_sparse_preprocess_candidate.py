from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_cam05_r18_aux768_sparse_preprocess_candidate.py"
SPEC = importlib.util.spec_from_file_location("cam05_r18_sparse_preprocess", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_auxiliary_interval_changes_to_match_preprocessor_gate() -> None:
    text = "batch-size=6\ninterval=19\ngie-unique-id=3\n"
    assert MODULE.set_aux_interval(text, 0) == "batch-size=6\ninterval=0\ngie-unique-id=3\n"


@pytest.mark.parametrize("text", ["interval=19\ninterval=19\n", "interval=4\n"])
def test_interval_change_rejects_unexpected_aux_config(text: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.set_aux_interval(text, 0)


def test_profile_pins_primary_and_one_in_twenty_auxiliary_contract() -> None:
    source = SCRIPT.read_text()
    assert "PRIMARY_ENGINE_SHA256 = \"e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16\"" in source
    assert "GATE_PERIOD = 20" in source
    assert "GATE_PHASE = 0" in source
    assert "interval=0" in source
    assert '"primary_input_unchanged": True' in source
    assert '"exact_sidecar_join_unchanged": True' in source
