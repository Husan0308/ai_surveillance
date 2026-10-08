from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cam05_candidate_direct_replay.py"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13/lowconf_verifier.hpp"


def _load_runner():
    spec = importlib.util.spec_from_file_location("cam05_candidate_replay", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_crop_is_opt_in_and_exactly_named() -> None:
    runner = _load_runner()
    assert runner.candidate_centered_crop_environment(False) == []
    assert runner.candidate_centered_crop_environment(True) == [
        "V11_CONTEXT_CANDIDATE_CROP=1"
    ]


def test_explicit_640_verifier_uses_its_batch_one_config_contract() -> None:
    runner = _load_runner()
    assert runner.verifier_runtime_contract({}, "yolo26m_context_verifier.txt") == (
        "YOLO26m-640-FP16", 1
    )


def test_both_verifier_crop_paths_use_one_selected_policy() -> None:
    source = SOURCE.read_text()
    assert "static ShadowContextCrop lowconf_compute_verifier_crop(" in source
    assert "return shadow_context_compute_candidate_crop" in source
    assert "return shadow_context_compute_crop" in source
    assert source.count("lowconf_compute_verifier_crop(width, height,") == 2
    assert "candidate-centered-zoom-v1" in source


def test_candidate_builder_preserves_frozen_assets_and_stages_under_runtime() -> None:
    builder = (ROOT / "scripts/build_cam05_candidate_centered_crop_r43.py").read_text()
    assert 'OUT = ROOT / ".runtime/sixcam-cam05-final-20261008-r43-candidate-centered-crop"' in builder
    assert 'if not OUT.is_relative_to(ROOT / ".runtime")' in builder
    assert 'PRIMARY_ENGINE = "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"' in builder
    assert 'LOCAL_TRACKER = "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"' in builder
    assert 'V13_REFERENCE = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"' in builder
