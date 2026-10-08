from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_cam05_r19_sparse_exact_join_candidate.py"
SPEC = importlib.util.spec_from_file_location("cam05_r19_sparse_join", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_r19_is_an_isolated_candidate_of_the_reviewed_r18_profile() -> None:
    source = SCRIPT.read_text()
    assert 'PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r18-aux768-preprocess-sparse20"' in source
    assert "GATE_PERIOD = 20" in source
    assert "PRIMARY_ENGINE_SHA256 = \"e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16\"" in source
    assert "AUX_ENGINE_SHA256 = \"84832420e6e3ad68e1664b2ceb04b38474459960990c00d76987ca8bbda153be\"" in source


def test_sparse_skip_completes_same_source_frame_pts_key_with_empty_result() -> None:
    source = (ROOT / "services/camera_v11/native/six_camera_v13/detection.hpp").read_text()
    start = source.index("static GstPadProbeReturn auxiliary_sparse_preprocess_gate")
    end = source.index("#endif", start)
    gate = source[start:end]
    publish = gate.index("auxiliary_exact_join.publish(key, std::vector<AuxiliaryProposal>{})")
    drop = gate.index("return GST_PAD_PROBE_DROP", publish)
    assert "frame->source_id, frame->frame_num, frame->buf_pts" in gate
    assert publish < drop
    assert "auxiliary_sparse_skip_join_errors" in gate
    assert "auxiliary_join_before_tracker" not in gate
