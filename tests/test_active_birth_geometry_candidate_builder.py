from scripts.build_active_birth_geometry_recheck_candidate import runtime_description


def test_candidate_runtime_description_uses_effective_768_interval_19_profile() -> None:
    description = runtime_description({
        "aux_input": [768, 768],
        "aux_interval": 19,
        "aux_config": "config_infer_aux_yolo26m_768.txt",
    })

    assert "YOLO26m 768 auxiliary batch-6 interval=19" in description
    assert "config_infer_aux_yolo26m_768.txt" in description
    assert "896 auxiliary interval=2" not in description


def test_candidate_runtime_description_fails_closed_on_incomplete_profile() -> None:
    try:
        runtime_description({"aux_input": [768, 768], "aux_interval": 19})
    except RuntimeError as exc:
        assert "complete effective auxiliary profile" in str(exc)
    else:
        raise AssertionError("incomplete detector profile must not produce runtime provenance")
