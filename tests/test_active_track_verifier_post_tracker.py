from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKER = ROOT / "services/camera_v11/native/six_camera_v13/tracker.hpp"


def _function_body(source: str, signature: str, next_signature: str) -> str:
    start = source.index(signature)
    end = source.index(next_signature, start)
    return source[start:end]


def test_active_birth_verifier_only_submits_associated_native_targets():
    source = TRACKER.read_text()
    tracker_enter = _function_body(
        source,
        "static GstPadProbeReturn tracker_enter(",
        "static GstPadProbeReturn tracker_exit(",
    )
    tracker_exit = _function_body(
        source,
        "static GstPadProbeReturn tracker_exit(",
        "static void tracker_",
    )

    # Scheduling before NvDCF sees every detector proposal, including rows
    # that never become native targets. The verifier must run only after the
    # detector-to-active association provides an exact current native ID.
    assert "lowconf_verifier_submit(" not in tracker_enter
    association = tracker_exit.index("detection_to_active")
    submit = tracker_exit.index("lowconf_verifier_submit(")
    assignment = tracker_exit.index("lowconf_verifier_assign_active_track(")
    assert association < submit < assignment
    assert "row.detection_index < 0" in tracker_exit
    assert "tracker_birth_verification.begin(" in tracker_exit
    assert "tracker_birth_verification.cancel(" in tracker_exit

