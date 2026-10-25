from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DETECTION = ROOT / "scripts/yolo26m_person/detection.hpp"
MAIN = ROOT / "scripts/cam_six_validation/main.cpp"


def test_periodic_stats_do_not_export_and_shutdown_exports_after_null_and_unref():
    detection = DETECTION.read_text()
    main = MAIN.read_text()
    stats = detection.split("static void detection_stats()", 1)[1].split(
        "static void detection_link", 1
    )[0]
    assert "pgie_diag_dump" not in stats

    # Normal/runtime-error teardown has one call site. A second call is only
    # reachable for a setup-only exit before streaming has started.
    assert main.count("pgie_diag_dump();") == 2
    assert "std::atexit(pgie_diag_atexit_export)" in main
    assert "pgie_diag_pipeline_started.load(std::memory_order_acquire)" in main
    early_exit = main.split("static void pgie_diag_atexit_export()", 1)[1].split(
        "#endif", 1
    )[0]
    assert early_exit.index("pgie_diag_pipeline_started.load") < early_exit.index(
        "pgie_diag_dump();"
    )
    shutdown = main.split("if (!fatal) g_main_loop_run(loop);", 1)[1]
    assert shutdown.index("gst_element_set_state(pipeline, GST_STATE_NULL);") < shutdown.index(
        "gst_object_unref(pipeline);"
    ) < shutdown.index("pgie_diag_dump();")
    assert "current == GST_STATE_NULL && pending == GST_STATE_VOID_PENDING" in shutdown
    assert shutdown.index("if (pipeline_stopped)") < shutdown.index("pgie_diag_dump();")

    failure = main.split("static void fail(", 1)[1].split("static GstElement *make(", 1)[0]
    assert "fatal = true" in failure and "g_main_loop_quit(loop)" in failure


def test_export_is_opt_in_and_one_shot_before_opening_file():
    detection = DETECTION.read_text()
    dump = detection.split("static void pgie_diag_dump()", 1)[1].split(
        "static bool write_all_json_line", 1
    )[0]
    assert dump.index("if (!pgie_diag_enabled") < dump.index("compare_exchange_strong")
    assert dump.index("compare_exchange_strong") < dump.index("std::fopen(")
    assert 'PGIE_DIAG_TRACE_PATH "/work/pgie-stage-trace.jsonl"' in detection
