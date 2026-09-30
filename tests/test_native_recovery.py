"""Compile the actual native recovery-timeline helper, not a Python model."""
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_reset_keeps_mva_frame_timeline_without_changing_its_peer(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "glib-2.0"],
                           text=True, capture_output=True, check=True).stdout
    source = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    structure = source[source.index("typedef struct {\n  gboolean seen;\n  guint64 offset;"):
                       source.index("static SourceFrameContinuity source_frame_continuity")]
    helper = source[source.index("static gint\nsource_frame_continuity_update"):
                    source.index("static void\nsource_frame_continuity_apply")]
    harness = tmp_path / "timeline.c"
    harness.write_text("#include <glib.h>\n" + structure + helper + r'''
int main(void) {
  SourceFrameContinuity cam01 = {0}, cam04 = {0};
  gboolean reset;
  for (int i = 0; i <= 333; ++i) {
    g_assert_cmpint(source_frame_continuity_update(&cam01, i, &reset), ==, i);
    g_assert_false(reset);
    g_assert_cmpint(source_frame_continuity_update(&cam04, i, &reset), ==, i);
  }
  g_assert_cmpint(source_frame_continuity_update(&cam01, 0, &reset), ==, 334);
  g_assert_true(reset);
  g_assert_cmpint(source_frame_continuity_update(&cam04, 334, &reset), ==, 334);
  g_assert_false(reset);
  for (int i = 1; i <= 20; ++i) {
    g_assert_cmpint(source_frame_continuity_update(&cam01, i, &reset), ==, 334+i);
    g_assert_false(reset);
  }
  g_assert_cmpint(source_frame_continuity_update(&cam01, 0, &reset), ==, 355);
  g_assert_true(reset);
  g_assert_cmpint(source_frame_continuity_update(&cam01, 0, &reset), ==, 355);
  g_assert_false(reset);
  g_assert_cmpint(source_frame_continuity_update(&cam01, -1, &reset), ==, -1);
  g_assert_cmpint(cam01.last_output, ==, 355);
  cam01.last_raw = 5; cam01.last_output = G_MAXINT;
  g_assert_cmpint(source_frame_continuity_update(&cam01, 0, &reset), ==, -1);
  return 0;
}
''')
    binary = tmp_path / "timeline"
    subprocess.run([compiler, "-Wall", "-Wextra", "-Werror", str(harness),
                    "-o", str(binary), *shlex.split(flags)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=5)


def test_continuity_is_applied_once_before_pgie_and_not_on_preview():
    source = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    call = 'if (!g_strcmp0 (context->stage, "nvstreammux"))\n    source_frame_continuity_apply (batch_meta);'
    assert call in source
    assert source.count("source_frame_continuity_apply (batch_meta);") == 1
    preview = source[source.index("/* Latest-only V11"):source.index("frame_audit_camera_id (guint source_id", source.index("/* Latest-only V11"))]
    assert "source_frame_continuity" not in preview


def test_reconnectable_rtsp_eos_is_not_terminal_but_other_events_are_preserved(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "gstreamer-1.0"],
                           text=True, capture_output=True, check=True).stdout
    source = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    structure = source[source.index("typedef struct {\n  guint source_id;\n  gboolean continuous_rtsp;"):
                       source.index("static GstPadProbeReturn\nsource_recovery_event_probe")]
    helper = source[source.index("static GstPadProbeReturn\nsource_recovery_event_probe"):
                    source.index("static void\nsource_recovery_attach_events")]
    harness = tmp_path / "events.c"
    harness.write_text("#include <gst/gst.h>\n" + structure + helper + r'''
int main(void) {
  gst_init(NULL, NULL);
  SourceRecoveryEventContext live = {.source_id=0, .continuous_rtsp=TRUE};
  SourceRecoveryEventContext finite = {.source_id=0, .continuous_rtsp=FALSE};
  GstPadProbeInfo info = {.type=GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM};
  info.data = gst_event_new_eos();
  g_assert_cmpint(source_recovery_event_probe(NULL, &info, &live), ==, GST_PAD_PROBE_DROP);
  g_assert_cmpint(source_recovery_event_probe(NULL, &info, &finite), ==, GST_PAD_PROBE_OK);
  gst_event_unref(info.data);
  info.data = gst_event_new_flush_start();
  g_assert_cmpint(source_recovery_event_probe(NULL, &info, &live), ==, GST_PAD_PROBE_OK);
  gst_event_unref(info.data);
  info.data = gst_event_new_flush_stop(TRUE);
  g_assert_cmpint(source_recovery_event_probe(NULL, &info, &live), ==, GST_PAD_PROBE_OK);
  gst_event_unref(info.data);
  GstSegment segment;
  gst_segment_init(&segment, GST_FORMAT_TIME);
  info.data = gst_event_new_segment(&segment);
  g_assert_cmpint(source_recovery_event_probe(NULL, &info, &live), ==, GST_PAD_PROBE_OK);
  gst_event_unref(info.data);
  gst_deinit();
  return 0;
}
''')
    binary = tmp_path / "events"
    subprocess.run([compiler, "-Wall", "-Wextra", "-Werror", str(harness),
                    "-o", str(binary), *shlex.split(flags)], check=True, timeout=30)
    subprocess.run([str(binary)], check=True, timeout=5)


def test_recovery_event_policy_never_probes_preview_or_video_buffers():
    source = (ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c").read_text()
    attach = source[source.index("static void\nsource_recovery_attach_events"):
                    source.index("static gint\nsource_frame_continuity_update")]
    assert "source->config->type != NV_DS_SOURCE_RTSP" in attach
    assert "source->config->rtsp_reconnect_interval_sec > 0" in attach
    assert "preview_mux_queues[i] : source->bin" in attach
    assert "GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM" in attach
    assert "GST_PAD_PROBE_TYPE_BUFFER" not in attach
    assert "MV3DT_TEST_" not in attach
