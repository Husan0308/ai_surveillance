import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


NATIVE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "services/mv3dt_room/native/deepstream_test5_app_main.c"
)


def test_preview_is_independent_and_not_experiment_flag_gated() -> None:
    source = NATIVE_SOURCE.read_text()

    for old_switch in (
        "MV3DT_TEST_PREVIEW_DIRECT_CUDA_COPY",
        "MV3DT_TEST_EARLY_DECODER_PREVIEW_CAM04",
        "MV3DT_TEST_CAM04_PREQUEUE_SURFACE_COPY",
        "MV3DT_TEST_CAM04_ANALYTICS_QUEUE_BUFFERS",
    ):
        assert old_switch not in source

    assert "cudaHostRegister (writer->map" in source
    assert "cudaMemcpy2D (" in source
    assert "preview_insert_early_decoder_branch" in source
    assert "preview_source_pad_probe" in source
    assert "tracker_bin.tracker" not in source[
        source.index("static void\npreview_attach") :
        source.index("static void\npreview_close")
    ]


def test_preview_keeps_one_latest_slot_and_never_uses_cpu_payload_loop() -> None:
    source = NATIVE_SOURCE.read_text()

    assert '"max-size-buffers", 1u' in source
    assert '"leaky", 2' in source
    assert "worker->latest = pending;" in source
    assert "gst_buffer_unref (replaced.buffer);" in source
    assert "for (y = 0; y < PREVIEW_HEIGHT; y++)" not in source
    assert "writer->stride, dst_params->dataPtr, dst_params->pitch" in source


def test_cam04_preview_pad_precedes_analytics_and_detaches_surfaces() -> None:
    source = NATIVE_SOURCE.read_text()
    preview_pad = source.index(
        'tee_preview_src = gst_element_request_pad_simple (tee, "src_%u");'
    )
    analytics_pad = source.index(
        'tee_analytics_src = gst_element_request_pad_simple (tee, "src_%u");'
    )

    assert preview_pad < analytics_pad
    assert '"disable-passthrough", TRUE' in source
    assert '"output-buffers", 24u' in source
    assert "gst_element_sync_state_with_parent (surface_copy)" in source


def test_preview_analytics_boundary_keeps_fixed_nonleaky_limits() -> None:
    source = NATIVE_SOURCE.read_text()
    boundary = source[
        source.index("static gboolean\npreview_insert_bounded_mux_queue") :
        source.index("/* CAM-04's source queue")
    ]
    assert "camera_index == 1 ? 16u : 2u" in boundary
    assert '"leaky", 0' in boundary
    assert boundary.index("gst_element_set_state (queue, GST_STATE_NULL)") < boundary.index(
        "gst_bin_remove (GST_BIN (source_parent), queue)"
    )


def test_preview_preserves_decoder_timing_and_frame_identity() -> None:
    source = NATIVE_SOURCE.read_text()

    assert "preview_get_decode_timing" in source
    assert "pending.pts_ns" in source
    assert "pending.source_frame_num" in source
    assert "decoder_reference_ns" in source
    assert "decoder_out_ns" in source


def test_early_preview_reconnect_restores_retained_branch(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    pkg_config = shutil.which("pkg-config")
    if not compiler or not pkg_config:
        pytest.skip("native reconnect regression requires a C compiler and pkg-config")
    dependencies = subprocess.run(
        [pkg_config, "--cflags", "--libs", "gstreamer-1.0"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if dependencies.returncode:
        pytest.skip("native reconnect regression requires GStreamer development files")

    source = NATIVE_SOURCE.read_text()
    context_start = source.index("typedef struct {\n  gint camera_index;\n  guint source_id;")
    context_end = source.index("} PreviewDecodeProbeContext;", context_start)
    context = source[context_start : context_end + len("} PreviewDecodeProbeContext;")]
    restore = source[
        source.index("static gboolean\npreview_restore_early_decoder_branch") :
        source.index("static gboolean\npreview_insert_early_decoder_branch")
    ]
    removed = source[
        source.index("static void\npreview_early_decodebin_pad_removed") :
        source.index("static void\npreview_early_decodebin_pad_added")
    ]
    harness = tmp_path / "preview_reconnect.c"
    harness.write_text(
        "#include <gst/gst.h>\n" + context + "\n" + restore + "\n" + removed + r'''

static void
pad_finalized (gpointer data, GObject *object)
{
  (void) object;
  ++*(guint *) data;
}

static void
assert_peer (GstPad *pad, GstPad *expected)
{
  GstPad *peer = gst_pad_get_peer (pad);
  g_assert_true (peer == expected);
  if (peer)
    gst_object_unref (peer);
}

static GstPad *
add_decoder_output (GstElement *decoder, GstPad *producer_src,
    guint *finalized)
{
  GstPad *output = gst_ghost_pad_new ("src", producer_src);
  g_assert_nonnull (output);
  g_object_weak_ref (G_OBJECT (output), pad_finalized, finalized);
  g_assert_true (gst_element_add_pad (decoder, output));
  return gst_element_get_static_pad (decoder, "src");
}

int
main (int argc, char **argv)
{
  gst_init (&argc, &argv);
  GstElement *source_bin = gst_bin_new ("source");
  GstElement *decoder = gst_bin_new ("decoder");
  GstElement *producer = gst_element_factory_make ("identity", "producer");
  GstElement *tee = gst_element_factory_make ("tee", "mv3dt_early_preview_tee_1");
  GstElement *copy = gst_element_factory_make ("identity",
      "mv3dt_cam04_prequeue_surface_copy_1");
  GstElement *analytics = gst_element_factory_make ("queue", "analytics");
  g_assert_nonnull (source_bin);
  g_assert_nonnull (decoder);
  g_assert_nonnull (producer);
  g_assert_nonnull (tee);
  g_assert_nonnull (copy);
  g_assert_nonnull (analytics);
  g_assert_true (gst_bin_add (GST_BIN (decoder), producer));
  gst_bin_add_many (GST_BIN (source_bin), decoder, tee, copy, analytics, NULL);

  GstPad *producer_src = gst_element_get_static_pad (producer, "src");
  GstPad *tee_sink = gst_element_get_static_pad (tee, "sink");
  GstPad *tee_src = gst_element_request_pad_simple (tee, "src_%u");
  GstPad *copy_sink = gst_element_get_static_pad (copy, "sink");
  GstPad *copy_src = gst_element_get_static_pad (copy, "src");
  GstPad *analytics_sink = gst_element_get_static_pad (analytics, "sink");
  g_assert_nonnull (producer_src);
  g_assert_nonnull (tee_sink);
  g_assert_nonnull (tee_src);
  g_assert_nonnull (copy_sink);
  g_assert_nonnull (copy_src);
  g_assert_nonnull (analytics_sink);
  g_assert_cmpint (gst_pad_link (tee_src, copy_sink), ==, GST_PAD_LINK_OK);
  g_assert_cmpint (gst_pad_link (copy_src, analytics_sink), ==, GST_PAD_LINK_OK);

  guint finalized = 0;
  GstPad *output = add_decoder_output (decoder, producer_src, &finalized);
  g_assert_cmpint (gst_pad_link (output, tee_sink), ==, GST_PAD_LINK_OK);
  PreviewDecodeProbeContext context = {0};
  context.camera_index = 1;
  context.source_id = 1;
  context.source_bin = source_bin;
  context.analytics_queue = analytics;
  context.early_decoder_preview = TRUE;
  context.early_branch_attached = TRUE;
  context.early_decoded_src_pad = gst_object_ref (output);
  g_signal_connect (decoder, "pad-removed",
      G_CALLBACK (preview_early_decodebin_pad_removed), &context);

  for (guint cycle = 0; cycle < 2; ++cycle) {
    g_assert_true (gst_element_remove_pad (decoder, output));
    g_assert_null (context.early_decoded_src_pad);
    g_assert_true (context.early_branch_attached);
    assert_peer (analytics_sink, NULL);
    assert_peer (tee_sink, NULL);
    gst_object_unref (output);
    g_assert_cmpuint (finalized, ==, cycle + 1);

    output = add_decoder_output (decoder, producer_src, &finalized);
    /* The SDK's normal pad-added callback links the replacement to analytics. */
    g_assert_cmpint (gst_pad_link (output, analytics_sink), ==, GST_PAD_LINK_OK);
    g_assert_true (preview_restore_early_decoder_branch (&context, output));
    g_assert_true (context.early_decoded_src_pad == output);
    assert_peer (output, tee_sink);
    assert_peer (tee_src, copy_sink);
    assert_peer (copy_src, analytics_sink);
    GstElement *retained_tee = gst_bin_get_by_name (GST_BIN (source_bin),
        "mv3dt_early_preview_tee_1");
    GstElement *retained_copy = gst_bin_get_by_name (GST_BIN (source_bin),
        "mv3dt_cam04_prequeue_surface_copy_1");
    g_assert_true (retained_tee == tee);
    g_assert_true (retained_copy == copy);
    gst_object_unref (retained_tee);
    gst_object_unref (retained_copy);
  }

  g_assert_true (gst_element_remove_pad (decoder, output));
  g_assert_null (context.early_decoded_src_pad);
  gst_object_unref (output);
  g_assert_cmpuint (finalized, ==, 3);
  g_assert_true (gst_pad_unlink (tee_src, copy_sink));
  gst_element_release_request_pad (tee, tee_src);
  gst_object_unref (tee_src);
  gst_object_unref (producer_src);
  gst_object_unref (tee_sink);
  gst_object_unref (copy_sink);
  gst_object_unref (copy_src);
  gst_object_unref (analytics_sink);
  gst_object_unref (source_bin);
  gst_deinit ();
  return 0;
}
'''
    )
    executable = tmp_path / "preview_reconnect"
    compiled = subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", str(harness),
         "-o", str(executable), *shlex.split(dependencies.stdout)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "G_DEBUG": "fatal-warnings"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
