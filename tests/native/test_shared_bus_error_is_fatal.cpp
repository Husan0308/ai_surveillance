#define main six_camera_application_main
#include "/candidate/source/native/main.cpp"
#undef main

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

static void test_source_local_decoder_error_does_not_fail_process() {
  SourceCtx camera;
  camera.id = "CAM-02";
  camera.source_id = 1;
  camera.source_generation = 7;
  camera.source = gst_bin_new("synthetic-camera-source-generation-7");
  GstElement *decoder_child = gst_element_factory_make("fakesrc", "synthetic-decoder-child");
  assert(camera.source && decoder_child);
  assert(gst_bin_add(GST_BIN(camera.source), decoder_child));
  assert(source_message_owner::attach(GST_OBJECT(camera.source), &camera, 7));
  assert(source_message_owner::attach(GST_OBJECT(decoder_child), &camera, 7));
  std::filesystem::create_directories("/work/recovery-test-shm");
  assert(g_setenv("V11_TRACKED_OBJECT_FEED_DIR", "/work/recovery-test-shm", TRUE));
  sources = {&camera};
  fatal = false;
  shared_errors = 0;
  source_stall_recovery_enabled = false;

  GstMessage *message = gst_message_new_error(
      GST_OBJECT(decoder_child),
      g_error_new_literal(GST_STREAM_ERROR, GST_STREAM_ERROR_DECODE,
                          "injected source-local decoder error"),
      g_strdup("isolated decoder fixture"));
  assert(message);
  bus_message(nullptr, message, nullptr);

  assert(camera.errors == 1);
  assert(camera.last_failure_reason == "source_gstreamer_error");
  assert(camera.recovery_policy.state() == camera_recovery::SourceHealthState::OFFLINE);
  assert(camera.tracking_epoch.load() == 1);
  assert(camera.source_generation.load() == 7);
  assert(shared_errors == 0);
  assert(!fatal.load());

  gst_message_unref(message);
  sources.clear();
  gst_object_unref(camera.source);
}

int main() {
  gst_init(nullptr, nullptr);
  test_source_local_decoder_error_does_not_fail_process();
  assert(sources.empty());
  assert(loop == nullptr);
  fatal = false;
  shared_errors = 0;

  GstElement *mux_origin = gst_bin_new("synthetic-shared-mux-origin");
  assert(mux_origin);
  GError *error = g_error_new_literal(GST_STREAM_ERROR, GST_STREAM_ERROR_FAILED,
                                      "injected shared mux failure");
  GstMessage *message = gst_message_new_error(
      GST_OBJECT(mux_origin), error, g_strdup("isolated synthetic mux-origin error"));
  assert(message);

  bus_message(nullptr, message, nullptr);

  assert(shared_errors == 1);
  assert(fatal.load());
  gst_message_unref(message);
  gst_object_unref(mux_origin);
  std::puts("PASS: source-local decoder error stays camera-local and advances its tracking epoch");
  std::puts("PASS: an unowned shared mux-origin GStreamer error increments shared_errors and enters process-fatal state");
  return 0;
}
