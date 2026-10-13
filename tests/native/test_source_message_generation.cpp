#include "services/camera_v11/native/six_camera_v13/source_message_owner.hpp"

#include <cassert>
#include <cstdio>

using source_message_owner::Disposition;
using source_message_owner::Tag;

int main() {
  gst_init(nullptr, nullptr);
  int source_context = 0;
  GstElement *pipeline = gst_pipeline_new("test-pipeline");
  GstElement *old_source = gst_bin_new("old-source");
  GstElement *old_child = gst_element_factory_make("fakesrc", "old-child");
  GstElement *new_source = gst_bin_new("new-source");
  GstElement *new_child = gst_element_factory_make("fakesrc", "new-child");
  GstElement *shared_child = gst_element_factory_make("fakesrc", "shared-child");
  assert(pipeline && old_source && old_child && new_source && new_child && shared_child);
  assert(gst_bin_add(GST_BIN(pipeline), old_source));
  assert(gst_bin_add(GST_BIN(old_source), old_child));
  assert(gst_bin_add(GST_BIN(pipeline), new_source));
  assert(gst_bin_add(GST_BIN(new_source), new_child));
  assert(gst_bin_add(GST_BIN(pipeline), shared_child));
  assert(source_message_owner::attach(GST_OBJECT(old_source), &source_context, 1));
  assert(source_message_owner::attach(GST_OBJECT(old_child), &source_context, 1));
  assert(source_message_owner::attach(GST_OBJECT(new_source), &source_context, 2));
  assert(source_message_owner::attach(GST_OBJECT(new_child), &source_context, 2));

  // A message already queued by the retired child no longer has the current
  // source as an ancestor after the old source is removed. Its generation tag
  // still identifies it as camera-local and stale instead of shared-fatal.
  GstMessage *old_error = gst_message_new_error(GST_OBJECT(old_child),
      g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_READ,
                          "delayed retired-source error"), g_strdup("late bus delivery"));
  assert(old_error);
  gst_object_ref(old_source);
  assert(gst_bin_remove(GST_BIN(pipeline), old_source));
  Tag resolved;
  assert(source_message_owner::resolve(old_error->src, GST_OBJECT(new_source),
      &source_context, 2, true, &resolved) == Disposition::STALE_SOURCE);
  assert(resolved.owner == &source_context && resolved.generation == 1);
  gst_message_unref(old_error);

  GstMessage *old_eos = gst_message_new_eos(GST_OBJECT(old_child));
  assert(source_message_owner::resolve(old_eos->src, GST_OBJECT(new_source),
      &source_context, 2, true, &resolved) == Disposition::STALE_SOURCE);
  gst_message_unref(old_eos);

  GstMessage *current_error = gst_message_new_error(GST_OBJECT(new_child),
      g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_READ,
                          "current-source error"), g_strdup("current source"));
  assert(source_message_owner::resolve(current_error->src, GST_OBJECT(new_source),
      &source_context, 2, true, &resolved) == Disposition::CURRENT_SOURCE);
  assert(resolved.owner == &source_context && resolved.generation == 2);
  gst_message_unref(current_error);

  GstMessage *shared_error = gst_message_new_error(GST_OBJECT(shared_child),
      g_error_new_literal(GST_RESOURCE_ERROR, GST_RESOURCE_ERROR_READ,
                          "shared-pipeline error"), g_strdup("shared source"));
  assert(source_message_owner::resolve(shared_error->src, GST_OBJECT(new_source),
      &source_context, 2, true, &resolved) == Disposition::SHARED_OR_UNKNOWN);
  gst_message_unref(shared_error);

  // During the removal/backoff gap the generation is still the retired one,
  // but the missing active owner is sufficient to make it stale.
  Tag retired{&source_context, 1};
  assert(source_message_owner::classify(&retired, &source_context, 1, false) ==
         Disposition::STALE_SOURCE);

  gst_object_unref(old_source);
  gst_object_unref(pipeline);
  std::puts("PASS: current, retired, EOS, and shared GStreamer message ownership");
  return 0;
}
