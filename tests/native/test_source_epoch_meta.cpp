#include "services/camera_v11/native/six_camera_v13/source_epoch_meta.hpp"

#include <cassert>
#include <cstdio>

int main() {
  gst_init(nullptr, nullptr);
  GstBuffer *source = gst_buffer_new_allocate(nullptr, 16, nullptr);
  assert(source);
  const SourceEpochStamp expected{4, 17, 29};
  assert(source_epoch_meta_add(source, expected));

  NvDsBatchMeta *batch = nvds_create_batch_meta(1);
  assert(batch);
  NvDsFrameMeta *frame = nvds_acquire_frame_meta_from_pool(batch);
  assert(frame);
  frame->source_id = expected.source_id;
  frame->frame_num = 8;
  frame->buf_pts = 123456789;
  nvds_add_frame_meta_to_batch(batch, frame);
  nvds_copy_gst_meta_to_frame_meta(source, batch, frame);

  SourceEpochStamp actual;
  assert(source_epoch_meta_read_frame(frame, &actual));
  assert(actual.source_id == expected.source_id);
  assert(actual.source_generation == expected.source_generation);
  assert(actual.tracking_epoch == expected.tracking_epoch);

  // The frame-level value is a deep copy; changing the source GstBuffer
  // payload must not mutate metadata already handed to the batch.
  NvDsMeta *gst_meta = gst_buffer_get_nvds_meta(source);
  assert(gst_meta && gst_meta->meta_type == static_cast<gint>(source_epoch_meta_type()));
  static_cast<SourceEpochStamp *>(gst_meta->meta_data)->tracking_epoch = 99;
  SourceEpochStamp still_expected;
  assert(source_epoch_meta_read_frame(frame, &still_expected));
  assert(still_expected.tracking_epoch == expected.tracking_epoch);

  NvDsFrameMeta frame_without_stamp{};
  frame_without_stamp.source_id = expected.source_id;
  assert(!source_epoch_meta_read_frame(&frame_without_stamp, &actual));

  assert(nvds_destroy_batch_meta(batch));
  gst_buffer_unref(source);
  std::puts("PASS: DS9.1 source generation/epoch metadata survives GstBuffer-to-frame transfer");
  return 0;
}
