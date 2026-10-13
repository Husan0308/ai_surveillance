#pragma once

#include <gstnvdsmeta.h>
#include <nvdsmeta.h>

#include <new>

struct SourceEpochStamp {
  guint source_id = G_MAXUINT;
  guint source_generation = 0;
  guint64 tracking_epoch = 0;
};

static NvDsMetaType source_epoch_meta_type() {
  static gchar descriptor[] = "APSIDE.SIXCAM.SOURCE_EPOCH";
  static const NvDsMetaType type = nvds_get_user_meta_type(descriptor);
  return type;
}

static gpointer source_epoch_meta_copy(gpointer data, gpointer) {
  const auto *source = static_cast<const SourceEpochStamp *>(data);
  if (!source) return nullptr;
  return new (std::nothrow) SourceEpochStamp(*source);
}

static void source_epoch_meta_release(gpointer data, gpointer) {
  delete static_cast<SourceEpochStamp *>(data);
}

// DeepStream invokes this before nvstreammux to create the frame-level
// NvDsUserMeta. The callback input is the NvDsUserMeta that wraps NvDsMeta.
static gpointer source_epoch_to_frame_meta(gpointer data, gpointer) {
  auto *source_meta = static_cast<NvDsUserMeta *>(data);
  return source_meta ? source_epoch_meta_copy(source_meta->user_meta_data, nullptr) : nullptr;
}

// The release callback input is the frame-level NvDsUserMeta itself.
static void source_epoch_frame_meta_release(gpointer data, gpointer) {
  auto *frame_meta = static_cast<NvDsUserMeta *>(data);
  if (!frame_meta) return;
  source_epoch_meta_release(frame_meta->user_meta_data, nullptr);
  frame_meta->user_meta_data = nullptr;
}

static bool source_epoch_meta_add(GstBuffer *buffer, const SourceEpochStamp &stamp) {
  if (!buffer) return false;
  auto *payload = new (std::nothrow) SourceEpochStamp(stamp);
  if (!payload) return false;
  NvDsMeta *meta = gst_buffer_add_nvds_meta(buffer, payload, nullptr,
      source_epoch_meta_copy, source_epoch_meta_release);
  if (!meta) {
    delete payload;
    return false;
  }
  meta->meta_type = static_cast<gint>(source_epoch_meta_type());
  meta->gst_to_nvds_meta_transform_func = source_epoch_to_frame_meta;
  meta->gst_to_nvds_meta_release_func = source_epoch_frame_meta_release;
  return true;
}

static bool source_epoch_meta_read_frame(const NvDsFrameMeta *frame,
                                         SourceEpochStamp *stamp) {
  if (!frame || !stamp) return false;
  const NvDsMetaType expected = source_epoch_meta_type();
  for (NvDsMetaList *node = frame->frame_user_meta_list; node; node = node->next) {
    auto *meta = static_cast<NvDsUserMeta *>(node->data);
    if (!meta || meta->base_meta.meta_type != expected || !meta->user_meta_data) continue;
    *stamp = *static_cast<const SourceEpochStamp *>(meta->user_meta_data);
    return true;
  }
  return false;
}
