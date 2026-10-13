#pragma once

#include <gst/gst.h>

#include <cstdint>
#include <new>

namespace source_message_owner {

struct Tag {
  gpointer owner = nullptr;
  guint64 generation = 0;
};

enum class Disposition { CURRENT_SOURCE, STALE_SOURCE, SHARED_OR_UNKNOWN };

inline GQuark tag_quark() {
  return g_quark_from_static_string("sixcam-source-message-owner-v1");
}

inline void destroy_tag(gpointer data) {
  delete static_cast<Tag *>(data);
}

inline bool attach(GstObject *object, gpointer owner, guint64 generation) {
  if (!object || !owner || generation == 0) return false;
  const GQuark quark = tag_quark();
  if (auto *existing = static_cast<Tag *>(g_object_get_qdata(G_OBJECT(object), quark)))
    return existing->owner == owner && existing->generation == generation;
  auto *tag = new (std::nothrow) Tag{owner, generation};
  if (!tag) return false;
  g_object_set_qdata_full(G_OBJECT(object), quark, tag, destroy_tag);
  return true;
}

inline bool nearest_tag(GstObject *object, Tag *result) {
  if (!object || !result) return false;
  GstObject *cursor = GST_OBJECT(gst_object_ref(object));
  while (cursor) {
    auto *tag = static_cast<Tag *>(g_object_get_qdata(G_OBJECT(cursor), tag_quark()));
    if (tag) {
      *result = *tag;
      gst_object_unref(cursor);
      return true;
    }
    GstObject *parent = gst_object_get_parent(cursor);
    gst_object_unref(cursor);
    cursor = parent;
  }
  return false;
}

inline Disposition classify(const Tag *tag, gpointer expected_owner,
                            guint64 current_generation, bool current_owner_active) {
  if (!tag || tag->owner != expected_owner) return Disposition::SHARED_OR_UNKNOWN;
  if (!current_owner_active || tag->generation != current_generation)
    return Disposition::STALE_SOURCE;
  return Disposition::CURRENT_SOURCE;
}

// Resolve a GstMessage source against one camera's active source and its
// immutable owner-generation tags. Keeping this in the shared helper lets the
// native regression exercise the same ancestry/tag decision as bus_message().
inline Disposition resolve(GstObject *message_source, GstObject *active_source,
                           gpointer expected_owner, guint64 current_generation,
                           bool current_owner_active, Tag *resolved_tag) {
  if (!message_source || !expected_owner) return Disposition::SHARED_OR_UNKNOWN;
  if (active_source &&
      (message_source == active_source ||
       gst_object_has_as_ancestor(message_source, active_source))) {
    if (resolved_tag) *resolved_tag = Tag{expected_owner, current_generation};
    return Disposition::CURRENT_SOURCE;
  }
  Tag tag;
  if (!nearest_tag(message_source, &tag)) return Disposition::SHARED_OR_UNKNOWN;
  if (resolved_tag) *resolved_tag = tag;
  return classify(&tag, expected_owner, current_generation, current_owner_active);
}

}  // namespace source_message_owner
