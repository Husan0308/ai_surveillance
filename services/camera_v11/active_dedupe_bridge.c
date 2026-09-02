#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

static NvDsFrameMeta *find_frame(NvDsBatchMeta *batch_meta, unsigned int source_id) {
    if (!batch_meta) return NULL;
    for (NvDsMetaList *node = batch_meta->frame_meta_list; node != NULL; node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)node->data;
        if (!frame_meta) continue;
        if (frame_meta->source_id == source_id || frame_meta->pad_index == source_id) {
            return frame_meta;
        }
    }
    return NULL;
}

static int id_selected(uint64_t id, const uint64_t *ids, int count) {
    for (int i = 0; i < count; ++i) {
        if (ids[i] == id) return 1;
    }
    return 0;
}

static void hide_track(NvDsObjectMeta *obj) {
    if (!obj) return;
    obj->class_id = -1;
    obj->rect_params.border_width = 0;
    obj->rect_params.has_bg_color = 0;
    if (obj->text_params.display_text) {
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = NULL;
    }
    obj->text_params.set_bg_clr = 0;
}

/*
 * Hide selected already-tracked object IDs after nvtracker and before nvdsosd.
 * This is presentation/downstream metadata filtering only: it does not alter the
 * NvDCF internal target database, detector association, or ReAssoc state.
 */
int camera_v11_hide_track_ids(uintptr_t buffer_ptr,
                              unsigned int source_id,
                              const uint64_t *ids,
                              int count) {
    if (!buffer_ptr || !ids || count <= 0) return 0;

    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_id);
    if (!frame_meta) return -2;

    int hidden = 0;
    for (NvDsMetaList *node = frame_meta->obj_meta_list; node != NULL; node = node->next) {
        NvDsObjectMeta *obj = (NvDsObjectMeta *)node->data;
        if (!obj) continue;
        if (obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
        if (!id_selected((uint64_t)obj->object_id, ids, count)) continue;
        hide_track(obj);
        ++hidden;
    }
    return hidden;
}
