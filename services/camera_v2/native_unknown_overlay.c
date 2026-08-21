#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/*
 * Presentation-only overlay restored from rebuild/gpu-v2-clean.
 * Geometry is already mapped into final grid/fullscreen wall coordinates by
 * Python.  This helper only creates NvDsObjectMeta with a stable local track id
 * and the old yellow Unknown_C{camera}_{track} styling.
 */

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

static void set_color(NvOSD_ColorParams *dst,
                      float red,
                      float green,
                      float blue,
                      float alpha) {
    dst->red = red;
    dst->green = green;
    dst->blue = blue;
    dst->alpha = alpha;
}

static void style_unknown(NvDsObjectMeta *obj,
                          unsigned int source_id,
                          uint64_t track_id) {
    if (!obj) return;

    /* Exact gpu_v2_clean unknown color. */
    const float red = 0.965f;
    const float green = 0.725f;
    const float blue = 0.294f;

    obj->rect_params.border_width = 2;
    set_color(&obj->rect_params.border_color, red, green, blue, 1.0f);
    obj->rect_params.has_bg_color = 0;

    char label[64];
    g_snprintf(
        label,
        sizeof(label),
        "Unknown_C%u_%02" G_GUINT64_FORMAT,
        source_id + 1,
        (guint64)track_id
    );

    if (obj->text_params.display_text) {
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = NULL;
    }
    obj->text_params.display_text = g_strdup(label);
    obj->text_params.x_offset = (unsigned int)(
        obj->rect_params.left > 0.0f ? obj->rect_params.left : 0.0f
    );
    obj->text_params.y_offset = (unsigned int)(
        obj->rect_params.top >= 17.0f ? obj->rect_params.top - 17.0f : obj->rect_params.top
    );
    obj->text_params.font_params.font_name = "Monospace";
    obj->text_params.font_params.font_size = 10;
    set_color(&obj->text_params.font_params.font_color, 0.025f, 0.055f, 0.070f, 1.0f);
    obj->text_params.set_bg_clr = 1;
    set_color(&obj->text_params.text_bg_clr, red, green, blue, 1.0f);
}

int camera_v2_add_unknown_tracks(uintptr_t buffer_ptr,
                                 unsigned int source_id,
                                 const float *boxes,
                                 const uint64_t *track_ids,
                                 int count) {
    if (!buffer_ptr || !boxes || !track_ids || count <= 0) return 0;

    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_id);
    if (!frame_meta) return -2;

    int added = 0;
    for (int i = 0; i < count; ++i) {
        const float *b = boxes + (i * 5);
        const float x1 = b[0];
        const float y1 = b[1];
        const float x2 = b[2];
        const float y2 = b[3];
        const float conf = b[4];
        const uint64_t track_id = track_ids[i];

        if (track_id == 0 || x2 <= x1 || y2 <= y1) continue;

        NvDsObjectMeta *obj = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj) continue;

        const float width = x2 - x1;
        const float height = y2 - y1;
        obj->unique_component_id = 192;
        obj->class_id = 0;
        obj->object_id = (guint64)track_id;
        obj->confidence = conf;
        obj->tracker_confidence = -0.1f;
        strncpy(obj->obj_label, "Person", MAX_LABEL_SIZE - 1);
        obj->obj_label[MAX_LABEL_SIZE - 1] = '\0';

        obj->detector_bbox_info.org_bbox_coords.left = x1;
        obj->detector_bbox_info.org_bbox_coords.top = y1;
        obj->detector_bbox_info.org_bbox_coords.width = width;
        obj->detector_bbox_info.org_bbox_coords.height = height;

        obj->tracker_bbox_info.org_bbox_coords.left = x1;
        obj->tracker_bbox_info.org_bbox_coords.top = y1;
        obj->tracker_bbox_info.org_bbox_coords.width = width;
        obj->tracker_bbox_info.org_bbox_coords.height = height;

        obj->rect_params.left = x1;
        obj->rect_params.top = y1;
        obj->rect_params.width = width;
        obj->rect_params.height = height;
        style_unknown(obj, source_id, track_id);

        nvds_add_obj_meta_to_frame(frame_meta, obj, NULL);
        ++added;
    }
    return added;
}
