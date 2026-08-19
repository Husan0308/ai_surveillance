#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

typedef struct {
    uint64_t object_id;
    uint32_t source_id;
    uint32_t global_id;
    uint32_t state_code;
} CameraV2GlobalLabel;

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

static int is_generic_person_label(const char *label) {
    if (!label || !label[0]) return 1;
    if (g_ascii_strcasecmp(label, "Person") == 0) return 1;
    if (g_ascii_strcasecmp(label, "person") == 0) return 1;
    if (g_ascii_strncasecmp(label, "Unknown_", 8) == 0) return 1;
    return 0;
}

/* DeepStream keeps detector, tracker and last-writer rectangles separately.
 * At the nvtracker src probe the canonical visual location is tracker_bbox_info.
 * Copy it into rect_params immediately before styling so OSD, count and heatmap
 * all use the exact same current NvDCF rectangle. */
static void sync_rect_from_tracker(NvDsObjectMeta *obj) {
    if (!obj) return;
    NvOSD_RectParams *rect = &obj->tracker_bbox_info.org_bbox_coords;
    if (rect->width <= 1.0f || rect->height <= 1.0f) return;
    obj->rect_params.left = rect->left;
    obj->rect_params.top = rect->top;
    obj->rect_params.width = rect->width;
    obj->rect_params.height = rect->height;
}

static int should_style_track(const NvDsObjectMeta *obj) {
    if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) return 0;
    if (obj->unique_component_id == 191) return 0;
    if (obj->rect_params.width <= 1.0f || obj->rect_params.height <= 1.0f) return 0;
    if (obj->rect_params.border_width == 0) return 0;
    return 1;
}

static void write_display_label(NvDsObjectMeta *obj,
                                const char *display_label,
                                float red,
                                float green,
                                float blue) {
    obj->rect_params.border_width = 2;
    set_color(&obj->rect_params.border_color, red, green, blue, 1.0f);
    obj->rect_params.has_bg_color = 0;

    if (obj->text_params.display_text) {
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = NULL;
    }
    obj->text_params.display_text = g_strdup(display_label);
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

static void set_local_track_label(NvDsObjectMeta *obj, unsigned int source_id) {
    if (!obj) return;
    char local_unknown[64];
    const int known = !is_generic_person_label(obj->obj_label);
    const char *display_label = obj->obj_label;

    if (!known) {
        g_snprintf(
            local_unknown,
            sizeof(local_unknown),
            "Unknown_C%u_%02" G_GUINT64_FORMAT,
            source_id + 1,
            (guint64)obj->object_id
        );
        display_label = local_unknown;
    }

    const float red = known ? 0.239f : 0.965f;
    const float green = known ? 0.863f : 0.725f;
    const float blue = known ? 0.592f : 0.294f;
    write_display_label(obj, display_label, red, green, blue);
}

int camera_v2_apply_local_track_style(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int styled = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            sync_rect_from_tracker(obj);
            if (!should_style_track(obj)) continue;
            set_local_track_label(obj, frame_meta->source_id);
            ++styled;
        }
    }
    return styled;
}

static const CameraV2GlobalLabel *find_global_label(const CameraV2GlobalLabel *rows,
                                                     int count,
                                                     uint32_t source_id,
                                                     uint64_t object_id) {
    if (!rows || count <= 0) return NULL;
    for (int i = 0; i < count; ++i) {
        if (rows[i].source_id == source_id && rows[i].object_id == object_id) return &rows[i];
    }
    return NULL;
}

int camera_v2_apply_global_track_style(uintptr_t buffer_ptr,
                                       const CameraV2GlobalLabel *rows,
                                       int count) {
    if (!buffer_ptr || !rows || count <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int styled = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;
        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            sync_rect_from_tracker(obj);
            if (!should_style_track(obj)) continue;
            const CameraV2GlobalLabel *match = find_global_label(
                rows, count, (uint32_t)frame_meta->source_id, (uint64_t)obj->object_id
            );
            if (!match || match->global_id == 0) continue;

            char text[48];
            float red = 0.239f, green = 0.863f, blue = 0.592f;
            if (match->state_code == 1) {
                g_snprintf(text, sizeof(text), "G%03u?", match->global_id);
                red = 0.965f; green = 0.725f; blue = 0.294f;
            } else if (match->state_code == 3) {
                g_snprintf(text, sizeof(text), "G%03u!", match->global_id);
                red = 0.941f; green = 0.310f; blue = 0.310f;
            } else {
                g_snprintf(text, sizeof(text), "G%03u", match->global_id);
                red = 0.239f; green = 0.863f; blue = 0.592f;
            }
            write_display_label(obj, text, red, green, blue);
            ++styled;
        }
    }
    return styled;
}
