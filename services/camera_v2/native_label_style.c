#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

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

static void set_local_track_label(NvDsObjectMeta *obj, unsigned int source_id) {
    if (!obj) return;

    /* Cross-camera ReID is intentionally absent. NvDCF object_id is camera-local,
     * so include the camera number in every unknown label. This prevents a local
     * track #02 on two cameras from looking like one shared Global ID. */
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
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            if (obj->rect_params.width <= 1.0f || obj->rect_params.height <= 1.0f) continue;
            set_local_track_label(obj, frame_meta->source_id);
            ++styled;
        }
    }
    return styled;
}
