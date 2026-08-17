#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define MAX_LABEL_STATES 512
#define LABEL_STATE_RETIRE_FRAMES 80

typedef struct {
    int valid;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_frame_num;
    char known_name[MAX_LABEL_SIZE];
} CameraV2LabelState;

static CameraV2LabelState g_label_states[MAX_LABEL_STATES];

static int is_person_fallback(const char *label) {
    if (!label || !label[0]) return 1;
    if (g_ascii_strcasecmp(label, "Person") == 0) return 1;
    if (g_ascii_strcasecmp(label, "person") == 0) return 1;
    return 0;
}

static int is_unknown_label(const char *label) {
    return label && g_ascii_strncasecmp(label, "Unknown_", 8) == 0;
}

static CameraV2LabelState *find_label_state(unsigned int source_id,
                                             uint64_t object_id,
                                             int create) {
    int free_index = -1;
    int oldest_index = 0;
    uint64_t oldest_frame = UINT64_MAX;

    for (int i = 0; i < MAX_LABEL_STATES; ++i) {
        CameraV2LabelState *state = &g_label_states[i];
        if (state->valid && state->source_id == source_id && state->object_id == object_id) {
            return state;
        }
        if (!state->valid && free_index < 0) free_index = i;
        if (state->valid && state->last_frame_num < oldest_frame) {
            oldest_frame = state->last_frame_num;
            oldest_index = i;
        }
    }

    if (!create) return NULL;
    int index = free_index >= 0 ? free_index : oldest_index;
    CameraV2LabelState *state = &g_label_states[index];
    memset(state, 0, sizeof(*state));
    state->valid = 1;
    state->source_id = source_id;
    state->object_id = object_id;
    return state;
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

static void set_reference_label(NvDsObjectMeta *obj,
                                unsigned int source_id,
                                uint64_t frame_num) {
    if (!obj) return;

    CameraV2LabelState *state = find_label_state(source_id, (uint64_t)obj->object_id, 0);
    int upstream_unknown = is_unknown_label(obj->obj_label);
    int upstream_known = !is_person_fallback(obj->obj_label) && !upstream_unknown;

    if (upstream_known) {
        state = find_label_state(source_id, (uint64_t)obj->object_id, 1);
        if (state) {
            g_strlcpy(state->known_name, obj->obj_label, sizeof(state->known_name));
            state->last_frame_num = frame_num;
        }
    } else if (state && state->known_name[0]) {
        /* Keep a recognized name attached to the same NvDCF track even if a
         * downstream display-hold box carries the generic Person label. */
        state->last_frame_num = frame_num;
    }

    const char *known_name = (state && state->known_name[0]) ? state->known_name : NULL;
    int known = known_name != NULL;
    char fallback_unknown[64];
    const char *display_label = known_name;

    if (!known && upstream_unknown) {
        /* GlobalReIDManager already assigned the stable session-level Unknown ID. */
        display_label = obj->obj_label;
    } else if (!known) {
        g_snprintf(
            fallback_unknown,
            sizeof(fallback_unknown),
            "Unknown_%02" G_GUINT64_FORMAT,
            (guint64)obj->object_id
        );
        display_label = fallback_unknown;
    }

    /* Reference UI colors: known = teal/green, unknown = amber. */
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
    obj->text_params.x_offset = (unsigned int)(obj->rect_params.left > 0.0f ? obj->rect_params.left : 0.0f);
    obj->text_params.y_offset = (unsigned int)(
        obj->rect_params.top >= 17.0f ? obj->rect_params.top - 17.0f : obj->rect_params.top
    );
    obj->text_params.font_params.font_name = "Monospace";
    obj->text_params.font_params.font_size = 10;
    set_color(&obj->text_params.font_params.font_color, 0.025f, 0.055f, 0.070f, 1.0f);
    obj->text_params.set_bg_clr = 1;
    set_color(&obj->text_params.text_bg_clr, red, green, blue, 1.0f);
}

int camera_v2_apply_identity_style(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int styled = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;

        unsigned int source_id = frame_meta->source_id;
        uint64_t frame_num = (uint64_t)frame_meta->frame_num;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            if (obj->rect_params.width <= 1.0f || obj->rect_params.height <= 1.0f) continue;
            set_reference_label(obj, source_id, frame_num);
            ++styled;
        }

        for (int i = 0; i < MAX_LABEL_STATES; ++i) {
            CameraV2LabelState *state = &g_label_states[i];
            if (!state->valid || state->source_id != source_id) continue;
            if (frame_num > state->last_frame_num &&
                frame_num - state->last_frame_num > LABEL_STATE_RETIRE_FRAMES) {
                memset(state, 0, sizeof(*state));
            }
        }
    }
    return styled;
}
