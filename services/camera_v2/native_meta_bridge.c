#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

typedef struct {
    uint64_t object_id;
    uint64_t frame_num;
    uint32_t source_id;
    float left;
    float top;
    float width;
    float height;
    float confidence;
    float tracker_confidence;
} CameraV2TrackRow;

static NvDsFrameMeta *find_frame(NvDsBatchMeta *batch_meta, unsigned int source_id) {
    if (!batch_meta) return NULL;
    for (NvDsMetaList *node = batch_meta->frame_meta_list; node != NULL; node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
        if (!frame_meta) continue;
        if (frame_meta->source_id == source_id || frame_meta->pad_index == source_id) {
            return frame_meta;
        }
    }
    return NULL;
}

static void style_green(NvDsObjectMeta *obj) {
    obj->rect_params.border_width = 3;
    obj->rect_params.border_color.red = 0.10;
    obj->rect_params.border_color.green = 1.00;
    obj->rect_params.border_color.blue = 0.15;
    obj->rect_params.border_color.alpha = 1.00;
    obj->rect_params.has_bg_color = 0;
}

static float clampf_local(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static float env_float_local(const char *name, float fallback, float low, float high) {
    const char *value = g_getenv(name);
    if (!value || !value[0]) return fallback;
    char *end = NULL;
    double parsed = g_ascii_strtod(value, &end);
    if (end == value || !isfinite(parsed)) return fallback;
    if (parsed < low) parsed = low;
    if (parsed > high) parsed = high;
    return (float) parsed;
}

static float env_margin(const char *name, float fallback) {
    return env_float_local(name, fallback, 0.0f, 0.25f);
}

static int is_displayable_track(const NvDsObjectMeta *obj, float min_tracker_conf) {
    if (!obj) return 0;
    if (obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) return 0;
    if (obj->unique_component_id == 191) return 0;
    if (obj->rect_params.width <= 1.0f || obj->rect_params.height <= 1.0f) return 0;
    if (obj->tracker_confidence >= 0.0f && obj->tracker_confidence < min_tracker_conf) return 0;
    return 1;
}

static void hide_track(NvDsObjectMeta *obj) {
    if (!obj) return;
    /* Mark hidden objects as non-person downstream so labels, counts and heatmap
     * all ignore the same rejected NvDCF state. This probe runs after nvtracker,
     * so changing class_id here cannot affect tracker association. */
    obj->class_id = -1;
    obj->rect_params.border_width = 0;
    obj->rect_params.has_bg_color = 0;
    if (obj->text_params.display_text) {
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = NULL;
    }
    obj->text_params.set_bg_clr = 0;
}

static int add_boxes_to_frame(NvDsBatchMeta *batch_meta,
                              NvDsFrameMeta *frame_meta,
                              const float *boxes,
                              int count) {
    if (!batch_meta || !frame_meta || count < 0) return -1;
    if (count == 0) return 0;
    if (!boxes) return -1;

    int added = 0;
    for (int i = 0; i < count; ++i) {
        const float *b = boxes + (i * 5);
        float x1 = b[0];
        float y1 = b[1];
        float x2 = b[2];
        float y2 = b[3];
        float conf = b[4];
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsObjectMeta *obj = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj) continue;

        float width = x2 - x1;
        float height = y2 - y1;
        obj->unique_component_id = 91;
        obj->class_id = 0;
        obj->object_id = UNTRACKED_OBJECT_ID;
        obj->confidence = conf;
        obj->tracker_confidence = -0.1f;
        strncpy(obj->obj_label, "Person", MAX_LABEL_SIZE - 1);
        obj->obj_label[MAX_LABEL_SIZE - 1] = '\0';

        obj->detector_bbox_info.org_bbox_coords.left = x1;
        obj->detector_bbox_info.org_bbox_coords.top = y1;
        obj->detector_bbox_info.org_bbox_coords.width = width;
        obj->detector_bbox_info.org_bbox_coords.height = height;

        obj->rect_params.left = x1;
        obj->rect_params.top = y1;
        obj->rect_params.width = width;
        obj->rect_params.height = height;
        style_green(obj);

        nvds_add_obj_meta_to_frame(frame_meta, obj, NULL);
        ++added;
    }
    return added;
}

int camera_v2_add_boxes(uintptr_t buffer_ptr,
                        unsigned int source_id,
                        const float *boxes,
                        int count) {
    if (!buffer_ptr || !boxes || count <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;
    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_id);
    if (!frame_meta) return 0;
    return add_boxes_to_frame(batch_meta, frame_meta, boxes, count);
}

int camera_v2_apply_detector_result(uintptr_t buffer_ptr,
                                    unsigned int source_id,
                                    const float *boxes,
                                    int count) {
    if (!buffer_ptr || count < 0) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;
    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_id);
    if (!frame_meta) return -2;

    frame_meta->bInferDone = TRUE;
    return add_boxes_to_frame(batch_meta, frame_meta, boxes, count);
}

int camera_v2_style_and_count_tracked(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float side_margin = env_margin("CAMERA_V2_TRACK_BOX_SIDE_MARGIN", 0.0f);
    const float top_margin = env_margin("CAMERA_V2_TRACK_BOX_TOP_MARGIN", 0.0f);
    const float bottom_margin = env_margin("CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN", 0.0f);
    const float min_tracker_conf = env_float_local(
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF", 0.28f, 0.0f, 1.0f
    );
    int count = 0;

    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fnode->data;
        if (!frame_meta) continue;

        float frame_w = frame_meta->pipeline_width > 1 ? (float) frame_meta->pipeline_width
                                                        : (float) frame_meta->source_frame_width;
        float frame_h = frame_meta->pipeline_height > 1 ? (float) frame_meta->pipeline_height
                                                         : (float) frame_meta->source_frame_height;
        if (frame_w <= 1.0f || frame_h <= 1.0f) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;

            if (!is_displayable_track(obj, min_tracker_conf)) {
                hide_track(obj);
                continue;
            }

            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float width = obj->rect_params.width;
            float height = obj->rect_params.height;

            float new_left = left - width * side_margin;
            float new_top = top - height * top_margin;
            float new_right = left + width + width * side_margin;
            float new_bottom = top + height + height * bottom_margin;

            new_left = clampf_local(new_left, 0.0f, frame_w - 1.0f);
            new_top = clampf_local(new_top, 0.0f, frame_h - 1.0f);
            new_right = clampf_local(new_right, new_left + 1.0f, frame_w);
            new_bottom = clampf_local(new_bottom, new_top + 1.0f, frame_h);

            obj->rect_params.left = new_left;
            obj->rect_params.top = new_top;
            obj->rect_params.width = new_right - new_left;
            obj->rect_params.height = new_bottom - new_top;
            style_green(obj);
            ++count;
        }
    }
    return count;
}

int camera_v2_copy_tracks(uintptr_t buffer_ptr,
                          CameraV2TrackRow *rows,
                          int max_rows) {
    if (!buffer_ptr || !rows || max_rows <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float min_tracker_conf = env_float_local(
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF", 0.28f, 0.0f, 1.0f
    );
    int count = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fnode->data;
        if (!frame_meta) continue;
        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!is_displayable_track(obj, min_tracker_conf)) continue;
            if (obj->rect_params.border_width == 0) continue;
            if (count >= max_rows) return count;
            CameraV2TrackRow *dst = &rows[count++];
            dst->object_id = (uint64_t)obj->object_id;
            dst->frame_num = (uint64_t)frame_meta->frame_num;
            dst->source_id = (uint32_t)frame_meta->source_id;
            dst->left = obj->rect_params.left;
            dst->top = obj->rect_params.top;
            dst->width = obj->rect_params.width;
            dst->height = obj->rect_params.height;
            dst->confidence = obj->confidence;
            dst->tracker_confidence = obj->tracker_confidence;
        }
    }
    return count;
}

/* Expand only the rectangle that will be drawn. This function is deliberately
 * called after count/heatmap sampling, so tracker truth, people counts and the
 * floor-contact heatmap remain based on the unmodified current NvDCF rectangle.
 * Wide seated/reclined people get a little more side room for elbows/arms. */
int camera_v2_expand_display_boxes(uintptr_t buffer_ptr,
                                   float base_side_margin,
                                   float base_top_margin,
                                   float base_bottom_margin) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    base_side_margin = clampf_local(base_side_margin, 0.0f, 0.20f);
    base_top_margin = clampf_local(base_top_margin, 0.0f, 0.15f);
    base_bottom_margin = clampf_local(base_bottom_margin, 0.0f, 0.20f);
    const float min_tracker_conf = env_float_local(
        "CAMERA_V2_MIN_DISPLAY_TRACK_CONF", 0.28f, 0.0f, 1.0f
    );
    int expanded = 0;

    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fnode->data;
        if (!frame_meta) continue;

        float frame_w = frame_meta->pipeline_width > 1 ? (float) frame_meta->pipeline_width
                                                        : (float) frame_meta->source_frame_width;
        float frame_h = frame_meta->pipeline_height > 1 ? (float) frame_meta->pipeline_height
                                                         : (float) frame_meta->source_frame_height;
        if (frame_w <= 1.0f || frame_h <= 1.0f) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!is_displayable_track(obj, min_tracker_conf)) continue;

            const float left = obj->rect_params.left;
            const float top = obj->rect_params.top;
            const float width = obj->rect_params.width;
            const float height = obj->rect_params.height;
            if (width <= 2.0f || height <= 4.0f) continue;

            const float aspect = width / height;
            const float wide_extra = clampf_local((aspect - 0.62f) * 0.12f, 0.0f, 0.055f);
            const float side_margin = clampf_local(base_side_margin + wide_extra, 0.0f, 0.18f);
            const float top_margin = clampf_local(
                base_top_margin + (aspect > 0.90f ? 0.015f : 0.0f), 0.0f, 0.12f
            );
            const float bottom_margin = clampf_local(
                base_bottom_margin + (aspect > 0.82f ? 0.020f : 0.0f), 0.0f, 0.18f
            );

            float side_px = width * side_margin;
            float top_px = height * top_margin;
            float bottom_px = height * bottom_margin;
            if (side_px < 5.0f) side_px = 5.0f;
            if (top_px < 4.0f) top_px = 4.0f;
            if (bottom_px < 7.0f) bottom_px = 7.0f;

            float new_left = clampf_local(left - side_px, 0.0f, frame_w - 1.0f);
            float new_top = clampf_local(top - top_px, 0.0f, frame_h - 1.0f);
            float new_right = clampf_local(left + width + side_px, new_left + 1.0f, frame_w);
            float new_bottom = clampf_local(top + height + bottom_px, new_top + 1.0f, frame_h);

            obj->rect_params.left = new_left;
            obj->rect_params.top = new_top;
            obj->rect_params.width = new_right - new_left;
            obj->rect_params.height = new_bottom - new_top;
            ++expanded;
        }
    }
    return expanded;
}

uint64_t camera_v2_shadow_promoted_total(void) {
    return 0;
}

int camera_v2_count_tracked(uintptr_t buffer_ptr) {
    return camera_v2_style_and_count_tracked(buffer_ptr);
}
