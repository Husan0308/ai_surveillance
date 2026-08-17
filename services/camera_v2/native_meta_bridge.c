#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

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

/* Sentinel VMS overlay: use the same teal accent as the Qt shell instead of the
 * diagnostic neon-green rectangle. A very light translucent fill keeps the box
 * readable on both bright and dark CCTV regions without hiding the person. */
static void style_sentinel(NvDsObjectMeta *obj) {
    obj->rect_params.border_width = 2;
    obj->rect_params.border_color.red = 0.224;
    obj->rect_params.border_color.green = 0.851;
    obj->rect_params.border_color.blue = 0.773;
    obj->rect_params.border_color.alpha = 0.98;
    obj->rect_params.has_bg_color = 1;
    obj->rect_params.bg_color.red = 0.035;
    obj->rect_params.bg_color.green = 0.090;
    obj->rect_params.bg_color.blue = 0.105;
    obj->rect_params.bg_color.alpha = 0.055;
}

static float clampf_local(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
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
        style_sentinel(obj);

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

/* Emulate a primary detector result on exactly the live source frame where the
 * asynchronous YOLO observation is attached. Empty detector results are valid:
 * bInferDone is TRUE with zero object meta, matching nvinfer interval semantics. */
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

/*
 * Live OSD helper AFTER nvtracker.
 *
 * Strict rule: never invent an object here. No timer hold, no stale shadow-history
 * promotion, no synthetic prediction metadata. Only current-frame NvDsObjectMeta
 * produced by NvDCF is styled for display. This removes lingering giant rectangles
 * when a person has already left the camera view.
 *
 * NvDCF keeps its tight bbox internally. We enlarge rect_params only for display so
 * hands/head/feet have a small safety margin without contaminating DCF features.
 */
int camera_v2_style_and_count_tracked(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float side_margin = 0.07f;
    const float top_margin = 0.05f;
    const float bottom_margin = 0.10f;
    int count = 0;

    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fnode->data;
        if (!frame_meta) continue;

        float frame_w = (float) frame_meta->source_frame_width;
        float frame_h = (float) frame_meta->source_frame_height;
        if (frame_w <= 1.0f) frame_w = (float) frame_meta->pipeline_width;
        if (frame_h <= 1.0f) frame_h = (float) frame_meta->pipeline_height;
        if (frame_w <= 1.0f || frame_h <= 1.0f) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;

            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float width = obj->rect_params.width;
            float height = obj->rect_params.height;
            if (width <= 1.0f || height <= 1.0f) continue;

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
            style_sentinel(obj);
            ++count;
        }
    }
    return count;
}

/* Kept for Python ABI compatibility with the previous diagnostic build. */
uint64_t camera_v2_shadow_promoted_total(void) {
    return 0;
}

int camera_v2_count_tracked(uintptr_t buffer_ptr) {
    return camera_v2_style_and_count_tracked(buffer_ptr);
}
