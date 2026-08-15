#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define MAX_VISUAL_STATES 512
#define MAX_REAL_BOXES_PER_FRAME 64
#define DISPLAY_HOLD_FRAMES 5

typedef struct {
    int valid;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_frame_num;
    float cx;
    float cy;
    float vx;
    float vy;
    float display_w;
    float display_h;
} VisualTrackState;

static VisualTrackState g_visual_states[MAX_VISUAL_STATES];

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
        obj->rect_params.border_width = 3;
        obj->rect_params.border_color.red = 0.10;
        obj->rect_params.border_color.green = 1.00;
        obj->rect_params.border_color.blue = 0.15;
        obj->rect_params.border_color.alpha = 1.00;
        obj->rect_params.has_bg_color = 0;

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

/* Emulate a primary detector's per-frame metadata contract for nvtracker. */
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

static float clampf_local(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static float rect_iou(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2) {
    float x1 = ax1 > bx1 ? ax1 : bx1;
    float y1 = ay1 > by1 ? ay1 : by1;
    float x2 = ax2 < bx2 ? ax2 : bx2;
    float y2 = ay2 < by2 ? ay2 : by2;
    float iw = x2 - x1;
    float ih = y2 - y1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    float inter = iw * ih;
    float aa = (ax2 - ax1) * (ay2 - ay1);
    float bb = (bx2 - bx1) * (by2 - by1);
    float uni = aa + bb - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

static int find_visual_state(unsigned int source_id, uint64_t object_id) {
    int free_index = -1;
    int oldest_index = 0;
    uint64_t oldest_frame = UINT64_MAX;
    for (int i = 0; i < MAX_VISUAL_STATES; ++i) {
        VisualTrackState *s = &g_visual_states[i];
        if (s->valid && s->source_id == source_id && s->object_id == object_id) {
            return i;
        }
        if (!s->valid && free_index < 0) free_index = i;
        if (s->valid && s->last_frame_num < oldest_frame) {
            oldest_frame = s->last_frame_num;
            oldest_index = i;
        }
    }
    return free_index >= 0 ? free_index : oldest_index;
}

static void style_green(NvDsObjectMeta *obj) {
    obj->rect_params.border_width = 3;
    obj->rect_params.border_color.red = 0.10;
    obj->rect_params.border_color.green = 1.00;
    obj->rect_params.border_color.blue = 0.15;
    obj->rect_params.border_color.alpha = 1.00;
    obj->rect_params.has_bg_color = 0;
}

static void set_display_rect(NvDsObjectMeta *obj,
                             float cx,
                             float cy,
                             float width,
                             float height,
                             float frame_w,
                             float frame_h) {
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    if (left < 0.0f) {
        right -= left;
        left = 0.0f;
    }
    if (right > frame_w) {
        float shift = right - frame_w;
        left -= shift;
        right -= shift;
    }
    if (top < 0.0f) {
        bottom -= top;
        top = 0.0f;
    }
    if (bottom > frame_h) {
        float shift = bottom - frame_h;
        top -= shift;
        bottom -= shift;
    }

    left = clampf_local(left, 0.0f, frame_w - 1.0f);
    top = clampf_local(top, 0.0f, frame_h - 1.0f);
    right = clampf_local(right, left + 1.0f, frame_w);
    bottom = clampf_local(bottom, top + 1.0f, frame_h);

    obj->rect_params.left = left;
    obj->rect_params.top = top;
    obj->rect_params.width = right - left;
    obj->rect_params.height = bottom - top;
    style_green(obj);
}

/*
 * Post-tracker visualization continuity layer.
 *
 * NvDCF remains the only real tracker. This function runs AFTER nvtracker and only
 * changes rect_params used by downstream OSD. It never feeds the adjusted boxes
 * back into NvDCF.
 *
 * 1) Position gets a small velocity lead (~1 video frame) so the rendered box does
 *    not visibly trail a walking person.
 * 2) Display width/height shrink slowly but expand immediately, so a raised arm or
 *    changing pose does not make the box collapse around the torso.
 * 3) If NvDCF suppresses current-frame output for a very short confidence dip, the
 *    last tracked ID is rendered for at most DISPLAY_HOLD_FRAMES with motion
 *    prediction. This mirrors the short lost-track buffer used by mature MOT
 *    systems, but is display-only and cannot create a tracker target.
 */
int camera_v2_style_and_count_tracked(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float side_margin = 0.10f;
    const float top_margin = 0.08f;
    const float bottom_margin = 0.14f;
    const float lead_frames = 1.10f;
    const float velocity_alpha = 0.45f;
    const float shrink_per_frame = 0.965f;

    int count = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) fnode->data;
        if (!frame_meta) continue;

        unsigned int source_id = frame_meta->source_id;
        uint64_t frame_num = (uint64_t) frame_meta->frame_num;
        float frame_w = (float) frame_meta->source_frame_width;
        float frame_h = (float) frame_meta->source_frame_height;
        if (frame_w <= 1.0f) frame_w = (float) frame_meta->pipeline_width;
        if (frame_h <= 1.0f) frame_h = (float) frame_meta->pipeline_height;
        if (frame_w <= 1.0f || frame_h <= 1.0f) continue;

        unsigned char seen[MAX_VISUAL_STATES];
        memset(seen, 0, sizeof(seen));
        float real_boxes[MAX_REAL_BOXES_PER_FRAME][4];
        int real_count = 0;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;

            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float width = obj->rect_params.width;
            float height = obj->rect_params.height;
            if (width <= 1.0f || height <= 1.0f) continue;

            float raw_cx = left + width * 0.5f;
            float raw_cy = top + height * 0.5f;
            int idx = find_visual_state(source_id, (uint64_t) obj->object_id);
            VisualTrackState *s = &g_visual_states[idx];

            float vx = 0.0f;
            float vy = 0.0f;
            float base_w = width * (1.0f + side_margin * 2.0f);
            float base_h = height * (1.0f + top_margin + bottom_margin);
            float display_w = base_w;
            float display_h = base_h;

            if (s->valid && s->source_id == source_id && s->object_id == (uint64_t) obj->object_id) {
                uint64_t delta_frames = frame_num > s->last_frame_num ? frame_num - s->last_frame_num : 1;
                if (delta_frames > 8) delta_frames = 8;
                float measured_vx = (raw_cx - s->cx) / (float) delta_frames;
                float measured_vy = (raw_cy - s->cy) / (float) delta_frames;
                float max_vx = width * 0.45f;
                float max_vy = height * 0.45f;
                measured_vx = clampf_local(measured_vx, -max_vx, max_vx);
                measured_vy = clampf_local(measured_vy, -max_vy, max_vy);
                vx = s->vx * (1.0f - velocity_alpha) + measured_vx * velocity_alpha;
                vy = s->vy * (1.0f - velocity_alpha) + measured_vy * velocity_alpha;

                float retained_w = s->display_w;
                float retained_h = s->display_h;
                for (uint64_t k = 0; k < delta_frames; ++k) {
                    retained_w *= shrink_per_frame;
                    retained_h *= shrink_per_frame;
                }
                if (retained_w > display_w) display_w = retained_w;
                if (retained_h > display_h) display_h = retained_h;
            }

            float display_cx = raw_cx + vx * lead_frames;
            float display_cy = raw_cy + vy * lead_frames;
            set_display_rect(obj, display_cx, display_cy, display_w, display_h, frame_w, frame_h);

            s->valid = 1;
            s->source_id = source_id;
            s->object_id = (uint64_t) obj->object_id;
            s->last_frame_num = frame_num;
            s->cx = raw_cx;
            s->cy = raw_cy;
            s->vx = vx;
            s->vy = vy;
            s->display_w = display_w;
            s->display_h = display_h;
            seen[idx] = 1;

            if (real_count < MAX_REAL_BOXES_PER_FRAME) {
                real_boxes[real_count][0] = obj->rect_params.left;
                real_boxes[real_count][1] = obj->rect_params.top;
                real_boxes[real_count][2] = obj->rect_params.left + obj->rect_params.width;
                real_boxes[real_count][3] = obj->rect_params.top + obj->rect_params.height;
                ++real_count;
            }
            ++count;
        }

        /* Short display-only hold for tracker output gaps / brief shadow state. */
        for (int i = 0; i < MAX_VISUAL_STATES; ++i) {
            VisualTrackState *s = &g_visual_states[i];
            if (!s->valid || s->source_id != source_id || seen[i]) continue;
            if (frame_num <= s->last_frame_num) continue;

            uint64_t age = frame_num - s->last_frame_num;
            if (age > 60) {
                s->valid = 0;
                continue;
            }
            if (age > DISPLAY_HOLD_FRAMES) continue;

            float decay = 1.0f - 0.025f * (float) age;
            if (decay < 0.86f) decay = 0.86f;
            float cx = s->cx + s->vx * (float) age * 0.92f;
            float cy = s->cy + s->vy * (float) age * 0.92f;
            float width = s->display_w * decay;
            float height = s->display_h * decay;
            float left = cx - width * 0.5f;
            float top = cy - height * 0.5f;
            float right = cx + width * 0.5f;
            float bottom = cy + height * 0.5f;

            int overlaps_real = 0;
            for (int r = 0; r < real_count; ++r) {
                if (rect_iou(left, top, right, bottom,
                             real_boxes[r][0], real_boxes[r][1],
                             real_boxes[r][2], real_boxes[r][3]) >= 0.32f) {
                    overlaps_real = 1;
                    break;
                }
            }
            if (overlaps_real) continue;

            NvDsObjectMeta *obj = nvds_acquire_obj_meta_from_pool(batch_meta);
            if (!obj) continue;
            obj->unique_component_id = 92;
            obj->class_id = 0;
            obj->object_id = s->object_id;
            obj->confidence = -0.1f;
            obj->tracker_confidence = -0.1f;
            strncpy(obj->obj_label, "Person", MAX_LABEL_SIZE - 1);
            obj->obj_label[MAX_LABEL_SIZE - 1] = '\0';
            set_display_rect(obj, cx, cy, width, height, frame_w, frame_h);
            nvds_add_obj_meta_to_frame(frame_meta, obj, NULL);
            ++count;
        }
    }
    return count;
}

int camera_v2_count_tracked(uintptr_t buffer_ptr) {
    return camera_v2_style_and_count_tracked(buffer_ptr);
}
