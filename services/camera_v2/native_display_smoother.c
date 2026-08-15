#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define MAX_SMOOTH_STATES 512

typedef struct {
    int valid;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_frame_num;
    float target_cx;
    float target_cy;
    float display_cx;
    float display_cy;
    float display_w;
    float display_h;
    float vx;
    float vy;
} SmoothBoxState;

static SmoothBoxState g_smooth_states[MAX_SMOOTH_STATES];

static float clampf_smooth(float value, float low, float high) {
    if (value < low) return low;
    if (value > high) return high;
    return value;
}

static int find_state(unsigned int source_id, uint64_t object_id) {
    int free_index = -1;
    int oldest_index = 0;
    uint64_t oldest_frame = UINT64_MAX;
    for (int i = 0; i < MAX_SMOOTH_STATES; ++i) {
        SmoothBoxState *s = &g_smooth_states[i];
        if (s->valid && s->source_id == source_id && s->object_id == object_id) return i;
        if (!s->valid && free_index < 0) free_index = i;
        if (s->valid && s->last_frame_num < oldest_frame) {
            oldest_frame = s->last_frame_num;
            oldest_index = i;
        }
    }
    return free_index >= 0 ? free_index : oldest_index;
}

static float alpha_for_steps(float one_step_alpha, uint64_t steps) {
    float remain = 1.0f;
    for (uint64_t i = 0; i < steps; ++i) remain *= (1.0f - one_step_alpha);
    return 1.0f - remain;
}

static void write_rect(NvDsObjectMeta *obj,
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

    if (left < 0.0f) { right -= left; left = 0.0f; }
    if (right > frame_w) { float d = right - frame_w; left -= d; right -= d; }
    if (top < 0.0f) { bottom -= top; top = 0.0f; }
    if (bottom > frame_h) { float d = bottom - frame_h; top -= d; bottom -= d; }

    left = clampf_smooth(left, 0.0f, frame_w - 1.0f);
    top = clampf_smooth(top, 0.0f, frame_h - 1.0f);
    right = clampf_smooth(right, left + 1.0f, frame_w);
    bottom = clampf_smooth(bottom, top + 1.0f, frame_h);

    obj->rect_params.left = left;
    obj->rect_params.top = top;
    obj->rect_params.width = right - left;
    obj->rect_params.height = bottom - top;
}

/*
 * Display-only smoother for REAL current-frame NvDCF objects.
 *
 * It never creates metadata and therefore cannot leave a ghost rectangle behind.
 * Position follows quickly; size changes are eased enough to avoid visible jumps
 * when a person raises an arm, turns, sits or stands. A very small velocity lead
 * offsets the interpolation latency without shooting the box ahead of the target.
 */
int camera_v2_smooth_display_boxes(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float center_alpha_1f = 0.86f;
    const float velocity_alpha = 0.45f;
    const float lead_frames = 0.12f;
    const float expand_alpha_1f = 0.34f;
    const float shrink_alpha_1f = 0.22f;
    int smoothed = 0;

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

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;

            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float width = obj->rect_params.width;
            float height = obj->rect_params.height;
            if (width <= 1.0f || height <= 1.0f) continue;

            float target_cx = left + width * 0.5f;
            float target_cy = top + height * 0.5f;
            int idx = find_state(source_id, (uint64_t) obj->object_id);
            SmoothBoxState *s = &g_smooth_states[idx];

            if (!s->valid || s->source_id != source_id || s->object_id != (uint64_t) obj->object_id ||
                (frame_num > s->last_frame_num && frame_num - s->last_frame_num > 20)) {
                memset(s, 0, sizeof(*s));
                s->valid = 1;
                s->source_id = source_id;
                s->object_id = (uint64_t) obj->object_id;
                s->last_frame_num = frame_num;
                s->target_cx = target_cx;
                s->target_cy = target_cy;
                s->display_cx = target_cx;
                s->display_cy = target_cy;
                s->display_w = width;
                s->display_h = height;
                write_rect(obj, target_cx, target_cy, width, height, frame_w, frame_h);
                ++smoothed;
                continue;
            }

            uint64_t delta_frames = frame_num > s->last_frame_num ? frame_num - s->last_frame_num : 1;
            if (delta_frames > 6) delta_frames = 6;

            float measured_vx = (target_cx - s->target_cx) / (float) delta_frames;
            float measured_vy = (target_cy - s->target_cy) / (float) delta_frames;
            float max_vx = width * 0.30f + 3.0f;
            float max_vy = height * 0.30f + 3.0f;
            measured_vx = clampf_smooth(measured_vx, -max_vx, max_vx);
            measured_vy = clampf_smooth(measured_vy, -max_vy, max_vy);
            s->vx = s->vx * (1.0f - velocity_alpha) + measured_vx * velocity_alpha;
            s->vy = s->vy * (1.0f - velocity_alpha) + measured_vy * velocity_alpha;

            float desired_cx = target_cx + s->vx * lead_frames;
            float desired_cy = target_cy + s->vy * lead_frames;
            float center_alpha = alpha_for_steps(center_alpha_1f, delta_frames);
            float expand_alpha = alpha_for_steps(expand_alpha_1f, delta_frames);
            float shrink_alpha = alpha_for_steps(shrink_alpha_1f, delta_frames);

            s->display_cx += (desired_cx - s->display_cx) * center_alpha;
            s->display_cy += (desired_cy - s->display_cy) * center_alpha;
            s->display_w += (width - s->display_w) * (width >= s->display_w ? expand_alpha : shrink_alpha);
            s->display_h += (height - s->display_h) * (height >= s->display_h ? expand_alpha : shrink_alpha);

            s->target_cx = target_cx;
            s->target_cy = target_cy;
            s->last_frame_num = frame_num;
            write_rect(obj, s->display_cx, s->display_cy, s->display_w, s->display_h, frame_w, frame_h);
            ++smoothed;
        }

        /* State alone is harmless, but retire it quickly so recycled IDs start clean. */
        for (int i = 0; i < MAX_SMOOTH_STATES; ++i) {
            SmoothBoxState *s = &g_smooth_states[i];
            if (!s->valid || s->source_id != source_id) continue;
            if (frame_num > s->last_frame_num && frame_num - s->last_frame_num > 40) s->valid = 0;
        }
    }
    return smoothed;
}
