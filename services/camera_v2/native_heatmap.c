#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define HEAT_MAX_SOURCES 16
#define HEAT_GRID_W 48
#define HEAT_GRID_H 27
#define HEAT_MAX_TRACKS 512
#define HEAT_MAX_POINTS_PER_SOURCE 96

typedef struct {
    int valid;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_frame_num;
    uint64_t last_vote_frame;
    uint64_t last_presence_frame;
    uint64_t moving_until;
    unsigned int motion_votes;
    float smooth_gx;
    float smooth_gy;
    float anchor_gx;
    float anchor_gy;
    float deposit_gx;
    float deposit_gy;
} HeatTrackState;

typedef struct {
    float score;
    float value;
    int gx;
    int gy;
} HeatCandidate;

static float g_heat[HEAT_MAX_SOURCES][HEAT_GRID_H][HEAT_GRID_W];
static uint32_t g_touch[HEAT_MAX_SOURCES][HEAT_GRID_H][HEAT_GRID_W];
static uint64_t g_last_frame[HEAT_MAX_SOURCES];
static HeatTrackState g_tracks[HEAT_MAX_TRACKS];
static uint64_t g_rendered_points_total = 0;

/* Reference-look defaults: one/few people remain blue/cyan; repeatedly occupied
 * floor cells warm through green/yellow and sustained traffic eventually turns red. */
static float g_deposit = 0.0030f;
static float g_decay = 0.999968f;
static float g_low = 0.00050f;
static float g_yellow = 0.070f;
static float g_red = 0.200f;
static unsigned int g_max_points_per_source = 72;

/* Every real NvDCF person contributes a very small presence pulse. Motion adds a
 * denser interpolated trail. This guarantees the heatmap covers every tracked
 * person while preventing one stationary person from becoming red immediately. */
static const uint64_t PRESENCE_INTERVAL_FRAMES = 12; /* ~0.6 s @20 FPS */
static const float PRESENCE_INITIAL_SCALE = 0.60f;
static const float PRESENCE_HOLD_SCALE = 0.14f;
static const float MOTION_CONFIRM_DIST2 = 0.0576f; /* 0.24^2 cell */
static const float DEPOSIT_DIST2 = 0.0049f;        /* 0.07^2 cell */
static const uint64_t MOTION_VOTE_WINDOW = 20;    /* ~1.0 s @20 FPS */
static const uint64_t MOVING_HOLD_FRAMES = 14;
static const float FOOT_EMA_ALPHA = 0.58f;
static const float FOOT_LIFT_RATIO = 0.08f;

static float clampf_heat(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float sqr(float v) { return v * v; }

void camera_v2_heatmap_configure(float deposit,
                                 float decay,
                                 float low_threshold,
                                 float yellow_threshold,
                                 float red_threshold,
                                 unsigned int max_points_per_source) {
    g_deposit = clampf_heat(deposit, 0.0003f, 0.03f);
    g_decay = clampf_heat(decay, 0.95f, 0.9999999f);
    g_low = clampf_heat(low_threshold, 0.00025f, 0.50f);
    g_yellow = clampf_heat(yellow_threshold, g_low + 0.001f, 0.95f);
    g_red = clampf_heat(red_threshold, g_yellow + 0.002f, 1.0f);
    if (max_points_per_source < 12) max_points_per_source = 12;
    if (max_points_per_source > HEAT_MAX_POINTS_PER_SOURCE) {
        max_points_per_source = HEAT_MAX_POINTS_PER_SOURCE;
    }
    g_max_points_per_source = max_points_per_source;
}

void camera_v2_heatmap_reset(void) {
    memset(g_heat, 0, sizeof(g_heat));
    memset(g_touch, 0, sizeof(g_touch));
    memset(g_last_frame, 0, sizeof(g_last_frame));
    memset(g_tracks, 0, sizeof(g_tracks));
    g_rendered_points_total = 0;
}

static int find_track(unsigned int source_id, uint64_t object_id) {
    int free_index = -1;
    int oldest_index = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < HEAT_MAX_TRACKS; ++i) {
        HeatTrackState *s = &g_tracks[i];
        if (s->valid && s->source_id == source_id && s->object_id == object_id) return i;
        if (!s->valid && free_index < 0) free_index = i;
        if (s->valid && s->last_frame_num < oldest) {
            oldest = s->last_frame_num;
            oldest_index = i;
        }
    }
    return free_index >= 0 ? free_index : oldest_index;
}

static void reset_track(HeatTrackState *s,
                        unsigned int source_id,
                        uint64_t object_id,
                        uint64_t frame_num,
                        float gx,
                        float gy) {
    memset(s, 0, sizeof(*s));
    s->valid = 1;
    s->source_id = source_id;
    s->object_id = object_id;
    s->last_frame_num = frame_num;
    s->last_vote_frame = frame_num;
    s->last_presence_frame = frame_num;
    s->smooth_gx = gx;
    s->smooth_gy = gy;
    s->anchor_gx = gx;
    s->anchor_gy = gy;
    s->deposit_gx = gx;
    s->deposit_gy = gy;
}

static void decay_source(unsigned int source_id, uint64_t frame_num) {
    if (source_id >= HEAT_MAX_SOURCES) return;
    uint64_t last = g_last_frame[source_id];
    if (last == 0 || frame_num <= last) {
        g_last_frame[source_id] = frame_num;
        return;
    }

    uint64_t delta = frame_num - last;
    if (delta > 4000) delta = 4000;
    float factor = 1.0f;
    for (uint64_t i = 0; i < delta; ++i) factor *= g_decay;

    for (int y = 0; y < HEAT_GRID_H; ++y) {
        for (int x = 0; x < HEAT_GRID_W; ++x) {
            float v = g_heat[source_id][y][x] * factor;
            g_heat[source_id][y][x] = v < g_low * 0.42f ? 0.0f : v;
        }
    }
    g_last_frame[source_id] = frame_num;
}

static void deposit_point(unsigned int source_id,
                          float gx,
                          float gy,
                          uint32_t frame_num,
                          float amount_scale) {
    static const float kernel[5][5] = {
        {0.010f, 0.025f, 0.045f, 0.025f, 0.010f},
        {0.025f, 0.080f, 0.180f, 0.080f, 0.025f},
        {0.045f, 0.180f, 1.000f, 0.180f, 0.045f},
        {0.025f, 0.080f, 0.180f, 0.080f, 0.025f},
        {0.010f, 0.025f, 0.045f, 0.025f, 0.010f},
    };
    amount_scale = clampf_heat(amount_scale, 0.08f, 1.10f);
    int cx = (int)(gx + 0.5f);
    int cy = (int)(gy + 0.5f);

    for (int ky = -2; ky <= 2; ++ky) {
        int y = cy + ky;
        if (y < 0 || y >= HEAT_GRID_H) continue;
        for (int kx = -2; kx <= 2; ++kx) {
            int x = cx + kx;
            if (x < 0 || x >= HEAT_GRID_W) continue;
            float add = g_deposit * amount_scale * kernel[ky + 2][kx + 2];
            float next = g_heat[source_id][y][x] + add;
            g_heat[source_id][y][x] = next > 1.0f ? 1.0f : next;
            g_touch[source_id][y][x] = frame_num;
        }
    }
}

static void deposit_segment(unsigned int source_id,
                            float x0,
                            float y0,
                            float x1,
                            float y1,
                            uint32_t frame_num) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float adx = dx < 0.0f ? -dx : dx;
    float ady = dy < 0.0f ? -dy : dy;
    float span = adx > ady ? adx : ady;

    int steps = (int)(span * 5.0f) + 2;
    if (steps < 2) steps = 2;
    if (steps > 28) steps = 28;
    float scale = clampf_heat(0.40f + span * 0.18f, 0.40f, 1.00f);

    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / (float)steps;
        deposit_point(source_id, x0 + dx * t, y0 + dy * t, frame_num, scale);
    }
}

/* Accumulate from every current real NvDCF person. A new/standing person gets a
 * faint blue presence halo; a moving person additionally paints a continuous path. */
int camera_v2_heatmap_update(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int heat_updates = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;
        unsigned int source_id = frame_meta->source_id;
        if (source_id >= HEAT_MAX_SOURCES) continue;
        uint64_t frame_num = (uint64_t)frame_meta->frame_num;
        decay_source(source_id, frame_num);

        float frame_w = frame_meta->pipeline_width > 1 ? (float)frame_meta->pipeline_width
                                                        : (float)frame_meta->source_frame_width;
        float frame_h = frame_meta->pipeline_height > 1 ? (float)frame_meta->pipeline_height
                                                         : (float)frame_meta->source_frame_height;
        if (frame_w <= 1.0f || frame_h <= 1.0f) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            float w = obj->rect_params.width;
            float h = obj->rect_params.height;
            if (w <= 2.0f || h <= 4.0f) continue;

            float foot_x = obj->rect_params.left + w * 0.5f;
            float foot_y = obj->rect_params.top + h * (1.0f - FOOT_LIFT_RATIO);
            float raw_gx = clampf_heat((foot_x / frame_w) * (float)(HEAT_GRID_W - 1),
                                       0.0f, (float)(HEAT_GRID_W - 1));
            float raw_gy = clampf_heat((foot_y / frame_h) * (float)(HEAT_GRID_H - 1),
                                       0.0f, (float)(HEAT_GRID_H - 1));

            int idx = find_track(source_id, (uint64_t)obj->object_id);
            HeatTrackState *s = &g_tracks[idx];
            if (!s->valid || s->source_id != source_id || s->object_id != (uint64_t)obj->object_id ||
                frame_num < s->last_frame_num || frame_num - s->last_frame_num > 18) {
                reset_track(s, source_id, (uint64_t)obj->object_id, frame_num, raw_gx, raw_gy);
                deposit_point(source_id, raw_gx, raw_gy, (uint32_t)frame_num, PRESENCE_INITIAL_SCALE);
                ++heat_updates;
                continue;
            }

            float gx = FOOT_EMA_ALPHA * raw_gx + (1.0f - FOOT_EMA_ALPHA) * s->smooth_gx;
            float gy = FOOT_EMA_ALPHA * raw_gy + (1.0f - FOOT_EMA_ALPHA) * s->smooth_gy;
            s->smooth_gx = gx;
            s->smooth_gy = gy;

            /* Low-rate occupancy pulse: even a stationary/seated tracked person is
             * represented. The scale is intentionally tiny so density, not one
             * person's dwell time, is what pushes a place toward yellow/red. */
            if (frame_num >= s->last_presence_frame + PRESENCE_INTERVAL_FRAMES) {
                deposit_point(source_id, gx, gy, (uint32_t)frame_num, PRESENCE_HOLD_SCALE);
                s->last_presence_frame = frame_num;
                ++heat_updates;
            }

            float anchor_dist2 = sqr(gx - s->anchor_gx) + sqr(gy - s->anchor_gy);
            if (anchor_dist2 >= MOTION_CONFIRM_DIST2) {
                if (frame_num - s->last_vote_frame <= MOTION_VOTE_WINDOW) {
                    s->motion_votes += 1;
                } else {
                    s->motion_votes = 1;
                }
                s->last_vote_frame = frame_num;
                s->anchor_gx = gx;
                s->anchor_gy = gy;
                if (s->motion_votes >= 2) {
                    s->moving_until = frame_num + MOVING_HOLD_FRAMES;
                    s->motion_votes = 0;
                }
            }

            if (frame_num <= s->moving_until) {
                float dist2 = sqr(gx - s->deposit_gx) + sqr(gy - s->deposit_gy);
                if (dist2 >= DEPOSIT_DIST2) {
                    deposit_segment(source_id, s->deposit_gx, s->deposit_gy, gx, gy, (uint32_t)frame_num);
                    s->deposit_gx = gx;
                    s->deposit_gy = gy;
                    s->moving_until = frame_num + MOVING_HOLD_FRAMES;
                    ++heat_updates;
                }
            } else if (anchor_dist2 < MOTION_CONFIRM_DIST2) {
                s->deposit_gx = gx;
                s->deposit_gy = gy;
            }

            s->last_frame_num = frame_num;
        }

        for (int i = 0; i < HEAT_MAX_TRACKS; ++i) {
            HeatTrackState *s = &g_tracks[i];
            if (!s->valid || s->source_id != source_id) continue;
            if (frame_num > s->last_frame_num && frame_num - s->last_frame_num > 100) {
                s->valid = 0;
            }
        }
    }
    return heat_updates;
}

static float smoothed_value(unsigned int sid, int gx, int gy) {
    static const float w[3][3] = {
        {0.05f, 0.10f, 0.05f},
        {0.10f, 0.40f, 0.10f},
        {0.05f, 0.10f, 0.05f},
    };
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        int y = gy + ky;
        if (y < 0 || y >= HEAT_GRID_H) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int x = gx + kx;
            if (x < 0 || x >= HEAT_GRID_W) continue;
            sum += g_heat[sid][y][x] * w[ky + 1][kx + 1];
        }
    }
    return sum;
}

static void push_candidate(HeatCandidate *items,
                           unsigned int *count,
                           unsigned int max_count,
                           HeatCandidate candidate) {
    if (*count < max_count) {
        items[*count] = candidate;
        ++(*count);
        return;
    }
    unsigned int min_index = 0;
    float min_score = items[0].score;
    for (unsigned int i = 1; i < max_count; ++i) {
        if (items[i].score < min_score) {
            min_score = items[i].score;
            min_index = i;
        }
    }
    if (candidate.score > min_score) items[min_index] = candidate;
}

static void heat_color(float value, NvOSD_ColorParams *color) {
    float r = 0.00f, g = 0.62f, b = 1.00f;
    if (value < g_yellow) {
        float u = (value - g_low) / (g_yellow - g_low + 0.000001f);
        u = clampf_heat(u, 0.0f, 1.0f);
        r = 0.00f + 0.16f * u;
        g = 0.62f + 0.32f * u;
        b = 1.00f - 0.36f * u;
    } else if (value < g_red) {
        float u = (value - g_yellow) / (g_red - g_yellow + 0.000001f);
        u = clampf_heat(u, 0.0f, 1.0f);
        r = 0.16f + 0.84f * u;
        g = 0.94f - 0.10f * u;
        b = 0.64f * (1.0f - u);
    } else {
        float u = (value - g_red) / (1.0f - g_red + 0.000001f);
        u = clampf_heat(u, 0.0f, 1.0f);
        r = 1.00f;
        g = 0.84f * (1.0f - u);
        b = 0.00f;
    }

    float strength = clampf_heat(value / g_red, 0.0f, 1.0f);
    color->red = r;
    color->green = g;
    color->blue = b;
    color->alpha = 0.055f + 0.175f * strength;
}

static NvDsDisplayMeta *new_display_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *anchor) {
    NvDsDisplayMeta *meta = nvds_acquire_display_meta_from_pool(batch_meta);
    if (!meta) return NULL;
    meta->num_rects = 0;
    meta->num_labels = 0;
    meta->num_lines = 0;
    meta->num_arrows = 0;
    meta->num_circles = 0;
    nvds_add_display_meta_to_frame(anchor, meta);
    return meta;
}

int camera_v2_heatmap_render(uintptr_t buffer_ptr,
                             unsigned int wall_width,
                             unsigned int wall_height,
                             unsigned int rows,
                             unsigned int columns,
                             unsigned int source_count) {
    if (!buffer_ptr || wall_width < 2 || wall_height < 2 || rows == 0 || columns == 0) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta || !batch_meta->frame_meta_list) return -1;
    NvDsFrameMeta *anchor = (NvDsFrameMeta *)batch_meta->frame_meta_list->data;
    if (!anchor) return -1;

    unsigned int max_sources = rows * columns;
    if (source_count < max_sources) max_sources = source_count;
    if (max_sources > HEAT_MAX_SOURCES) max_sources = HEAT_MAX_SOURCES;

    float tile_w = (float)wall_width / (float)columns;
    float tile_h = (float)wall_height / (float)rows;
    NvDsDisplayMeta *display = NULL;
    int rendered = 0;

    for (unsigned int sid = 0; sid < max_sources; ++sid) {
        HeatCandidate candidates[HEAT_MAX_POINTS_PER_SOURCE];
        unsigned int count = 0;
        uint64_t current_frame = g_last_frame[sid];

        for (int gy = 0; gy < HEAT_GRID_H; ++gy) {
            for (int gx = 0; gx < HEAT_GRID_W; ++gx) {
                float value = smoothed_value(sid, gx, gy);
                if (value < g_low) continue;
                uint32_t touched = g_touch[sid][gy][gx];
                uint64_t age = current_frame >= touched ? current_frame - touched : 0;
                float recency = 1.0f - clampf_heat((float)age / 1200.0f, 0.0f, 1.0f);
                HeatCandidate c;
                c.score = value + recency * 0.0008f;
                c.value = value;
                c.gx = gx;
                c.gy = gy;
                push_candidate(candidates, &count, g_max_points_per_source, c);
            }
        }

        unsigned int col = sid % columns;
        unsigned int row = sid / columns;
        float origin_x = (float)col * tile_w;
        float origin_y = (float)row * tile_h;
        float cell_w = tile_w / (float)HEAT_GRID_W;
        float cell_h = tile_h / (float)HEAT_GRID_H;
        float base_radius = (cell_w < cell_h ? cell_w : cell_h) * 0.88f;
        if (base_radius < 4.0f) base_radius = 4.0f;

        for (unsigned int i = 0; i < count; ++i) {
            if (!display || display->num_circles >= MAX_ELEMENTS_IN_DISPLAY_META) {
                display = new_display_meta(batch_meta, anchor);
                if (!display) return rendered;
            }
            HeatCandidate *c = &candidates[i];
            NvOSD_CircleParams *circle = &display->circle_params[display->num_circles++];
            float strength = clampf_heat(c->value / g_red, 0.0f, 1.0f);
            circle->xc = (unsigned int)(origin_x + ((float)c->gx + 0.5f) * cell_w);
            circle->yc = (unsigned int)(origin_y + ((float)c->gy + 0.5f) * cell_h);
            circle->radius = (unsigned int)(base_radius * (1.00f + 0.14f * strength));
            if (circle->radius < 4) circle->radius = 4;
            circle->circle_width = 1;
            circle->has_bg_color = 1;
            heat_color(c->value, &circle->bg_color);
            circle->circle_color = circle->bg_color;
            circle->circle_color.alpha *= 0.24f;
            ++rendered;
        }
    }

    g_rendered_points_total += (uint64_t)rendered;
    return rendered;
}

uint64_t camera_v2_heatmap_rendered_points_total(void) {
    return g_rendered_points_total;
}
