#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define POSE_HEAT_MAX_SOURCES 16
#define POSE_HEAT_GRID_W 48
#define POSE_HEAT_GRID_H 27
#define POSE_HEAT_MAX_TRACKS 512
#define POSE_HEAT_MAX_POINTS 96

typedef struct {
    int valid;
    int started;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_tick;
    uint64_t last_presence_tick;
    uint64_t dwell_ticks;
    unsigned int stable;
    float smooth_gx;
    float smooth_gy;
    float deposit_gx;
    float deposit_gy;
} PoseHeatTrack;

typedef struct {
    float score;
    float value;
    int gx;
    int gy;
} PoseHeatCandidate;

static float g_pose_heat[POSE_HEAT_MAX_SOURCES][POSE_HEAT_GRID_H][POSE_HEAT_GRID_W];
static uint32_t g_pose_touch[POSE_HEAT_MAX_SOURCES][POSE_HEAT_GRID_H][POSE_HEAT_GRID_W];
static uint64_t g_pose_last_tick[POSE_HEAT_MAX_SOURCES];
static PoseHeatTrack g_pose_tracks[POSE_HEAT_MAX_TRACKS];
static uint64_t g_pose_rendered_total = 0;

static float g_pose_deposit = 0.0028f;
static float g_pose_decay = 0.999968f;
static float g_pose_low = 0.00028f;
static float g_pose_yellow = 0.100f;
static float g_pose_red = 0.300f;
static unsigned int g_pose_max_points = 84;

static float clampf_pose(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float sqr_pose(float v) { return v * v; }

void camera_v2_pose_heatmap_configure(float deposit,
                                      float decay,
                                      float low_threshold,
                                      float yellow_threshold,
                                      float red_threshold,
                                      unsigned int max_points_per_source) {
    g_pose_deposit = clampf_pose(deposit, 0.0003f, 0.03f);
    g_pose_decay = clampf_pose(decay, 0.95f, 0.9999999f);
    g_pose_low = clampf_pose(low_threshold, 0.00020f, 0.50f);
    g_pose_yellow = clampf_pose(yellow_threshold, g_pose_low + 0.001f, 0.95f);
    g_pose_red = clampf_pose(red_threshold, g_pose_yellow + 0.002f, 1.0f);
    if (max_points_per_source < 12) max_points_per_source = 12;
    if (max_points_per_source > POSE_HEAT_MAX_POINTS) max_points_per_source = POSE_HEAT_MAX_POINTS;
    g_pose_max_points = max_points_per_source;
}

void camera_v2_pose_heatmap_reset(void) {
    memset(g_pose_heat, 0, sizeof(g_pose_heat));
    memset(g_pose_touch, 0, sizeof(g_pose_touch));
    memset(g_pose_last_tick, 0, sizeof(g_pose_last_tick));
    memset(g_pose_tracks, 0, sizeof(g_pose_tracks));
    g_pose_rendered_total = 0;
}

static int find_pose_track(unsigned int source_id, uint64_t object_id) {
    int free_index = -1;
    int oldest_index = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < POSE_HEAT_MAX_TRACKS; ++i) {
        PoseHeatTrack *s = &g_pose_tracks[i];
        if (s->valid && s->source_id == source_id && s->object_id == object_id) return i;
        if (!s->valid && free_index < 0) free_index = i;
        if (s->valid && s->last_tick < oldest) {
            oldest = s->last_tick;
            oldest_index = i;
        }
    }
    return free_index >= 0 ? free_index : oldest_index;
}

static void decay_pose_source(unsigned int sid, uint64_t tick) {
    if (sid >= POSE_HEAT_MAX_SOURCES) return;
    uint64_t last = g_pose_last_tick[sid];
    if (last == 0 || tick <= last) {
        g_pose_last_tick[sid] = tick;
        return;
    }
    uint64_t delta = tick - last;
    if (delta > 4000) delta = 4000;
    float factor = 1.0f;
    for (uint64_t i = 0; i < delta; ++i) factor *= g_pose_decay;
    for (int y = 0; y < POSE_HEAT_GRID_H; ++y) {
        for (int x = 0; x < POSE_HEAT_GRID_W; ++x) {
            float v = g_pose_heat[sid][y][x] * factor;
            g_pose_heat[sid][y][x] = v < g_pose_low * 0.36f ? 0.0f : v;
        }
    }
    g_pose_last_tick[sid] = tick;
}

static void deposit_pose_point(unsigned int sid,
                               float gx,
                               float gy,
                               uint32_t tick,
                               float scale) {
    scale = clampf_pose(scale, 0.05f, 1.15f);
    int cx = (int)(gx + 0.5f);
    int cy = (int)(gy + 0.5f);
    float depth = clampf_pose(gy / (float)(POSE_HEAT_GRID_H - 1), 0.0f, 1.0f);
    int radius = depth > 0.62f ? 3 : 2;
    float y_scale = 1.20f - 0.22f * depth;
    for (int ky = -radius; ky <= radius; ++ky) {
        int y = cy + ky;
        if (y < 0 || y >= POSE_HEAT_GRID_H) continue;
        for (int kx = -radius; kx <= radius; ++kx) {
            int x = cx + kx;
            if (x < 0 || x >= POSE_HEAT_GRID_W) continue;
            float d2 = (float)(kx * kx) + (float)(ky * ky) * y_scale;
            float r2 = (float)(radius * radius);
            if (d2 > r2 * 1.12f) continue;
            float kernel = 1.0f / (1.0f + 0.78f * d2);
            float next = g_pose_heat[sid][y][x] + g_pose_deposit * scale * kernel;
            g_pose_heat[sid][y][x] = next > 1.0f ? 1.0f : next;
            g_pose_touch[sid][y][x] = tick;
        }
    }
}

static void deposit_pose_segment(unsigned int sid,
                                 float x0,
                                 float y0,
                                 float x1,
                                 float y1,
                                 uint32_t tick,
                                 float confidence) {
    float dx = x1 - x0;
    float dy = y1 - y0;
    float adx = dx < 0.0f ? -dx : dx;
    float ady = dy < 0.0f ? -dy : dy;
    float span = adx > ady ? adx : ady;
    int steps = (int)(span * 4.0f) + 2;
    if (steps < 2) steps = 2;
    if (steps > 24) steps = 24;
    float scale = clampf_pose((0.36f + span * 0.10f) * (0.70f + 0.30f * confidence), 0.28f, 0.90f);
    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / (float)steps;
        deposit_pose_point(sid, x0 + dx * t, y0 + dy * t, tick, scale);
    }
}

/*
 * Authoritative camera-space heat input: a tracked person's ankle keypoint.
 * nx/ny are normalized to the original camera frame. No bounding-box coordinate
 * is accepted by this API, so production heat cannot silently fall back to bbox
 * bottom-center when ankles are missing.
 */
int camera_v2_pose_heatmap_update_anchor(unsigned int source_id,
                                         uint64_t object_id,
                                         uint64_t tick,
                                         float nx,
                                         float ny,
                                         float confidence) {
    if (source_id >= POSE_HEAT_MAX_SOURCES || object_id == UINT64_MAX) return 0;
    if (tick == 0 || confidence < 0.05f) return 0;
    nx = clampf_pose(nx, 0.0f, 1.0f);
    ny = clampf_pose(ny, 0.0f, 1.0f);
    confidence = clampf_pose(confidence, 0.0f, 1.0f);
    decay_pose_source(source_id, tick);

    float raw_gx = nx * (float)(POSE_HEAT_GRID_W - 1);
    float raw_gy = ny * (float)(POSE_HEAT_GRID_H - 1);
    int idx = find_pose_track(source_id, object_id);
    PoseHeatTrack *s = &g_pose_tracks[idx];
    if (!s->valid || s->source_id != source_id || s->object_id != object_id ||
        tick < s->last_tick || tick - s->last_tick > 90) {
        memset(s, 0, sizeof(*s));
        s->valid = 1;
        s->source_id = source_id;
        s->object_id = object_id;
        s->last_tick = tick;
        s->last_presence_tick = tick;
        s->stable = 1;
        s->smooth_gx = raw_gx;
        s->smooth_gy = raw_gy;
        s->deposit_gx = raw_gx;
        s->deposit_gy = raw_gy;
        return 0;
    }

    uint64_t delta = tick > s->last_tick ? tick - s->last_tick : 1;
    if (delta > 60) delta = 60;
    if (s->stable < 1000000U) s->stable += 1;
    float alpha = 0.62f;
    float gx = alpha * raw_gx + (1.0f - alpha) * s->smooth_gx;
    float gy = alpha * raw_gy + (1.0f - alpha) * s->smooth_gy;
    s->smooth_gx = gx;
    s->smooth_gy = gy;

    if (s->stable < 2) {
        s->last_tick = tick;
        return 0;
    }

    int updates = 0;
    float moved2 = sqr_pose(gx - s->deposit_gx) + sqr_pose(gy - s->deposit_gy);
    if (!s->started) {
        deposit_pose_point(source_id, gx, gy, (uint32_t)tick, 0.22f * (0.70f + 0.30f * confidence));
        s->started = 1;
        s->last_presence_tick = tick;
        s->deposit_gx = gx;
        s->deposit_gy = gy;
        updates += 1;
    } else if (moved2 >= 0.10f) {
        deposit_pose_segment(source_id, s->deposit_gx, s->deposit_gy, gx, gy, (uint32_t)tick, confidence);
        s->deposit_gx = gx;
        s->deposit_gy = gy;
        s->dwell_ticks = 0;
        s->last_presence_tick = tick;
        updates += 1;
    } else {
        s->dwell_ticks += delta;
        if (tick >= s->last_presence_tick + 20) {
            float dwell = clampf_pose((float)s->dwell_ticks / 400.0f, 0.0f, 1.0f);
            float scale = (0.08f + 0.20f * dwell) * (0.70f + 0.30f * confidence);
            deposit_pose_point(source_id, gx, gy, (uint32_t)tick, scale);
            s->last_presence_tick = tick;
            updates += 1;
        }
    }
    s->last_tick = tick;
    return updates;
}

static float smoothed_pose_value(unsigned int sid, int gx, int gy) {
    static const float w[3][3] = {
        {0.05f, 0.10f, 0.05f},
        {0.10f, 0.40f, 0.10f},
        {0.05f, 0.10f, 0.05f},
    };
    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        int y = gy + ky;
        if (y < 0 || y >= POSE_HEAT_GRID_H) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int x = gx + kx;
            if (x < 0 || x >= POSE_HEAT_GRID_W) continue;
            sum += g_pose_heat[sid][y][x] * w[ky + 1][kx + 1];
        }
    }
    return sum;
}

static void push_pose_candidate(PoseHeatCandidate *items,
                                unsigned int *count,
                                unsigned int max_count,
                                PoseHeatCandidate candidate) {
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

static void pose_heat_color(float value, NvOSD_ColorParams *color) {
    float r = 0.00f, g = 0.42f, b = 1.00f;
    if (value < g_pose_yellow) {
        float u = (value - g_pose_low) / (g_pose_yellow - g_pose_low + 0.000001f);
        u = clampf_pose(u, 0.0f, 1.0f);
        if (u < 0.50f) {
            float v = u * 2.0f;
            r = 0.00f;
            g = 0.42f + 0.46f * v;
            b = 1.00f;
        } else {
            float v = (u - 0.50f) * 2.0f;
            r = 0.18f * v;
            g = 0.88f + 0.06f * v;
            b = 1.00f - 0.78f * v;
        }
    } else if (value < g_pose_red) {
        float u = (value - g_pose_yellow) / (g_pose_red - g_pose_yellow + 0.000001f);
        u = clampf_pose(u, 0.0f, 1.0f);
        r = 0.18f + 0.82f * u;
        g = 0.94f - 0.24f * u;
        b = 0.22f * (1.0f - u);
    } else {
        float u = (value - g_pose_red) / (1.0f - g_pose_red + 0.000001f);
        u = clampf_pose(u, 0.0f, 1.0f);
        r = 1.00f;
        g = 0.70f * (1.0f - u);
        b = 0.00f;
    }
    float strength = clampf_pose(value / g_pose_red, 0.0f, 1.0f);
    color->red = r;
    color->green = g;
    color->blue = b;
    color->alpha = 0.030f + 0.145f * strength;
}

static NvDsDisplayMeta *new_pose_display_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *anchor) {
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

/* focus_source >= 0 renders that camera's heat over the whole focused frame. */
int camera_v2_pose_heatmap_render(uintptr_t buffer_ptr,
                                  unsigned int wall_width,
                                  unsigned int wall_height,
                                  unsigned int rows,
                                  unsigned int columns,
                                  unsigned int source_count,
                                  int focus_source) {
    if (!buffer_ptr || wall_width < 2 || wall_height < 2 || rows == 0 || columns == 0) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta || !batch_meta->frame_meta_list) return -1;
    NvDsFrameMeta *anchor = (NvDsFrameMeta *)batch_meta->frame_meta_list->data;
    if (!anchor) return -1;

    unsigned int max_sources = rows * columns;
    if (source_count < max_sources) max_sources = source_count;
    if (max_sources > POSE_HEAT_MAX_SOURCES) max_sources = POSE_HEAT_MAX_SOURCES;
    if (focus_source >= (int)max_sources) focus_source = -1;

    float tile_w = focus_source >= 0 ? (float)wall_width : (float)wall_width / (float)columns;
    float tile_h = focus_source >= 0 ? (float)wall_height : (float)wall_height / (float)rows;
    NvDsDisplayMeta *display = NULL;
    int rendered = 0;

    unsigned int sid_start = focus_source >= 0 ? (unsigned int)focus_source : 0U;
    unsigned int sid_end = focus_source >= 0 ? sid_start + 1U : max_sources;
    for (unsigned int sid = sid_start; sid < sid_end; ++sid) {
        PoseHeatCandidate candidates[POSE_HEAT_MAX_POINTS];
        unsigned int count = 0;
        uint64_t current_tick = g_pose_last_tick[sid];
        for (int gy = 0; gy < POSE_HEAT_GRID_H; ++gy) {
            for (int gx = 0; gx < POSE_HEAT_GRID_W; ++gx) {
                float value = smoothed_pose_value(sid, gx, gy);
                if (value < g_pose_low) continue;
                uint32_t touched = g_pose_touch[sid][gy][gx];
                uint64_t age = current_tick >= touched ? current_tick - touched : 0;
                float recency = 1.0f - clampf_pose((float)age / 1200.0f, 0.0f, 1.0f);
                PoseHeatCandidate c;
                c.score = value + recency * g_pose_low * 0.30f;
                c.value = value;
                c.gx = gx;
                c.gy = gy;
                push_pose_candidate(candidates, &count, g_pose_max_points, c);
            }
        }

        unsigned int col = focus_source >= 0 ? 0U : sid % columns;
        unsigned int row = focus_source >= 0 ? 0U : sid / columns;
        float origin_x = (float)col * tile_w;
        float origin_y = (float)row * tile_h;
        float cell_w = tile_w / (float)POSE_HEAT_GRID_W;
        float cell_h = tile_h / (float)POSE_HEAT_GRID_H;
        float base_radius = (cell_w < cell_h ? cell_w : cell_h) * 1.08f;
        if (base_radius < 4.0f) base_radius = 4.0f;

        for (unsigned int i = 0; i < count; ++i) {
            if (!display || display->num_circles >= MAX_ELEMENTS_IN_DISPLAY_META) {
                display = new_pose_display_meta(batch_meta, anchor);
                if (!display) return rendered;
            }
            PoseHeatCandidate *c = &candidates[i];
            NvOSD_CircleParams *circle = &display->circle_params[display->num_circles++];
            float strength = clampf_pose(c->value / g_pose_red, 0.0f, 1.0f);
            float depth = clampf_pose((float)c->gy / (float)(POSE_HEAT_GRID_H - 1), 0.0f, 1.0f);
            float perspective = 0.72f + 0.42f * depth;
            circle->xc = (unsigned int)(origin_x + ((float)c->gx + 0.5f) * cell_w);
            circle->yc = (unsigned int)(origin_y + ((float)c->gy + 0.5f) * cell_h);
            circle->radius = (unsigned int)(base_radius * perspective * (1.00f + 0.16f * strength));
            if (circle->radius < 4) circle->radius = 4;
            circle->circle_width = 1;
            circle->has_bg_color = 1;
            pose_heat_color(c->value, &circle->bg_color);
            circle->circle_color = circle->bg_color;
            circle->circle_color.alpha *= 0.18f;
            ++rendered;
        }
    }
    g_pose_rendered_total += (uint64_t)rendered;
    return rendered;
}

uint64_t camera_v2_pose_heatmap_rendered_points_total(void) {
    return g_pose_rendered_total;
}
