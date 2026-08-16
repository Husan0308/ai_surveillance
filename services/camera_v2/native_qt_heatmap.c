#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#define HM_MAX_SOURCES 16
#define HM_GRID_W 32
#define HM_GRID_H 18
#define HM_MAX_TRACKS 512
#define HM_MAX_POINTS 24

typedef struct {
    int valid;
    unsigned int source_id;
    uint64_t object_id;
    uint64_t last_frame;
    uint64_t last_vote_frame;
    uint64_t moving_until;
    unsigned int motion_votes;
    float anchor_x, anchor_y;
    float deposit_x, deposit_y;
} TrackState;

typedef struct {
    float score;
    float value;
    int x, y;
} Candidate;

static float g_heat[HM_MAX_SOURCES][HM_GRID_H][HM_GRID_W];
static uint32_t g_touch[HM_MAX_SOURCES][HM_GRID_H][HM_GRID_W];
static uint64_t g_last_frame[HM_MAX_SOURCES];
static TrackState g_tracks[HM_MAX_TRACKS];
static unsigned int g_counts[HM_MAX_SOURCES];
static uint64_t g_updates_total = 0;
static uint64_t g_points_total = 0;
static int g_points_last = 0;

static float g_deposit = 0.0025f;
static float g_decay = 0.99992f;
static float g_low = 0.0030f;
static float g_yellow = 0.070f;
static float g_red = 0.180f;
static unsigned int g_max_points = 18;

static const float MOTION_CONFIRM_D2 = 0.0576f; /* 0.24 cell */
static const float DEPOSIT_D2 = 0.0144f;        /* 0.12 cell */
static const uint64_t MOTION_WINDOW = 14;
static const uint64_t MOVING_HOLD = 10;

static float clampf_hm(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static float sqr_hm(float v) { return v * v; }

void camera_v2_qt_heatmap_configure(float deposit,
                                    float decay,
                                    float low,
                                    float yellow,
                                    float red,
                                    unsigned int max_points) {
    g_deposit = clampf_hm(deposit, 0.0003f, 0.03f);
    g_decay = clampf_hm(decay, 0.95f, 1.0f);
    g_low = clampf_hm(low, 0.0005f, 0.95f);
    g_yellow = clampf_hm(yellow, g_low + 0.005f, 0.98f);
    g_red = clampf_hm(red, g_yellow + 0.005f, 1.0f);
    if (max_points < 4) max_points = 4;
    if (max_points > HM_MAX_POINTS) max_points = HM_MAX_POINTS;
    g_max_points = max_points;
}

void camera_v2_qt_heatmap_reset(void) {
    memset(g_heat, 0, sizeof(g_heat));
    memset(g_touch, 0, sizeof(g_touch));
    memset(g_last_frame, 0, sizeof(g_last_frame));
    memset(g_tracks, 0, sizeof(g_tracks));
    memset(g_counts, 0, sizeof(g_counts));
    g_updates_total = 0;
    g_points_total = 0;
    g_points_last = 0;
}

static int find_track(unsigned int sid, uint64_t oid) {
    int free_i = -1, oldest_i = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < HM_MAX_TRACKS; ++i) {
        TrackState *s = &g_tracks[i];
        if (s->valid && s->source_id == sid && s->object_id == oid) return i;
        if (!s->valid && free_i < 0) free_i = i;
        if (s->valid && s->last_frame < oldest) {
            oldest = s->last_frame;
            oldest_i = i;
        }
    }
    return free_i >= 0 ? free_i : oldest_i;
}

static void reset_track(TrackState *s, unsigned int sid, uint64_t oid,
                        uint64_t frame, float gx, float gy) {
    memset(s, 0, sizeof(*s));
    s->valid = 1;
    s->source_id = sid;
    s->object_id = oid;
    s->last_frame = frame;
    s->last_vote_frame = frame;
    s->anchor_x = s->deposit_x = gx;
    s->anchor_y = s->deposit_y = gy;
}

static void decay_source(unsigned int sid, uint64_t frame) {
    if (sid >= HM_MAX_SOURCES) return;
    uint64_t last = g_last_frame[sid];
    if (!last || frame <= last) {
        g_last_frame[sid] = frame;
        return;
    }
    uint64_t d = frame - last;
    if (d > 2000) d = 2000;
    float factor = 1.0f;
    for (uint64_t i = 0; i < d; ++i) factor *= g_decay;
    for (int y = 0; y < HM_GRID_H; ++y) {
        for (int x = 0; x < HM_GRID_W; ++x) {
            float v = g_heat[sid][y][x] * factor;
            g_heat[sid][y][x] = v < 0.00035f ? 0.0f : v;
        }
    }
    g_last_frame[sid] = frame;
}

static void deposit_point(unsigned int sid, float gx, float gy,
                          uint32_t frame, float scale) {
    static const float kernel[3][3] = {
        {0.08f, 0.18f, 0.08f},
        {0.18f, 1.00f, 0.18f},
        {0.08f, 0.18f, 0.08f},
    };
    scale = clampf_hm(scale, 0.45f, 1.35f);
    int cx = (int)(gx + 0.5f), cy = (int)(gy + 0.5f);
    for (int ky = -1; ky <= 1; ++ky) {
        int y = cy + ky;
        if (y < 0 || y >= HM_GRID_H) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int x = cx + kx;
            if (x < 0 || x >= HM_GRID_W) continue;
            float next = g_heat[sid][y][x] + g_deposit * scale * kernel[ky + 1][kx + 1];
            g_heat[sid][y][x] = next > 1.0f ? 1.0f : next;
            g_touch[sid][y][x] = frame;
        }
    }
}

static void deposit_segment(unsigned int sid, float x0, float y0,
                            float x1, float y1, uint32_t frame) {
    float dx = x1 - x0, dy = y1 - y0;
    float ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy;
    float span = ax > ay ? ax : ay;
    int steps = (int)(span * 1.6f) + 1;
    if (steps < 1) steps = 1;
    if (steps > 16) steps = 16;
    float scale = clampf_hm(span, 0.50f, 1.20f);
    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / (float)steps;
        deposit_point(sid, x0 + dx * t, y0 + dy * t, frame, scale);
    }
}

static void update_track(unsigned int sid, uint64_t oid, uint64_t frame,
                         float gx, float gy) {
    int idx = find_track(sid, oid);
    TrackState *s = &g_tracks[idx];
    if (!s->valid || s->source_id != sid || s->object_id != oid ||
        frame < s->last_frame || frame - s->last_frame > 14) {
        reset_track(s, sid, oid, frame, gx, gy);
        return;
    }

    float anchor_d2 = sqr_hm(gx - s->anchor_x) + sqr_hm(gy - s->anchor_y);
    if (anchor_d2 >= MOTION_CONFIRM_D2) {
        s->motion_votes = (frame - s->last_vote_frame <= MOTION_WINDOW) ? s->motion_votes + 1 : 1;
        s->last_vote_frame = frame;
        s->anchor_x = gx;
        s->anchor_y = gy;
        if (s->motion_votes >= 2) {
            s->moving_until = frame + MOVING_HOLD;
            s->motion_votes = 0;
        }
    }

    if (frame <= s->moving_until) {
        float d2 = sqr_hm(gx - s->deposit_x) + sqr_hm(gy - s->deposit_y);
        if (d2 >= DEPOSIT_D2) {
            deposit_segment(sid, s->deposit_x, s->deposit_y, gx, gy, (uint32_t)frame);
            s->deposit_x = gx;
            s->deposit_y = gy;
            s->moving_until = frame + MOVING_HOLD;
            ++g_updates_total;
        }
    } else if (anchor_d2 < MOTION_CONFIRM_D2) {
        /* Follow stationary bbox jitter without depositing heat. */
        s->deposit_x = gx;
        s->deposit_y = gy;
    }
    s->last_frame = frame;
}

static void push_candidate(Candidate *items, unsigned int *count,
                           unsigned int max_count, Candidate c) {
    if (*count < max_count) {
        items[(*count)++] = c;
        return;
    }
    unsigned int min_i = 0;
    for (unsigned int i = 1; i < max_count; ++i)
        if (items[i].score < items[min_i].score) min_i = i;
    if (c.score > items[min_i].score) items[min_i] = c;
}

static void heat_color(float value, NvOSD_ColorParams *color) {
    float r = 0.05f, g = 0.58f, b = 1.0f;
    if (value < g_yellow) {
        float u = clampf_hm((value - g_low) / (g_yellow - g_low + 0.0001f), 0, 1);
        r = 0.05f + 0.90f * u;
        g = 0.58f + 0.37f * u;
        b = 1.00f - 0.92f * u;
    } else if (value < g_red) {
        float u = clampf_hm((value - g_yellow) / (g_red - g_yellow + 0.0001f), 0, 1);
        r = 1.0f; g = 0.95f * (1.0f - u); b = 0.02f;
    } else {
        float u = clampf_hm((value - g_red) / (1.0f - g_red + 0.0001f), 0, 1);
        r = 1.0f; g = 0.10f * (1.0f - u); b = 0.0f;
    }
    float strength = clampf_hm(value / g_red, 0, 1);
    color->red = r; color->green = g; color->blue = b;
    color->alpha = 0.045f + 0.105f * strength;
}

static NvDsDisplayMeta *new_display(NvDsBatchMeta *batch, NvDsFrameMeta *frame) {
    NvDsDisplayMeta *d = nvds_acquire_display_meta_from_pool(batch);
    if (!d) return NULL;
    d->num_rects = d->num_labels = d->num_lines = d->num_arrows = d->num_circles = 0;
    nvds_add_display_meta_to_frame(frame, d);
    return d;
}

static int render_source(NvDsBatchMeta *batch, NvDsFrameMeta *frame,
                         unsigned int sid, float frame_w, float frame_h) {
    Candidate items[HM_MAX_POINTS];
    unsigned int count = 0;
    uint64_t current = g_last_frame[sid];
    for (int y = 0; y < HM_GRID_H; ++y) {
        for (int x = 0; x < HM_GRID_W; ++x) {
            float value = g_heat[sid][y][x];
            if (value < g_low) continue;
            uint32_t touched = g_touch[sid][y][x];
            uint64_t age = current >= touched ? current - touched : 0;
            float recency = 1.0f - clampf_hm((float)age / 180.0f, 0, 1);
            Candidate c = {value + recency * 0.035f, value, x, y};
            push_candidate(items, &count, g_max_points, c);
        }
    }

    float cell_w = frame_w / HM_GRID_W;
    float cell_h = frame_h / HM_GRID_H;
    float base_radius = (cell_w < cell_h ? cell_w : cell_h) * 0.58f;
    if (base_radius < 4.0f) base_radius = 4.0f;
    NvDsDisplayMeta *display = NULL;
    int rendered = 0;
    for (unsigned int i = 0; i < count; ++i) {
        if (!display || display->num_circles >= MAX_ELEMENTS_IN_DISPLAY_META) {
            display = new_display(batch, frame);
            if (!display) return rendered;
        }
        Candidate *c = &items[i];
        NvOSD_CircleParams *circle = &display->circle_params[display->num_circles++];
        float strength = clampf_hm(c->value / g_red, 0, 1);
        circle->xc = (unsigned int)(((float)c->x + 0.5f) * cell_w);
        circle->yc = (unsigned int)(((float)c->y + 0.5f) * cell_h);
        circle->radius = (unsigned int)(base_radius * (1.0f + 0.10f * strength));
        if (circle->radius < 3) circle->radius = 3;
        circle->circle_width = 1;
        circle->has_bg_color = 1;
        heat_color(c->value, &circle->bg_color);
        circle->circle_color = circle->bg_color;
        circle->circle_color.alpha *= 0.45f;
        ++rendered;
    }
    return rendered;
}

/* Called on NvDCF output BEFORE nvstreamdemux. Display metadata is attached to
 * each NvDsFrameMeta, so after demux each Qt camera branch receives only its own
 * bbox + movement heat overlay. Accumulation always runs; visible controls render. */
int camera_v2_qt_heatmap_process(uintptr_t buffer_ptr, int visible) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) return -1;
    int rendered = 0;

    for (NvDsMetaList *fn = batch->frame_meta_list; fn; fn = fn->next) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fn->data;
        if (!fm || fm->source_id >= HM_MAX_SOURCES) continue;
        unsigned int sid = fm->source_id;
        uint64_t frame = fm->frame_num >= 0 ? (uint64_t)fm->frame_num : 0;
        decay_source(sid, frame);
        g_counts[sid] = 0;

        float fw = fm->pipeline_width > 1 ? (float)fm->pipeline_width : (float)fm->source_frame_width;
        float fh = fm->pipeline_height > 1 ? (float)fm->pipeline_height : (float)fm->source_frame_height;
        if (fw <= 1 || fh <= 1) continue;

        for (NvDsMetaList *on = fm->obj_meta_list; on; on = on->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)on->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            float w = obj->rect_params.width, h = obj->rect_params.height;
            if (w <= 2 || h <= 4) continue;
            ++g_counts[sid];
            float foot_x = obj->rect_params.left + w * 0.5f;
            float foot_y = obj->rect_params.top + h * 0.98f;
            float gx = clampf_hm((foot_x / fw) * (HM_GRID_W - 1), 0, HM_GRID_W - 1);
            float gy = clampf_hm((foot_y / fh) * (HM_GRID_H - 1), 0, HM_GRID_H - 1);
            update_track(sid, (uint64_t)obj->object_id, frame, gx, gy);
        }

        if (visible) rendered += render_source(batch, fm, sid, fw, fh);

        for (int i = 0; i < HM_MAX_TRACKS; ++i) {
            TrackState *s = &g_tracks[i];
            if (s->valid && s->source_id == sid && frame > s->last_frame && frame - s->last_frame > 80)
                s->valid = 0;
        }
    }
    g_points_last = rendered;
    g_points_total += (uint64_t)rendered;
    return rendered;
}

unsigned int camera_v2_qt_heatmap_current_count(unsigned int sid) {
    return sid < HM_MAX_SOURCES ? g_counts[sid] : 0;
}
uint64_t camera_v2_qt_heatmap_updates_total(void) { return g_updates_total; }
uint64_t camera_v2_qt_heatmap_points_total(void) { return g_points_total; }
int camera_v2_qt_heatmap_points_last(void) { return g_points_last; }
