/* Experiment-only causal PeopleNet 2.6.3 bottom-edge correction probe. */
#include "bbox_correction.h"

#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvds_tracker_meta.h"
#include "bbox_history_lifecycle.h"

#define MAX_SOURCES 32
#define MAX_HISTORIES 128
#define MAX_DETECTIONS 128
#define MAX_CANDIDATES (MAX_HISTORIES * MAX_DETECTIONS)
#define RELIABLE_LEN 8
#define CONF_LEN 12
#define HISTORY_MAX_GAP_FRAMES 2

/* Frozen from the validated full-dataset offline fit. */
#define TH_BOTTOM_UP 23.058527
#define TH_HEIGHT_DROP 0.10
#define TH_WIDTH_EXPAND 0.2062593746897553
#define TH_ASPECT_LOG 0.23007604131814643
#define TH_CONF_DROP 0.22973609999999994

/* Source-local, causal recovery hysteresis for weak post-episode echoes. */
#define RECOVERY_HYSTERESIS_FRAMES 60
#define RECOVERY_STRONG_REENTRY_DELTA 30.0

typedef enum { MODE_OFF = 0, MODE_SHADOW = 1, MODE_ACTIVE = 2 } Mode;

typedef struct {
  double l, t, r, b, conf;
  guint64 frame;
  NvDsObjectMeta *meta;
  int index;
} Box;

typedef struct {
  guint64 generation;
  guint64 first_frame;
  guint64 detector_hits;
  Box last;
  Box reliable[RELIABLE_LEN];
  int reliable_count;
  double confs[CONF_LEN];
  int conf_count;
  double conf_med;
  int precursor;
  int watch;
  gboolean frozen;
  gboolean anomaly;
} History;

typedef struct {
  History histories[MAX_HISTORIES];
  Pn263HistorySlot slots[MAX_HISTORIES];
  Pn263HistoryPool pool;
  gboolean anomaly_active_last_frame;
  guint64 recovery_until_frame;
} SourceState;

typedef struct {
  double cost;
  int history;
  int detection;
} Candidate;

typedef struct {
  gboolean bottom_up;
  gboolean height_drop;
  gboolean width_expand;
  gboolean aspect_log;
  gboolean conf_drop;
  gboolean core;
  gboolean precursor;
  int structural;
} Signals;

static SourceState states[MAX_SOURCES];
static Mode mode = MODE_OFF;
static FILE *decision_log;
static FILE *pre_log;
static FILE *post_log;
static FILE *tracker_state_log;
static FILE *recovery_log;
static FILE *history_log;
static guint64 frames_seen;
static guint64 detections_seen;
static guint64 decisions_seen;
static guint64 decisions_by_source[MAX_SOURCES];
static gint64 probe_time_us;
static gboolean initialized;

static inline double
box_w (const Box *b) { return b->r - b->l; }

static inline double
box_h (const Box *b) { return b->b - b->t; }

static inline double
box_cx (const Box *b) { return (b->l + b->r) * 0.5; }

static inline double
box_cy (const Box *b) { return (b->t + b->b) * 0.5; }

static int
double_cmp (const void *a, const void *b)
{
  const double x = *(const double *) a;
  const double y = *(const double *) b;
  return (x > y) - (x < y);
}

static double
median (const double *values, int n)
{
  double tmp[RELIABLE_LEN];
  int i;
  g_assert (n > 0 && n <= RELIABLE_LEN);
  for (i = 0; i < n; i++) tmp[i] = values[i];
  qsort (tmp, n, sizeof (double), double_cmp);
  return n & 1 ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

static double
median_conf (const double *values, int n)
{
  double tmp[CONF_LEN];
  int i;
  g_assert (n > 0 && n <= CONF_LEN);
  for (i = 0; i < n; i++) tmp[i] = values[i];
  qsort (tmp, n, sizeof (double), double_cmp);
  return n & 1 ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
}

static double
clampd (double x, double lo, double hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}

static double
iou (const Box *a, const Box *b)
{
  const double iw = fmax (0.0, fmin (a->r, b->r) - fmax (a->l, b->l));
  const double ih = fmax (0.0, fmin (a->b, b->b) - fmax (a->t, b->t));
  const double inter = iw * ih;
  return inter / fmax (box_w (a) * box_h (a) + box_w (b) * box_h (b) - inter, 1e-6);
}

static Box
predict_box (const History *h, guint64 current_frame, gboolean anchor)
{
  double cx[RELIABLE_LEN], cy[RELIABLE_LEN], w[RELIABLE_LEN];
  double hh[RELIABLE_LEN], bottom[RELIABLE_LEN];
  double dcx[RELIABLE_LEN], dcy[RELIABLE_LEN], dw[RELIABLE_LEN];
  double dh[RELIABLE_LEN], db[RELIABLE_LEN];
  const int want = anchor ? 8 : 3;
  const int n = MIN (h->reliable_count, want);
  const int start = h->reliable_count - n;
  int i, vn;
  Box p = h->last;

  for (i = 0; i < n; i++) {
    const Box *q = &h->reliable[start + i];
    cx[i] = box_cx (q); cy[i] = box_cy (q); w[i] = box_w (q);
    hh[i] = box_h (q); bottom[i] = q->b;
  }
  double pcx = median (cx, n), pcy = median (cy, n);
  double pw = median (w, n), ph = median (hh, n), pb = median (bottom, n);

  if (h->reliable_count >= 3) {
    const int vcount = MIN (h->reliable_count, 5);
    const int vstart = h->reliable_count - vcount;
    vn = 0;
    for (i = vstart + 1; i < h->reliable_count; i++) {
      const Box *a = &h->reliable[i - 1], *b = &h->reliable[i];
      dcx[vn] = box_cx (b) - box_cx (a);
      dcy[vn] = box_cy (b) - box_cy (a);
      dw[vn] = box_w (b) - box_w (a);
      dh[vn] = box_h (b) - box_h (a);
      db[vn] = b->b - a->b;
      vn++;
    }
    double steps = 1.0;
    if (anchor) {
      const guint64 last_reliable_frame = h->reliable[h->reliable_count - 1].frame;
      steps = MIN (4.0, MAX (1.0, (double) (current_frame - last_reliable_frame)));
    }
    pcx += steps * clampd (median (dcx, vn), -40, 40);
    pcy += steps * clampd (median (dcy, vn), -40, 40);
    pw += steps * clampd (median (dw, vn), -30, 30);
    ph += steps * clampd (median (dh, vn), -40, 40);
    pb += steps * clampd (median (db, vn), -40, 40);
  }
  pw = fmax (2.0, pw); ph = fmax (2.0, ph);
  p.l = pcx - pw * 0.5; p.r = pcx + pw * 0.5;
  p.b = pb; p.t = pb - ph;
  return p;
}

static double
association_cost (const Box *d, const Box *p)
{
  const double dx = (box_cx (d) - box_cx (p)) / fmax (box_w (p), 30.0);
  const double dy = (box_cy (d) - box_cy (p)) / fmax (box_h (p), 50.0);
  const double shape = fabs (log (fmax (box_w (d), 2.0) / fmax (box_w (p), 2.0))) +
      fabs (log (fmax (box_h (d), 2.0) / fmax (box_h (p), 2.0)));
  return hypot (dx, dy) + 0.25 * shape + 0.35 * (1.0 - iou (d, p));
}

static int
candidate_cmp (const void *a, const void *b)
{
  const Candidate *x = a, *y = b;
  if (x->cost != y->cost) return x->cost < y->cost ? -1 : 1;
  if (x->history != y->history) return x->history - y->history;
  return x->detection - y->detection;
}

static void
append_reliable (History *h, const Box *b)
{
  if (h->reliable_count == RELIABLE_LEN) {
    memmove (&h->reliable[0], &h->reliable[1],
        sizeof (Box) * (RELIABLE_LEN - 1));
    h->reliable_count--;
  }
  h->reliable[h->reliable_count++] = *b;
  if (h->conf_count == CONF_LEN) {
    memmove (&h->confs[0], &h->confs[1], sizeof (double) * (CONF_LEN - 1));
    h->conf_count--;
  }
  h->confs[h->conf_count++] = b->conf;
  h->conf_med = median_conf (h->confs, h->conf_count);
}

static gboolean
is_person (const NvDsObjectMeta *obj)
{
  return obj && (g_ascii_strcasecmp (obj->obj_label, "Person") == 0 ||
      g_ascii_strcasecmp (obj->obj_label, "person") == 0);
}

static Box
box_from_meta (NvDsObjectMeta *obj, guint64 frame, int index)
{
  const NvBbox_Coords *b = &obj->detector_bbox_info.org_bbox_coords;
  Box out = { b->left, b->top, b->left + b->width, b->top + b->height,
      obj->confidence, frame, obj, index };
  return out;
}

static Signals
signals_for (const Box *d, const Box *p, const History *h)
{
  Signals s = { 0 };
  const double bottom_up = p->b - d->b;
  const double height_drop = 1.0 - box_h (d) / fmax (box_h (p), 1.0);
  const double width_expand = box_w (d) / fmax (box_w (p), 1.0) - 1.0;
  const double aspect = fabs (log (fmax (box_w (d) / fmax (box_h (d), 1e-6), 1e-6) /
      fmax (box_w (p) / fmax (box_h (p), 1e-6), 1e-6)));
  const double conf_drop = h->conf_med - d->conf;
  const gboolean top_clipped = d->t <= 12.0 || p->t <= 12.0;
  int weak = 0;

  s.bottom_up = bottom_up > TH_BOTTOM_UP;
  s.height_drop = height_drop > TH_HEIGHT_DROP;
  s.width_expand = width_expand > TH_WIDTH_EXPAND;
  s.aspect_log = aspect > TH_ASPECT_LOG;
  s.conf_drop = conf_drop > TH_CONF_DROP;
  s.structural = s.width_expand + s.aspect_log + s.conf_drop;
  weak += bottom_up > 10.0;
  weak += height_drop > 0.04;
  weak += width_expand > 0.12 || width_expand < -0.18;
  weak += aspect > 0.12;
  weak += conf_drop > 0.12;
  s.precursor = top_clipped && weak >= 3;
  s.core = top_clipped && s.bottom_up && s.height_drop && s.structural >= 1;
  return s;
}

static const char *
source_name (guint source)
{
  if (source == 0) return "CAM-01";
  if (source == 1) return "CAM-04";
  static char text[32];
  g_snprintf (text, sizeof (text), "source-%u", source);
  return text;
}

static const char *
tracker_state_name (TRACKER_STATE state)
{
  switch (state) {
    case ACTIVE: return "ACTIVE";
    case INACTIVE: return "INACTIVE";
    case TENTATIVE: return "TENTATIVE";
    case PROJECTED: return "PROJECTED";
    case QUASIACTIVE: return "QUASIACTIVE";
    case EMPTY: return "EMPTY";
    default: return "UNKNOWN";
  }
}

static void
format_signals (const Signals *s, char *text, size_t size)
{
  text[0] = '\0';
#define ADD_SIGNAL(name, value) do { if (value) { if (text[0]) g_strlcat (text, "+", size); g_strlcat (text, name, size); } } while (0)
  ADD_SIGNAL ("bottom_up", s->bottom_up);
  ADD_SIGNAL ("height_drop", s->height_drop);
  ADD_SIGNAL ("width_expand", s->width_expand);
  ADD_SIGNAL ("aspect_log", s->aspect_log);
  ADD_SIGNAL ("conf_drop", s->conf_drop);
#undef ADD_SIGNAL
}

static int
new_history (SourceState *state, const Box *d, uint8_t *claimed)
{
  int id = pn263_history_pool_acquire (&state->pool, state->slots,
      MAX_HISTORIES, claimed, d->frame);
  History *h;
  if (id < 0) {
    g_printerr ("PN263_BBOX ERROR no unclaimed history slot frame=%" G_GUINT64_FORMAT
        " active=%zu capacity=%d\n", d->frame, state->pool.active_count,
        MAX_HISTORIES);
    return -1;
  }
  h = &state->histories[id];
  memset (h, 0, sizeof (*h));
  h->generation = state->slots[id].generation;
  h->first_frame = d->frame;
  h->detector_hits = 1;
  h->last = *d;
  append_reliable (h, d);
  claimed[id] = TRUE;
  return id;
}

static void
process_frame (NvDsFrameMeta *frame_meta)
{
  Box dets[MAX_DETECTIONS], predictions[MAX_HISTORIES];
  Candidate candidates[MAX_CANDIDATES];
  uint8_t history_used[MAX_HISTORIES] = { 0 };
  uint8_t was_used[MAX_HISTORIES];
  gboolean detection_used[MAX_DETECTIONS] = { 0 };
  NvDsMetaList *item;
  int nd = 0, nc = 0, i, j;
  gboolean anomaly_active_this_frame = FALSE;
  const guint source = frame_meta->source_id;
  const guint64 frame = frame_meta->frame_num;
  const double frame_w = frame_meta->source_frame_width ? frame_meta->source_frame_width : 1920.0;
  const double frame_h = frame_meta->source_frame_height ? frame_meta->source_frame_height : 1080.0;
  SourceState *state;
  uint64_t created_before, retired_before, evicted_before, reused_before;
  gboolean epoch_reset;

  if (source >= MAX_SOURCES) return;
  state = &states[source];
  for (i = 0; i < MAX_HISTORIES; i++)
    was_used[i] = state->slots[i].used;
  created_before = state->pool.created_count;
  retired_before = state->pool.retired_count;
  evicted_before = state->pool.evicted_count;
  reused_before = state->pool.reused_count;
  epoch_reset = pn263_history_pool_begin_frame (&state->pool, state->slots,
      MAX_HISTORIES, frame, HISTORY_MAX_GAP_FRAMES);
  for (i = 0; i < MAX_HISTORIES; i++) {
    if (was_used[i] && !state->slots[i].used)
      memset (&state->histories[i], 0, sizeof (state->histories[i]));
  }
  if (epoch_reset) {
    state->anomaly_active_last_frame = FALSE;
    state->recovery_until_frame = 0;
  }
  for (item = frame_meta->obj_meta_list; item && nd < MAX_DETECTIONS; item = item->next) {
    NvDsObjectMeta *obj = item->data;
    if (!is_person (obj)) continue;
    dets[nd] = box_from_meta (obj, frame, nd);
    nd++;
  }
  detections_seen += nd;
  frames_seen++;

  for (i = 0; i < MAX_HISTORIES; i++) {
    History *h = &state->histories[i];
    if (!state->slots[i].used || h->generation != state->slots[i].generation ||
        frame < state->slots[i].last_frame ||
        frame - state->slots[i].last_frame > HISTORY_MAX_GAP_FRAMES) continue;
    predictions[i] = predict_box (h, frame, FALSE);
    for (j = 0; j < nd; j++) {
      const double center = hypot (box_cx (&dets[j]) - box_cx (&predictions[i]),
          box_cy (&dets[j]) - box_cy (&predictions[i]));
      if (center <= fmax (90.0, 0.65 * box_h (&predictions[i])) &&
          (iou (&dets[j], &predictions[i]) >= 0.08 ||
           center / fmax (box_h (&predictions[i]), 50.0) < 0.42)) {
        if (nc < MAX_CANDIDATES) {
          candidates[nc++] = (Candidate) {
            association_cost (&dets[j], &predictions[i]), i, j
          };
        }
      }
    }
  }
  qsort (candidates, nc, sizeof (Candidate), candidate_cmp);

  for (i = 0; i < nc; i++) {
    Candidate *c = &candidates[i];
    History *h;
    Box *d, anchor;
    Signals s = { 0 };
    gboolean is_anomaly = FALSE;
    double alt = G_MAXDOUBLE;
    char signal_text[128];
    if (history_used[c->history] || detection_used[c->detection]) continue;
    h = &state->histories[c->history];
    if (!state->slots[c->history].used ||
        h->generation != state->slots[c->history].generation) continue;
    for (j = 0; j < nc; j++) {
      if (candidates[j].detection == c->detection &&
          candidates[j].history != c->history &&
          !history_used[candidates[j].history])
        alt = fmin (alt, candidates[j].cost);
    }
    if (alt < G_MAXDOUBLE && alt - c->cost < 0.12) continue;
    history_used[c->history] = TRUE;
    detection_used[c->detection] = TRUE;
    d = &dets[c->detection];
    h->detector_hits++;
    anchor = predict_box (h, frame, TRUE);

    if (h->reliable_count >= 4) {
      s = signals_for (d, &anchor, h);
      h->precursor = s.precursor ? h->precursor + 1 : 0;
      if (h->precursor >= 2) h->watch = 12;
      else if (h->watch > 0) h->watch--;
      if (h->watch > 0) h->frozen = TRUE;
      if (s.core) {
        const double candidate_delta = anchor.b - TH_BOTTOM_UP - d->b;
        const gboolean weak_reentry = frame <= state->recovery_until_frame &&
            candidate_delta < RECOVERY_STRONG_REENTRY_DELTA && !h->anomaly;
        if (weak_reentry) {
          if (recovery_log) {
            fprintf (recovery_log,
                "%s,%u,%" G_GUINT64_FORMAT ",%d,%.6f,%" G_GUINT64_FORMAT
                ",weak_reentry_after_episode\n",
                source_name (source), source, frame, c->history,
                candidate_delta, state->recovery_until_frame);
          }
        } else {
          h->anomaly = TRUE;
        }
      }
      const gboolean plausible = anchor.b - d->b <= 10.0 &&
          1.0 - box_h (d) / fmax (box_h (&anchor), 1.0) <= 0.04 &&
          s.structural < 2;
      if (h->anomaly && plausible) {
        h->anomaly = FALSE; h->frozen = FALSE; h->precursor = 0;
      } else if (!h->anomaly && h->watch == 0) {
        h->frozen = FALSE;
      }
      is_anomaly = h->anomaly;
    }

    double proposed_b = d->b;
    if (is_anomaly) {
      anomaly_active_this_frame = TRUE;
      proposed_b = fmax (d->t + 2.0, fmin (frame_h, anchor.b - TH_BOTTOM_UP));
      format_signals (&s, signal_text, sizeof (signal_text));
      decisions_seen++;
      decisions_by_source[source]++;
      if (decision_log) {
        fprintf (decision_log,
            "%s,%u,%" G_GUINT64_FORMAT ",%d,%.6f,%.6f,%.6f,%.6f,%.6f,"
            "%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%d\n",
            source_name (source), source, frame, c->history, d->conf,
            d->l, d->t, d->r, d->b, d->l, d->t, d->r, proposed_b,
            proposed_b - d->b, signal_text,
            "top_clipped+bottom_up+height_drop+structural", mode == MODE_ACTIVE);
      }
      if (mode == MODE_ACTIVE) {
        NvDsObjectMeta *obj = d->meta;
        const double left = clampd (d->l, 0.0, frame_w - 2.0);
        const double top = clampd (d->t, 0.0, frame_h - 2.0);
        const double right = clampd (d->r, left + 2.0, frame_w);
        const double bottom = clampd (proposed_b, top + 2.0, frame_h);
        obj->detector_bbox_info.org_bbox_coords.left = left;
        obj->detector_bbox_info.org_bbox_coords.top = top;
        obj->detector_bbox_info.org_bbox_coords.width = right - left;
        obj->detector_bbox_info.org_bbox_coords.height = bottom - top;
        obj->rect_params.left = left;
        obj->rect_params.top = top;
        obj->rect_params.width = right - left;
        obj->rect_params.height = bottom - top;
      }
      h->last = *d;
    } else {
      if (!h->frozen) append_reliable (h, d);
      h->last = *d;
    }
    pn263_history_pool_touch (&state->pool, state->slots, c->history, frame);
    if (pre_log) {
      const NvBbox_Coords *after = &d->meta->detector_bbox_info.org_bbox_coords;
      fprintf (pre_log,
          "%s,%u,%u,%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT ","
          "%" G_GUINT64_FORMAT ",%d,%d,%" G_GUINT64_FORMAT ","
          "%" G_GUINT64_FORMAT ",%.6f,"
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
          "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
          "%d,%d,%d,%d,%d,%d,%d\n",
          source_name (source), source, frame_meta->pad_index,
          frame_meta->buf_pts, frame_meta->ntp_timestamp, frame, c->detection,
          c->history, h->first_frame, h->detector_hits, d->conf,
          d->l, d->t, d->r, d->b, box_w (d), box_h (d),
          box_w (d) * box_h (d), box_cx (d), d->b,
          after->left, after->top, after->width, after->height,
          frame_w, frame_h, d->l <= 1.0, d->t <= 1.0,
          d->r >= frame_w - 1.0, d->b >= frame_h - 1.0,
          0, is_anomaly, mode == MODE_ACTIVE && is_anomaly);
    }
  }

  for (j = 0; j < nd; j++) {
    if (!detection_used[j]) {
      const int history_id = new_history (state, &dets[j], history_used);
      if (pre_log) {
        const Box *d = &dets[j];
        const guint64 first_frame = history_id >= 0 ?
            state->histories[history_id].first_frame : frame;
        const guint64 detector_hits = history_id >= 0 ?
            state->histories[history_id].detector_hits : 0;
        fprintf (pre_log,
            "%s,%u,%u,%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT ","
            "%" G_GUINT64_FORMAT ",%d,%d,%" G_GUINT64_FORMAT ","
            "%" G_GUINT64_FORMAT ",%.6f,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
            "%d,%d,%d,%d,1,0,0\n",
            source_name (source), source, frame_meta->pad_index,
            frame_meta->buf_pts, frame_meta->ntp_timestamp, frame, j,
            history_id, first_frame, detector_hits, d->conf,
            d->l, d->t, d->r, d->b, box_w (d), box_h (d),
            box_w (d) * box_h (d), box_cx (d), d->b,
            d->l, d->t, box_w (d), box_h (d), frame_w, frame_h,
            d->l <= 1.0, d->t <= 1.0, d->r >= frame_w - 1.0,
            d->b >= frame_h - 1.0);
      }
    }
  }
  if (state->anomaly_active_last_frame && !anomaly_active_this_frame)
    state->recovery_until_frame = frame + RECOVERY_HYSTERESIS_FRAMES;
  state->anomaly_active_last_frame = anomaly_active_this_frame;
  if (history_log) {
    fprintf (history_log,
        "frame,%s,%u,%" G_GUINT64_FORMAT ",%zu,%zu,%" G_GUINT64_FORMAT
        ",%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT
        ",%d\n",
        source_name (source), source, frame, state->pool.active_count,
        state->pool.high_water,
        state->pool.created_count - created_before,
        state->pool.retired_count - retired_before,
        state->pool.evicted_count - evicted_before,
        state->pool.reused_count - reused_before, epoch_reset);
  }
  if (pre_log) fflush (pre_log);
  if (decision_log) fflush (decision_log);
  if (recovery_log) fflush (recovery_log);
  if (history_log) fflush (history_log);
}

static GstPadProbeReturn
pre_tracker_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch_meta;
  NvDsMetaList *item;
  const gint64 start = g_get_monotonic_time ();
  (void) pad; (void) user_data;
  if (!buffer || !(batch_meta = gst_buffer_get_nvds_batch_meta (buffer)))
    return GST_PAD_PROBE_OK;
  for (item = batch_meta->frame_meta_list; item; item = item->next)
    process_frame ((NvDsFrameMeta *) item->data);
  probe_time_us += g_get_monotonic_time () - start;
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
post_tracker_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch_meta;
  NvDsMetaList *frame_item;
  (void) pad; (void) user_data;
  if (!post_log || !buffer || !(batch_meta = gst_buffer_get_nvds_batch_meta (buffer)))
    return GST_PAD_PROBE_OK;
  for (frame_item = batch_meta->frame_meta_list; frame_item; frame_item = frame_item->next) {
    NvDsFrameMeta *f = frame_item->data;
    NvDsMetaList *obj_item;
    for (obj_item = f->obj_meta_list; obj_item; obj_item = obj_item->next) {
      NvDsObjectMeta *o = obj_item->data;
      if (!is_person (o)) continue;
      const NvBbox_Coords *d = &o->detector_bbox_info.org_bbox_coords;
      const NvBbox_Coords *t = &o->tracker_bbox_info.org_bbox_coords;
      fprintf (post_log,
              "%s,%u,%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT ",%" G_GUINT64_FORMAT ","
              "%" G_GUINT64_FORMAT ",%.6f,%.6f,%d,"
              "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              source_name (f->source_id), f->source_id, (guint64) f->frame_num,
              f->buf_pts, f->ntp_timestamp,
              (guint64) o->object_id,
              o->confidence, o->tracker_confidence,
              o->confidence >= 0.0f,
          d->left, d->top, d->width, d->height,
          t->left, t->top, t->width, t->height,
          o->rect_params.left, o->rect_params.top,
          o->rect_params.width, o->rect_params.height);
    }
  }
  /* NvDCF's public past-frame metadata carries exact state and age if emitted.
   * Keep the newest sample per target instead of rewriting full trajectories
   * every batch. */
  for (NvDsMetaList *u = batch_meta->batch_user_meta_list; u; u = u->next) {
    NvDsUserMeta *user = (NvDsUserMeta *) u->data;
    if (!user || user->base_meta.meta_type != NVDS_TRACKER_PAST_FRAME_META ||
        !user->user_meta_data) continue;
    NvDsTargetMiscDataBatch *past = (NvDsTargetMiscDataBatch *) user->user_meta_data;
    for (guint si = 0; si < past->numFilled; si++) {
      NvDsTargetMiscDataStream *stream = &past->list[si];
      for (guint ti = 0; ti < stream->numFilled; ti++) {
        NvDsTargetMiscDataObject *target = &stream->list[ti];
        if (!target->list || target->numObj == 0) continue;
        NvDsTargetMiscDataFrame *sample = &target->list[target->numObj - 1];
        if (tracker_state_log)
          fprintf (tracker_state_log,
              "%s,%u,%" G_GUINT64_FORMAT ",%s,%" G_GUINT64_FORMAT ",%s,%u,"
              "%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.6f,%.6f\n",
              source_name (stream->streamID), stream->streamID,
              (guint64) sample->frameNum,
              target->objLabel[0] ? target->objLabel : "",
              (guint64) target->uniqueId, tracker_state_name (sample->trackerState),
              sample->age, sample->confidence, sample->tBbox.left,
              sample->tBbox.top, sample->tBbox.width, sample->tBbox.height,
              sample->visibility, sample->ptWorldFeet[0], sample->ptWorldFeet[1]);
      }
    }
  }
  if (post_log) fflush (post_log);
  if (tracker_state_log) fflush (tracker_state_log);
  return GST_PAD_PROBE_OK;
}

static void
shutdown_logs (void)
{
  guint source;
  for (source = 0; source < MAX_SOURCES; source++) {
    SourceState *state = &states[source];
    const guint64 frame = state->pool.has_last_frame ? state->pool.last_frame : 0;
    pn263_history_pool_clear (&state->pool, state->slots, MAX_HISTORIES);
    memset (state->histories, 0, sizeof (state->histories));
    if (history_log && source < 2) {
      fprintf (history_log,
          "final,%s,%u,%" G_GUINT64_FORMAT ",%zu,%zu,0,0,0,0,0\n",
          source_name (source), source, frame, state->pool.active_count,
          state->pool.high_water);
    }
  }
  g_print ("PN263_BBOX_SUMMARY mode=%s frames=%" G_GUINT64_FORMAT
      " detections=%" G_GUINT64_FORMAT " decisions=%" G_GUINT64_FORMAT
      " cam01=%" G_GUINT64_FORMAT " cam04=%" G_GUINT64_FORMAT
      " probe_us=%" G_GINT64_FORMAT " per_detection_us=%.6f\n",
      mode == MODE_ACTIVE ? "active" : mode == MODE_SHADOW ? "shadow" : "off",
      frames_seen, detections_seen, decisions_seen,
      decisions_by_source[0], decisions_by_source[1], probe_time_us,
      detections_seen ? (double) probe_time_us / detections_seen : 0.0);
  if (decision_log) fclose (decision_log);
  if (pre_log) fclose (pre_log);
  if (post_log) fclose (post_log);
  if (recovery_log) fclose (recovery_log);
  if (history_log) fclose (history_log);
  if (tracker_state_log) fclose (tracker_state_log);
  decision_log = pre_log = post_log = recovery_log = history_log = NULL;
}

static FILE *
open_csv (const char *directory, const char *name, const char *header)
{
  char *path = g_build_filename (directory, name, NULL);
  FILE *f = fopen (path, "w");
  if (!f) g_printerr ("PN263_BBOX ERROR cannot open %s\n", path);
  else {
    (void) setvbuf (f, NULL, _IOLBF, 0);
    fprintf (f, "%s\n", header);
    fflush (f);
  }
  g_free (path);
  return f;
}

gboolean
pn263_bbox_correction_attach (GstElement *primary_gie, GstElement *tracker)
{
  const char *mode_text = g_getenv ("PN263_BBOX_MODE");
  const char *directory = g_getenv ("PN263_BBOX_LOG_DIR");
  GstPad *pad;
  if (initialized) return TRUE;
  initialized = TRUE;
  if (!mode_text || g_strcmp0 (mode_text, "off") == 0) mode = MODE_OFF;
  else if (g_strcmp0 (mode_text, "shadow") == 0) mode = MODE_SHADOW;
  else if (g_strcmp0 (mode_text, "active") == 0) mode = MODE_ACTIVE;
  else {
    g_printerr ("PN263_BBOX ERROR invalid PN263_BBOX_MODE=%s\n", mode_text);
    return FALSE;
  }
  if (!directory) directory = ".";
  if (g_mkdir_with_parents (directory, 0755) != 0) {
    g_printerr ("PN263_BBOX ERROR cannot create log directory %s\n", directory);
    return FALSE;
  }
  decision_log = open_csv (directory, "bbox_decisions.csv",
      "source,source_id,frame,history,confidence,orig_left,orig_top,orig_right,orig_bottom,proposed_left,proposed_top,proposed_right,proposed_bottom,bottom_delta,signals,trigger_reason,applied");
  pre_log = open_csv (directory, "pretracker_geometry.csv",
      "camera_id,source_id,pad_index,pts,ntp_timestamp,frame,detection_index,history_id,history_first_frame,detector_hit_count,confidence,left,top,right,bottom,width,height,area,bottom_center_x,bottom_center_y,received_left,received_top,received_width,received_height,frame_width,frame_height,touches_left,touches_top,touches_right,touches_bottom,history_created,anomaly,applied");
  post_log = open_csv (directory, "posttracker_geometry.csv",
      "camera_id,source_id,frame,pts,ntp_timestamp,native_track_id,detector_confidence,tracker_confidence,detector_associated,det_left,det_top,det_width,det_height,tracker_left,tracker_top,tracker_width,tracker_height,rect_left,rect_top,rect_width,rect_height");
  tracker_state_log = open_csv (directory, "tracker_state.csv",
      "camera_id,source_id,frame,label,native_track_id,nvdcf_state,track_age,tracker_confidence,left,top,width,height,visibility,world_foot_x,world_foot_y");
  recovery_log = open_csv (directory, "bbox_recovery_suppressed.csv",
      "source,source_id,frame,history,candidate_bottom_delta,recovery_until,reason");
  history_log = open_csv (directory, "bbox_history_stats.csv",
      "phase,source,source_id,frame,active_histories,high_water,created,retired,evicted,reused,frame_reset");
  if (!decision_log || !pre_log || !post_log || !tracker_state_log ||
      !recovery_log || !history_log) return FALSE;
  pn263_history_pool_init (&states[0].pool);
  pn263_history_pool_init (&states[1].pool);
  fprintf (history_log, "initial,CAM-01,0,0,0,0,0,0,0,0,0\n");
  fprintf (history_log, "initial,CAM-04,1,0,0,0,0,0,0,0,0\n");
  fflush (history_log);
  atexit (shutdown_logs);

  if (!primary_gie || !tracker) {
    g_printerr ("PN263_BBOX ERROR PGIE or tracker element missing\n");
    return FALSE;
  }
  pad = gst_element_get_static_pad (primary_gie, "src");
  if (!pad) return FALSE;
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, pre_tracker_probe, NULL, NULL);
  gst_object_unref (pad);
  pad = gst_element_get_static_pad (tracker, "src");
  if (!pad) return FALSE;
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, post_tracker_probe, NULL, NULL);
  gst_object_unref (pad);
  g_print ("PN263_BBOX attached mode=%s log_dir=%s causal=current-and-past-only\n",
      mode == MODE_ACTIVE ? "active" : mode == MODE_SHADOW ? "shadow" : "off",
      directory);
  return TRUE;
}
