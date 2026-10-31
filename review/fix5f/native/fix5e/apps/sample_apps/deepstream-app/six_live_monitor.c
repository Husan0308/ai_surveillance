/* Isolated official-app adapter: authentication, read-only metadata, OSD labels.
 * No source, detector, tracker, CUDA, image-copy or encoder implementation. */
#define _GNU_SOURCE
#include "six_live_monitor.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <gst/base/gstbasesink.h>

#define TELEMETRY_OBJECT_LIMIT 256u
#define TELEMETRY_OBJECT_SCAN_LIMIT 1024u
#define TELEMETRY_FRAME_META_LIMIT 6u
#define PREVIEW_PROGRESS_STAGE 4u
#define SOURCE_OWNER_META_NAME "NVIDIA.SIXLIVE.SOURCE_GENERATION"
#define SOURCE_OWNER_MAGIC G_GUINT64_CONSTANT (0x5349584c49564547)
#define SOURCE_OWNER_VERSION 1u

typedef struct {
  guint64 count, objects, last_frame, first_ns, last_ns, pts, pts_regressions, max_gap_ns;
  guint width, height;
} Stage;
typedef struct {
  guint64 object_id;
  gfloat detector_confidence, tracker_confidence;
  gfloat bbox[4], detector_bbox[4], tracker_bbox[4], rect_params[4];
  gboolean tracker_id_valid, detector_bbox_valid, tracker_bbox_valid;
} TelemetryObject;
typedef struct {
  gboolean valid, complete, pts_valid;
  guint64 sequence, frame_num, pts, monotonic_ns, source_generation;
  guint64 source_sequence;
  guint source_id, source_width, source_height, pipeline_width, pipeline_height;
  guint object_count;
  TelemetryObject objects[TELEMETRY_OBJECT_LIMIT];
} TrackerFrameSnapshot;
typedef struct {
  guint64 magic;
  guint64 generation;
  guint64 source_sequence;
  guint64 source_pts;
  guint32 source_id;
  guint32 version;
} SourceOwnerToken;
typedef enum {
  SOURCE_OWNER_OK,
  SOURCE_OWNER_MISSING,
  SOURCE_OWNER_DUPLICATE,
  SOURCE_OWNER_MALFORMED,
  SOURCE_OWNER_WRONG_SOURCE,
  SOURCE_OWNER_STALE,
  SOURCE_OWNER_UNAVAILABLE
} SourceOwnerResult;
typedef struct {
  guint64 revision, generation;
  const gchar *state;
} SourceLifecycleSnapshot;
static Stage stages[5][6];
static gchar *users[6], *passwords[6], *uri_hashes[6];
static const gchar *rooms[6] = {"Dev Room", "Entrance", "Main Room", "Dev Room", "Entrance", "Main Room"};
static GMutex monitor_lock;
static gboolean enabled, fatal, telemetry_enabled;
static guint timer;
static GstElement *display_sink;
static guint64 last_counts[6], last_tick;
static gdouble fps[6];
static GMutex telemetry_lock;
static TrackerFrameSnapshot tracker_snapshots[6];
static guint64 preview_stage_generation[6];
static SourceLifecycleSnapshot source_lifecycle[6];
static guint64 lifecycle_sent_revision[6];
static guint64 source_generation[6];
static guint64 source_output_sequence[6];
static gboolean source_epoch_ready[6];
static gint source_owner_meta_type = NVDS_GST_INVALID_META;
static gint source_owner_attach_failures[6];
static gint source_owner_missing[6];
static gint source_owner_rejected[6];
static gint ownership_stopping;
static GThread *telemetry_thread;
static gint telemetry_stop;
static GstElement *live_pipeline;
static gchar pipeline_status[16] = "starting";
static gboolean pipeline_failed;
static void update (guint stage, guint id, guint64 frame, guint64 pts,
    guint width, guint height, guint objects);

static guint64 monotonic_ns (void) {
  return (guint64) g_get_monotonic_time () * 1000;
}

static void emit_pipeline_state (const gchar *state) {
  if (!state || (strcmp (state, "starting") && strcmp (state, "playing") &&
          strcmp (state, "error") && strcmp (state, "stopped")))
    return;
  g_print ("SIX_PIPELINE {\"status\":\"%s\",\"monotonic_ns\":%" G_GUINT64_FORMAT "}\n",
      state, monotonic_ns ());
  fflush (stdout);
}

void six_live_pipeline_state (const gchar *state) {
  if (!enabled || !telemetry_enabled || !state)
    return;
  g_mutex_lock (&telemetry_lock);
  if (!strcmp (state, "error"))
    pipeline_failed = TRUE;
  if (!strcmp (state, "stopped") || !strcmp (state, "starting") ||
      !strcmp (state, "playing") || !strcmp (state, "error"))
    g_strlcpy (pipeline_status, pipeline_failed && strcmp (state, "stopped") ? "error" : state,
        sizeof pipeline_status);
  gchar copy[sizeof pipeline_status];
  g_strlcpy (copy, pipeline_status, sizeof copy);
  g_mutex_unlock (&telemetry_lock);
  emit_pipeline_state (copy);
}

static void set_source_lifecycle_locked (guint id, const gchar *state) {
  if (id >= 6)
    return;
  if (!g_strcmp0 (source_lifecycle[id].state, state))
    return;
  source_lifecycle[id].state = state;
  source_lifecycle[id].generation = source_generation[id];
  source_lifecycle[id].revision++;
}

void six_live_source_reconnecting (guint id) {
  if (!enabled || id >= 6 || g_atomic_int_get (&ownership_stopping))
    return;
  /* Quarantine immediately, but do not claim a new epoch until the source
   * bin has stopped every internal producer and reached GST_STATE_NULL. */
  g_mutex_lock (&monitor_lock);
  g_mutex_lock (&telemetry_lock);
  source_epoch_ready[id] = FALSE;
  if (g_strcmp0 (source_lifecycle[id].state, "reconnecting") != 0)
    set_source_lifecycle_locked (id, "reconnecting");
  g_mutex_unlock (&telemetry_lock);
  g_mutex_unlock (&monitor_lock);
}

/* Called only after reset_source_pipeline() confirms the complete source bin
 * is NULL with no pending state transition. All frames already beyond the
 * source-bin output have immutable old tokens; all internal queues/decoder
 * tasks have been stopped before the next generation is opened. */
void six_live_source_generation_boundary (guint id) {
  if (!enabled || id >= 6 || g_atomic_int_get (&ownership_stopping))
    return;
  g_mutex_lock (&monitor_lock);
  g_mutex_lock (&telemetry_lock);
  if (!source_epoch_ready[id]) {
    source_generation[id]++;
    for (guint stage = 0; stage < 5; stage++)
      memset (&stages[stage][id], 0, sizeof stages[stage][id]);
    last_counts[id] = 0;
    tracker_snapshots[id].valid = FALSE;
    preview_stage_generation[id] = source_generation[id];
    source_epoch_ready[id] = TRUE;
    source_lifecycle[id].generation = source_generation[id];
    source_lifecycle[id].revision++;
  }
  g_mutex_unlock (&telemetry_lock);
  g_mutex_unlock (&monitor_lock);
}

static gpointer source_owner_copy (gpointer data, gpointer user_data) {
  (void) user_data;
  if (!data)
    return NULL;
  SourceOwnerToken *copy = g_malloc (sizeof *copy);
  memcpy (copy, data, sizeof *copy);
  return copy;
}

static void source_owner_release (gpointer data, gpointer user_data) {
  (void) user_data;
  g_free (data);
}

/* DeepStream 9.1 calls the transform with an NvDsUserMeta that wraps the
 * Gst-side data, as demonstrated by deepstream-gst-metadata-test. */
static gpointer source_owner_to_frame_meta (gpointer data, gpointer user_data) {
  (void) user_data;
  NvDsUserMeta *gst_user_meta = data;
  SourceOwnerToken *token = gst_user_meta ? gst_user_meta->user_meta_data : NULL;
  if (!token || token->magic != SOURCE_OWNER_MAGIC ||
      token->version != SOURCE_OWNER_VERSION || token->source_id >= 6 ||
      token->source_sequence == 0)
    return NULL;
  return source_owner_copy (token, NULL);
}

static void source_owner_frame_meta_release (gpointer data, gpointer user_data) {
  (void) user_data;
  NvDsUserMeta *frame_user_meta = data;
  if (!frame_user_meta)
    return;
  source_owner_release (frame_user_meta->user_meta_data, NULL);
  frame_user_meta->user_meta_data = NULL;
}

static gboolean buffer_has_source_owner_meta (GstBuffer *buffer) {
  gpointer iter = NULL;
  GstMeta *gst_meta;
  while ((gst_meta = gst_buffer_iterate_meta (buffer, &iter))) {
    if (gst_meta->info->api != NVDS_META_API_TYPE)
      continue;
    NvDsMeta *nvds_meta = (NvDsMeta *) gst_meta;
    if (nvds_meta->meta_type == source_owner_meta_type)
      return TRUE;
  }
  return FALSE;
}

static GstPadProbeReturn source_lineage_probe (GstPad *pad,
    GstPadProbeInfo *info, gpointer data) {
  (void) pad;
  guint id = GPOINTER_TO_UINT (data);
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  if (!enabled || !buffer || id >= 6 || source_owner_meta_type < 0 ||
      g_atomic_int_get (&ownership_stopping))
    return GST_PAD_PROBE_OK;

  /* This probe is on the source-bin ghost src pad, after decode/conversion
   * and immediately before the corresponding nvstreammux sink pad. */
  if (!g_mutex_trylock (&monitor_lock)) {
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }
  if (!g_mutex_trylock (&telemetry_lock)) {
    g_mutex_unlock (&monitor_lock);
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }
  gboolean can_stamp = source_epoch_ready[id] &&
      g_strcmp0 (source_lifecycle[id].state, "reconnecting") != 0 &&
      g_strcmp0 (source_lifecycle[id].state, "offline") != 0;
  SourceOwnerToken token = {0};
  if (can_stamp) {
    token.magic = SOURCE_OWNER_MAGIC;
    token.version = SOURCE_OWNER_VERSION;
    token.source_id = id;
    token.generation = source_generation[id];
    token.source_sequence = ++source_output_sequence[id];
    token.source_pts = GST_BUFFER_PTS_IS_VALID (buffer) ?
        GST_BUFFER_PTS (buffer) : GST_CLOCK_TIME_NONE;
  }
  g_mutex_unlock (&telemetry_lock);
  g_mutex_unlock (&monitor_lock);
  if (!can_stamp) {
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }

  /* A GstBuffer header may be shared after a converter/tee. Make only the
   * header writable; this does not map or copy NVMM pixel payloads. */
  GstBuffer *writable = gst_buffer_make_writable (buffer);
  if (!writable) {
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }
  GST_PAD_PROBE_INFO_DATA (info) = writable;
  if (buffer_has_source_owner_meta (writable)) {
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }

  SourceOwnerToken *owned = g_malloc (sizeof *owned);
  memcpy (owned, &token, sizeof *owned);
  NvDsMeta *meta = gst_buffer_add_nvds_meta (writable, owned, NULL,
      source_owner_copy, source_owner_release);
  if (!meta) {
    source_owner_release (owned, NULL);
    g_atomic_int_inc (&source_owner_attach_failures[id]);
    return GST_PAD_PROBE_OK;
  }
  meta->meta_type = source_owner_meta_type;
  meta->gst_to_nvds_meta_transform_func = source_owner_to_frame_meta;
  meta->gst_to_nvds_meta_release_func = source_owner_frame_meta_release;
  return GST_PAD_PROBE_OK;
}

static SourceOwnerResult source_frame_owner (NvDsFrameMeta *fm,
    SourceOwnerToken *copy) {
  if (!fm || fm->source_id >= 6 || fm->pad_index != fm->source_id)
    return SOURCE_OWNER_WRONG_SOURCE;
  if (source_owner_meta_type < 0)
    return SOURCE_OWNER_MISSING;
  NvDsUserMeta *found = NULL;
  guint matches = 0;
  for (NvDsMetaList *node = fm->frame_user_meta_list; node; node = node->next) {
    NvDsUserMeta *user_meta = node->data;
    if (user_meta && user_meta->base_meta.meta_type == source_owner_meta_type) {
      found = user_meta;
      matches++;
    }
  }
  if (!matches)
    return SOURCE_OWNER_MISSING;
  if (matches != 1)
    return SOURCE_OWNER_DUPLICATE;
  SourceOwnerToken *token = found ? found->user_meta_data : NULL;
  if (!token || token->magic != SOURCE_OWNER_MAGIC ||
      token->version != SOURCE_OWNER_VERSION || token->source_id >= 6 ||
      token->source_sequence == 0)
    return SOURCE_OWNER_MALFORMED;
  if (token->source_id != fm->source_id)
    return SOURCE_OWNER_WRONG_SOURCE;
  if (!g_mutex_trylock (&telemetry_lock))
    return SOURCE_OWNER_UNAVAILABLE;
  gboolean current = !g_atomic_int_get (&ownership_stopping) &&
      source_epoch_ready[fm->source_id] &&
      source_generation[fm->source_id] == token->generation &&
      g_strcmp0 (source_lifecycle[fm->source_id].state, "reconnecting") != 0 &&
      g_strcmp0 (source_lifecycle[fm->source_id].state, "offline") != 0;
  g_mutex_unlock (&telemetry_lock);
  if (!current)
    return SOURCE_OWNER_STALE;
  if (copy)
    memcpy (copy, token, sizeof *copy);
  return SOURCE_OWNER_OK;
}

gboolean six_live_preview_frame_generation (NvDsFrameMeta *frame_meta,
    guint64 *generation) {
  if (!enabled || !generation)
    return FALSE;
  /* Resolve from the immutable user-meta token on this exact NvDsFrameMeta.
   * Frame number and PTS may repeat after reconnect and cannot establish
   * generation ownership for a cached tiler frame. */
  SourceOwnerToken owner = {0};
  if (source_frame_owner (frame_meta, &owner) != SOURCE_OWNER_OK)
    return FALSE;
  *generation = owner.generation;
  return TRUE;
}

void six_live_preview_frame (guint id, guint frame_num, guint64 source_pts,
    guint64 observed_generation) {
  if (!enabled || !telemetry_enabled || id >= 6)
    return;
  if (!GST_CLOCK_TIME_IS_VALID (source_pts) || !g_mutex_trylock (&monitor_lock))
    return;
  if (!g_mutex_trylock (&telemetry_lock)) {
    g_mutex_unlock (&monitor_lock);
    return;
  }
  gboolean current_generation = source_epoch_ready[id] &&
      source_generation[id] == observed_generation &&
      g_strcmp0 (source_lifecycle[id].state, "reconnecting") != 0 &&
      g_strcmp0 (source_lifecycle[id].state, "offline") != 0;
  g_mutex_unlock (&telemetry_lock);
  if (!current_generation) {
    g_mutex_unlock (&monitor_lock);
    return;
  }
  Stage *stage = &stages[PREVIEW_PROGRESS_STAGE][id];
  if (preview_stage_generation[id] != observed_generation) {
    memset (stage, 0, sizeof *stage);
    preview_stage_generation[id] = observed_generation;
  }
  if (stage->count && (frame_num <= stage->last_frame ||
          (GST_CLOCK_TIME_IS_VALID (stage->pts) && source_pts <= stage->pts))) {
    g_mutex_unlock (&monitor_lock);
    return;
  }
  update (PREVIEW_PROGRESS_STAGE, id, frame_num, source_pts,
      stage->width, stage->height, 0);
  g_mutex_unlock (&monitor_lock);
}

static void wipe (gchar **p) {
  if (*p) { explicit_bzero (*p, strlen (*p)); g_free (*p); *p = NULL; }
}
static gchar *decode_secret (GKeyFile *key, const gchar *group, const gchar *name) {
  gchar *encoded = g_key_file_get_string (key, group, name, NULL);
  gsize size = 0;
  if (!encoded || strlen (encoded) > 4096) { g_free (encoded); return NULL; }
  guchar *raw = g_base64_decode (encoded, &size);
  explicit_bzero (encoded, strlen (encoded)); g_free (encoded);
  if (!size || size > 2048 || memchr (raw, 0, size) || memchr (raw, '\n', size) || memchr (raw, '\r', size)) {
    if (raw) { explicit_bzero (raw, size); g_free (raw); } return NULL;
  }
  gchar *value = g_strndup ((gchar *) raw, size);
  explicit_bzero (raw, size); g_free (raw); return value;
}
gboolean six_live_prepare (AppCtx *app) {
  const gchar *path = g_getenv ("SIX_LIVE_AUTH_FILE");
  if (!path) return TRUE;
  struct stat st; gchar data[65537]; gsize used = 0; GKeyFile *key = g_key_file_new ();
  int fd = open (path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0 || fstat (fd, &st) || !S_ISREG (st.st_mode) || st.st_uid != getuid () ||
      (st.st_mode & 077) || st.st_size <= 0 || st.st_size > 65536) goto fail;
  while (used < (gsize) st.st_size) {
    ssize_t got = read (fd, data + used, st.st_size - used);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) goto fail;
    used += got;
  }
  close (fd); fd = -1;
  if (!g_key_file_load_from_data (key, data, used, G_KEY_FILE_NONE, NULL) || app->config.num_source_sub_bins != 6) goto fail;
  for (guint i = 0; i < 6; i++) {
    gchar group[16]; g_snprintf (group, sizeof group, "source%u", i);
    NvDsSourceConfig *cfg = &app->config.multi_source_config[i];
    if (cfg->type != NV_DS_SOURCE_RTSP || cfg->select_rtp_protocol != 4 || cfg->camera_id != i+1 || !cfg->uri) goto fail;
    users[i] = decode_secret (key, group, "username64");
    passwords[i] = decode_secret (key, group, "password64");
    uri_hashes[i] = g_key_file_get_string (key, group, "uri_sha256", NULL);
    gchar *actual = g_compute_checksum_for_string (G_CHECKSUM_SHA256, cfg->uri, -1);
    gboolean ok = users[i] && passwords[i] && uri_hashes[i] && !g_strcmp0 (actual, uri_hashes[i]);
    g_free (actual); if (!ok) goto fail;
  }
  explicit_bzero (data, sizeof data); g_key_file_unref (key);
  /* The soak/launcher owns bounded runtime and orderly shutdown. An 800 s
   * process alarm here terminated healthy daily monitoring with exit 124,
   * bypassing six_live_stop() and its credential/telemetry cleanup. */
  enabled = TRUE;
  g_atomic_int_set (&ownership_stopping, 0);
  g_mutex_lock (&monitor_lock);
  g_mutex_lock (&telemetry_lock);
  for (guint i = 0; i < 6; i++) {
    source_epoch_ready[i] = TRUE; /* Initial generation 0 is a valid epoch. */
    source_lifecycle[i].state = "starting";
    source_lifecycle[i].generation = 0;
    source_lifecycle[i].revision++;
  }
  g_mutex_unlock (&telemetry_lock);
  g_mutex_unlock (&monitor_lock);
  telemetry_enabled = g_strcmp0 (g_getenv ("SIX_ML_TELEMETRY_ENABLED"), "1") == 0;
  return TRUE;
fail:
  if (fd >= 0) close (fd);
  explicit_bzero (data, sizeof data); g_key_file_unref (key);
  for (guint i=0;i<6;i++) { wipe (&users[i]); wipe (&passwords[i]); wipe (&uri_hashes[i]); }
  g_printerr ("SIX_LIVE_ERROR code=CREDENTIAL_HANDOFF_INVALID\n"); return FALSE;
}
gboolean six_live_source_auth (GstElement *source, guint id, const gchar *uri) {
  if (!g_getenv ("SIX_LIVE_AUTH_FILE")) return TRUE;
  if (!enabled || id >= 6 || !source || !uri) return FALSE;
  gchar *actual = g_compute_checksum_for_string (G_CHECKSUM_SHA256, uri, -1);
  gboolean ok = !g_strcmp0 (actual, uri_hashes[id]); g_free (actual);
  if (!ok) return FALSE;
  g_object_set (source, "user-id", users[id], "user-pw", passwords[id], NULL);
  return TRUE;
}
static void update (guint stage, guint id, guint64 frame, guint64 pts, guint width, guint height, guint objects) {
  if (id >= 6 || stage >= 5) { fatal = TRUE; return; }
  guint64 now = (guint64) g_get_monotonic_time () * 1000;
  Stage *s = &stages[stage][id];
  if (s->count && now - s->last_ns > s->max_gap_ns) s->max_gap_ns = now - s->last_ns;
  if (s->count && GST_CLOCK_TIME_IS_VALID (pts) && GST_CLOCK_TIME_IS_VALID (s->pts) && pts < s->pts) s->pts_regressions++;
  if (!s->count) s->first_ns = now;
  s->count++; s->objects += objects; s->last_frame=frame; s->last_ns=now; s->pts=pts;
  if (width) s->width=width;
  if (height) s->height=height;
}

static gboolean rect_values (gfloat left, gfloat top, gfloat width, gfloat height,
    guint max_width, guint max_height, gfloat out[4]) {
  if (!isfinite (left) || !isfinite (top) || !isfinite (width) || !isfinite (height) ||
      left < 0.0f || top < 0.0f || width <= 0.0f || height <= 0.0f ||
      !max_width || !max_height || left + width > max_width + 1.0f ||
      top + height > max_height + 1.0f)
    return FALSE;
  out[0] = left; out[1] = top; out[2] = left + width; out[3] = top + height;
  return TRUE;
}

static gboolean snapshot_tracker_frame (NvDsFrameMeta *fm, GstBuffer *buffer,
    const SourceOwnerToken *owner) {
  if (!fm || !owner || fm->source_id >= 6 || fm->pad_index != fm->source_id ||
      !g_mutex_trylock (&telemetry_lock))
    return FALSE;
  TrackerFrameSnapshot *snapshot = &tracker_snapshots[fm->source_id];
  TrackerFrameSnapshot next;
  memset (&next, 0, sizeof next);
  next.valid = TRUE;
  next.complete = fm->source_frame_width > 0 && fm->source_frame_height > 0 &&
      fm->pipeline_width > 0 && fm->pipeline_height > 0;
  next.sequence = snapshot->sequence + 1;
  next.frame_num = fm->frame_num;
  next.pts_valid = GST_CLOCK_TIME_IS_VALID (fm->buf_pts);
  next.pts = next.pts_valid ? fm->buf_pts : 0;
  next.monotonic_ns = monotonic_ns ();
  next.source_generation = owner->generation;
  next.source_sequence = owner->source_sequence;
  next.source_id = fm->source_id;
  next.source_width = fm->source_frame_width;
  next.source_height = fm->source_frame_height;
  next.pipeline_width = fm->pipeline_width;
  next.pipeline_height = fm->pipeline_height;
  guint scanned = 0;
  for (NvDsMetaList *node = fm->obj_meta_list; node; node = node->next) {
    if (++scanned > TELEMETRY_OBJECT_SCAN_LIMIT) {
      next.complete = FALSE;
      break;
    }
    NvDsObjectMeta *object = node->data;
    if (!object || object->class_id != 0 || object->object_id == UNTRACKED_OBJECT_ID)
      continue;
    if (next.object_count >= TELEMETRY_OBJECT_LIMIT) {
      next.complete = FALSE;
      break;
    }
    TelemetryObject *out = &next.objects[next.object_count];
    out->object_id = object->object_id; /* ID 0 is a valid NvDCF ID. */
    out->tracker_id_valid = TRUE;
    out->detector_confidence = object->confidence;
    out->tracker_confidence = object->tracker_confidence;
    out->detector_bbox_valid = rect_values (
        object->detector_bbox_info.org_bbox_coords.left,
        object->detector_bbox_info.org_bbox_coords.top,
        object->detector_bbox_info.org_bbox_coords.width,
        object->detector_bbox_info.org_bbox_coords.height,
        next.pipeline_width, next.pipeline_height, out->detector_bbox);
    out->tracker_bbox_valid = rect_values (
        object->tracker_bbox_info.org_bbox_coords.left,
        object->tracker_bbox_info.org_bbox_coords.top,
        object->tracker_bbox_info.org_bbox_coords.width,
        object->tracker_bbox_info.org_bbox_coords.height,
        next.pipeline_width, next.pipeline_height, out->tracker_bbox);
    if (!rect_values (object->rect_params.left, object->rect_params.top,
            object->rect_params.width, object->rect_params.height,
            next.pipeline_width, next.pipeline_height, out->rect_params) ||
        !isfinite (out->detector_confidence) || !isfinite (out->tracker_confidence)) {
      next.complete = FALSE;
      break;
    }
    memcpy (out->bbox, out->rect_params, sizeof out->bbox);
    next.object_count++;
  }
  if (!GST_BUFFER_PTS_IS_VALID (buffer))
    next.complete = FALSE;
  *snapshot = next; /* Only copied scalar data survives this probe callback. */
  g_mutex_unlock (&telemetry_lock);
  return TRUE;
}

static void append_number (GString *line, gdouble value) {
  gchar number[G_ASCII_DTOSTR_BUF_SIZE];
  g_ascii_dtostr (number, sizeof number, value);
  g_string_append (line, number);
}

static void append_rect (GString *line, const gfloat rect[4], gboolean valid) {
  if (!valid) { g_string_append (line, "null"); return; }
  g_string_append_c (line, '[');
  for (guint i = 0; i < 4; i++) {
    if (i) g_string_append_c (line, ',');
    append_number (line, rect[i]);
  }
  g_string_append_c (line, ']');
}

static void emit_tracker_frame (const TrackerFrameSnapshot *s) {
  if (!s->valid)
    return;
  GString *line = g_string_sized_new (2048);
  g_string_append_printf (line,
      "SIX_TRACKER_FRAME {\"source_id\":%u,\"camera_id\":\"CAM-%02u\","
      "\"frame_num\":%" G_GUINT64_FORMAT ",\"source_sequence\":%" G_GUINT64_FORMAT ","
      "\"pts_valid\":%s,\"pts_ns\":",
      s->source_id, s->source_id + 1, s->frame_num, s->source_sequence,
      s->pts_valid ? "true" : "false");
  if (s->pts_valid) g_string_append_printf (line, "%" G_GUINT64_FORMAT, s->pts);
  else g_string_append (line, "null");
  g_string_append_printf (line,
      ",\"monotonic_ns\":%" G_GUINT64_FORMAT ",\"source_generation\":%" G_GUINT64_FORMAT
      ",\"source_frame_width\":%u,\"source_frame_height\":%u,"
      "\"pipeline_width\":%u,\"pipeline_height\":%u,\"complete\":%s,\"objects\":[",
      s->monotonic_ns, s->source_generation, s->source_width, s->source_height,
      s->pipeline_width, s->pipeline_height, s->complete ? "true" : "false");
  gboolean comma = FALSE;
  for (guint i = 0; s->complete && i < s->object_count; i++) {
    const TelemetryObject *o = &s->objects[i];
    if (comma) g_string_append_c (line, ',');
    comma = TRUE;
    g_string_append_printf (line,
        "{\"class_id\":0,\"native_track_id\":%" G_GUINT64_FORMAT ",\"bbox\":",
        o->object_id);
    append_rect (line, o->bbox, TRUE);
    g_string_append (line, ",\"detector_bbox\":");
    append_rect (line, o->detector_bbox, o->detector_bbox_valid);
    g_string_append (line, ",\"tracker_bbox\":");
    append_rect (line, o->tracker_bbox, o->tracker_bbox_valid);
    g_string_append (line, ",\"rect_params\":");
    append_rect (line, o->rect_params, TRUE);
    g_string_append (line, ",\"detector_confidence\":");
    append_number (line, o->detector_confidence);
    g_string_append (line, ",\"tracker_confidence\":");
    append_number (line, o->tracker_confidence);
    g_string_append_c (line, '}');
  }
  g_string_append (line, "],\"detector_proposals\":[");
  comma = FALSE;
  for (guint i = 0; s->complete && i < s->object_count; i++) {
    const TelemetryObject *o = &s->objects[i];
    if (!o->detector_bbox_valid || !isfinite (o->detector_confidence) ||
        o->detector_confidence < 0.0f || o->detector_confidence > 1.0f)
      continue;
    if (comma) g_string_append_c (line, ',');
    comma = TRUE;
    g_string_append_printf (line, "{\"proposal_index\":%u,\"confidence\":", i);
    append_number (line, o->detector_confidence);
    g_string_append (line, ",\"bbox\":");
    append_rect (line, o->detector_bbox, TRUE);
    g_string_append_c (line, '}');
  }
  g_string_append (line, "]}\n");
  if (line->len <= 240u * 1024u) {
    g_print ("%s", line->str);
    fflush (stdout);
  }
  g_string_free (line, TRUE);
}

static gpointer telemetry_pump (gpointer unused) {
  guint64 sent_frame[6] = {0};
  (void) unused;
  while (!g_atomic_int_get (&telemetry_stop)) {
    for (guint i = 0; i < 6; i++) {
      TrackerFrameSnapshot snapshot;
      SourceLifecycleSnapshot lifecycle;
      gboolean have_frame = FALSE, have_lifecycle = FALSE;
      if (!g_mutex_trylock (&telemetry_lock))
        continue;
      if (tracker_snapshots[i].valid && tracker_snapshots[i].sequence != sent_frame[i]) {
        snapshot = tracker_snapshots[i];
        have_frame = TRUE;
      }
      lifecycle = source_lifecycle[i];
      if (lifecycle.revision > lifecycle_sent_revision[i])
        have_lifecycle = TRUE;
      g_mutex_unlock (&telemetry_lock);
      if (have_lifecycle) {
        g_print ("SIX_SOURCE_STATE {\"source_id\":%u,\"camera_id\":\"CAM-%02u\","
            "\"state\":\"%s\",\"source_generation\":%" G_GUINT64_FORMAT
            ",\"monotonic_ns\":%" G_GUINT64_FORMAT "}\n",
            i, i + 1, lifecycle.state ? lifecycle.state : "offline",
            lifecycle.generation, monotonic_ns ());
        lifecycle_sent_revision[i] = lifecycle.revision;
      }
      if (have_frame) {
        emit_tracker_frame (&snapshot);
        sent_frame[i] = snapshot.sequence;
      }
    }
    fflush (stdout);
    g_usleep (100000);
  }
  return NULL;
}
static GstPadProbeReturn decode_probe (GstPad *pad, GstPadProbeInfo *info, gpointer data) {
  guint id = GPOINTER_TO_UINT (data); GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  if (!enabled || !buffer) return GST_PAD_PROBE_OK;
  guint width=0,height=0; GstCaps *caps=gst_pad_get_current_caps (pad);
  if (caps && gst_caps_get_size (caps)) {
    const GstStructure *st=gst_caps_get_structure (caps,0); gint w=0,h=0;
    gst_structure_get_int(st,"width",&w);gst_structure_get_int(st,"height",&h);
    if (gst_caps_features_contains (gst_caps_get_features(caps,0), "memory:NVMM")) {
      width=w>0?(guint)w:0;height=h>0?(guint)h:0;
    }
  }
  if (caps) gst_caps_unref (caps);
  g_mutex_lock (&monitor_lock);
  if (!width || !height || id>=6) fatal=TRUE;
  else if (!g_atomic_int_get (&ownership_stopping) &&
      g_mutex_trylock (&telemetry_lock)) {
    gboolean accepted = source_epoch_ready[id] &&
        g_strcmp0 (source_lifecycle[id].state, "offline") != 0;
    if (accepted) {
      update (0,id,stages[0][id].count,GST_BUFFER_PTS(buffer),width,height,0);
      set_source_lifecycle_locked (id, "online");
    }
    g_mutex_unlock (&telemetry_lock);
  }
  g_mutex_unlock (&monitor_lock); return GST_PAD_PROBE_OK;
}
void six_live_decode_pad (GstPad *pad, guint id) {
  if (enabled) gst_pad_add_probe (pad,GST_PAD_PROBE_TYPE_BUFFER,decode_probe,GUINT_TO_POINTER(id),NULL);
}
static GstPadProbeReturn metadata_probe (GstPad *pad, GstPadProbeInfo *info, gpointer data) {
  (void) pad; guint stage=GPOINTER_TO_UINT(data); GstBuffer *buffer=GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch=buffer?gst_buffer_get_nvds_batch_meta(buffer):NULL;
  if (!enabled || g_atomic_int_get (&ownership_stopping) ||
      stage < 1 || stage > 3 || !batch) return GST_PAD_PROBE_OK;
  g_mutex_lock (&monitor_lock);
  guint frame_meta_count = 0;
  for (NvDsMetaList *l=batch->frame_meta_list;l;l=l->next) {
    if (++frame_meta_count > TELEMETRY_FRAME_META_LIMIT) { fatal=TRUE; break; }
    NvDsFrameMeta *fm=l->data; guint people=0;
    if (!fm || fm->source_id>=6 || fm->pad_index!=fm->source_id) { fatal=TRUE; continue; }
    SourceOwnerToken owner = {0};
    SourceOwnerResult owner_result = source_frame_owner (fm, &owner);
    if (owner_result != SOURCE_OWNER_OK) {
      if (owner_result == SOURCE_OWNER_MISSING ||
          owner_result == SOURCE_OWNER_UNAVAILABLE)
        g_atomic_int_inc (&source_owner_missing[fm->source_id]);
      else
        g_atomic_int_inc (&source_owner_rejected[fm->source_id]);
      continue;
    }
    guint object_meta_count = 0;
    for (NvDsMetaList *o=fm->obj_meta_list;o;o=o->next) {
      if (++object_meta_count > TELEMETRY_OBJECT_SCAN_LIMIT) { fatal=TRUE; break; }
      NvDsObjectMeta *obj=o->data;
      if (obj && obj->class_id==0) people++; else if (obj) fatal=TRUE;
    }
    if (stage == 3 && telemetry_enabled &&
        !snapshot_tracker_frame (fm, buffer, &owner))
      continue;
    update(stage,fm->source_id,fm->frame_num,fm->buf_pts,fm->source_frame_width,fm->source_frame_height,people);
  }
  g_mutex_unlock (&monitor_lock); return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn labels_probe (GstPad *pad,GstPadProbeInfo *info,gpointer data) {
  (void)pad;(void)data;GstBuffer *b=GST_PAD_PROBE_INFO_BUFFER(info);
  NvDsBatchMeta *batch=b?gst_buffer_get_nvds_batch_meta(b):NULL;
  if (!batch || !batch->frame_meta_list) return GST_PAD_PROBE_OK;
  NvDsDisplayMeta *dm=nvds_acquire_display_meta_from_pool(batch);dm->num_labels=6;
  guint64 now=(guint64)g_get_monotonic_time()*1000;
  g_mutex_lock(&monitor_lock);
  for(guint i=0;i<6;i++) {
    gboolean healthy=stages[0][i].count && stages[3][i].count && now-stages[0][i].last_ns<2000000000ULL && now-stages[3][i].last_ns<2000000000ULL;
    NvOSD_TextParams *t=&dm->text_params[i];
    t->display_text=g_strdup_printf("LIVE CAM-%02u | %s | %.1f FPS | %s",i+1,rooms[i],fps[i],healthy?"OK":"WAIT/STALE");
    t->x_offset=(i%3)*640+8;t->y_offset=(i/3)*540+8;
    t->font_params.font_name="Sans";t->font_params.font_size=14;
    t->font_params.font_color=(NvOSD_ColorParams){1,1,1,1};t->set_bg_clr=1;
    t->text_bg_clr=(NvOSD_ColorParams){0,0,0,0.8};
  }
  g_mutex_unlock(&monitor_lock);
  nvds_add_display_meta_to_frame(batch->frame_meta_list->data,dm);return GST_PAD_PROBE_OK;
}
static gboolean tick (gpointer data) {
  (void)data;
  guint64 now=(guint64)g_get_monotonic_time()*1000, rendered=0;
  gboolean stats_ok=FALSE;
  if (telemetry_enabled && live_pipeline) {
    GstState current = GST_STATE_VOID_PENDING;
    GstStateChangeReturn state_result = gst_element_get_state (live_pipeline, &current, NULL, 0);
    if (state_result != GST_STATE_CHANGE_FAILURE && current == GST_STATE_PLAYING)
      six_live_pipeline_state ("playing");
  }
  if (display_sink && GST_IS_BASE_SINK(display_sink)) {
    GstStructure *st=gst_base_sink_get_stats(GST_BASE_SINK(display_sink));
    if(st) {stats_ok=gst_structure_get_uint64(st,"rendered",&rendered);gst_structure_free(st);}
  }
  guint64 generations[6] = {0};
  /* A busy telemetry lock makes this sample unavailable. Never publish a
   * fabricated generation, and hold monitor_lock across both snapshots so
   * reconnect cannot pair old counters with a new generation. This applies
   * even when detailed ML tracker telemetry is disabled. */
  g_mutex_lock (&monitor_lock);
  if (!g_mutex_trylock (&telemetry_lock)) {
    g_mutex_unlock (&monitor_lock);
    return G_SOURCE_CONTINUE;
  }
  memcpy (generations, source_generation, sizeof generations);
  g_mutex_unlock (&telemetry_lock);
  for(guint i=0;i<6;i++) {
    if(last_tick && now>last_tick) fps[i]=(stages[3][i].count-last_counts[i])*1e9/(now-last_tick);
    last_counts[i]=stages[3][i].count;
    GString *line=g_string_new("SIX_PROGRESS ");
    g_string_append_printf(line,"{\"source_id\":%u,\"camera_id\":\"CAM-%02u\",\"monotonic_ns\":%"G_GUINT64_FORMAT",\"source_generation\":%"G_GUINT64_FORMAT",\"fatal_metadata\":%s,\"frame_owner_attached_failures\":%d,\"frame_owner_missing\":%d,\"frame_owner_rejected\":%d,\"sink_rendered\":",i,i+1,now,generations[i],fatal?"true":"false",g_atomic_int_get(&source_owner_attach_failures[i]),g_atomic_int_get(&source_owner_missing[i]),g_atomic_int_get(&source_owner_rejected[i]));
    if(stats_ok)g_string_append_printf(line,"%"G_GUINT64_FORMAT,rendered);else g_string_append(line,"null");
    g_string_append(line,",\"stages\":[");
    for(guint j=0;j<5;j++) {
      Stage *st=&stages[j][i];
      g_string_append_printf(line,"%s{\"count\":%"G_GUINT64_FORMAT",\"objects\":%"G_GUINT64_FORMAT",\"frame\":%"G_GUINT64_FORMAT",\"first_ns\":%"G_GUINT64_FORMAT",\"last_ns\":%"G_GUINT64_FORMAT",\"pts\":%"G_GUINT64_FORMAT",\"pts_regressions\":%"G_GUINT64_FORMAT",\"max_gap_ns\":%"G_GUINT64_FORMAT",\"width\":%u,\"height\":%u}",j?",":"",st->count,st->objects,st->last_frame,st->first_ns,st->last_ns,st->pts,st->pts_regressions,st->max_gap_ns,st->width,st->height);
    }
    g_string_append(line,"]}\n");g_print("%s",line->str);g_string_free(line,TRUE);
  }
  last_tick=now;fflush(stdout);g_mutex_unlock(&monitor_lock);return G_SOURCE_CONTINUE;
}
static gboolean attach_probe (GstElement *element,const gchar *name,GstPadProbeCallback cb,guint stage) {
  const gchar *stage_name = stage == 1 ? "streammux_src" :
      stage == 2 ? "pgie_src" : stage == 3 ? "nvtracker_src" : "osd_sink";
  if (!element) {
    g_printerr ("SIX_LIVE_ATTACH_ERROR stage=%s code=ELEMENT_MISSING\n", stage_name);
    return FALSE;
  }
  GstPad *pad=gst_element_get_static_pad(element,name);
  if (!pad) {
    g_printerr ("SIX_LIVE_ATTACH_ERROR stage=%s element=%s pad=%s code=PAD_MISSING\n",
        stage_name, GST_ELEMENT_NAME (element), name);
    return FALSE;
  }
  gulong id=gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_BUFFER,cb,GUINT_TO_POINTER(stage),NULL);
  gst_object_unref(pad);
  if (!id) {
    g_printerr ("SIX_LIVE_ATTACH_ERROR stage=%s element=%s pad=%s code=PROBE_ATTACH_FAILED\n",
        stage_name, GST_ELEMENT_NAME (element), name);
    return FALSE;
  }
  return TRUE;
}
static gboolean mosaic_preview_labels_enabled (void)
{
  const gchar *value = g_getenv ("V11_UI_MOSAIC_PREVIEW_ENABLED");
  return value && g_strcmp0 (value, "1") == 0;
}

gboolean six_live_attach (AppCtx *app) {
  if(!enabled)return TRUE;
  gchar owner_meta_name[] = SOURCE_OWNER_META_NAME;
  source_owner_meta_type = nvds_get_user_meta_type (owner_meta_name);
  if (source_owner_meta_type < NVDS_START_USER_META) {
    g_printerr ("SIX_LIVE_ATTACH_ERROR stage=source_lineage code=USER_META_TYPE_INVALID\n");
    return FALSE;
  }
  for (guint i = 0; i < app->pipeline.multi_src_bin.num_bins; i++) {
    NvDsSrcBin *source = &app->pipeline.multi_src_bin.sub_bins[i];
    if (!source->config || source->config->type != NV_DS_SOURCE_RTSP ||
        source->bin_id >= 6 || source->config->camera_id != source->bin_id + 1) {
      g_printerr ("SIX_LIVE_ATTACH_ERROR stage=source_lineage source_id=%u code=SOURCE_MAPPING_INVALID\n",
          source->bin_id);
      return FALSE;
    }
    source->generation_begin_hook = six_live_source_reconnecting;
    source->generation_boundary_hook = six_live_source_generation_boundary;
    GstPad *source_output = gst_element_get_static_pad (source->bin, "src");
    if (!source_output) {
      g_printerr ("SIX_LIVE_ATTACH_ERROR stage=source_lineage source_id=%u code=OUTPUT_PAD_MISSING\n",
          source->bin_id);
      return FALSE;
    }
    source->generation_probe_id = gst_pad_add_probe (source_output,
        GST_PAD_PROBE_TYPE_BUFFER, source_lineage_probe,
        GUINT_TO_POINTER (source->bin_id), NULL);
    gst_object_unref (source_output);
    if (!source->generation_probe_id) {
      g_printerr ("SIX_LIVE_ATTACH_ERROR stage=source_lineage source_id=%u code=PROBE_ATTACH_FAILED\n",
          source->bin_id);
      return FALSE;
    }
    g_print ("SIX_FRAME_OWNER event=probe_installed source_id=%u camera_id=CAM-%02u boundary=source_bin_ghost_src\n",
        source->bin_id, source->bin_id + 1);
  }
  if(!attach_probe(app->pipeline.multi_src_bin.streammux,"src",metadata_probe,1) ||
     !attach_probe(app->pipeline.common_elements.primary_gie_bin.primary_gie,"src",metadata_probe,2) ||
     !attach_probe(app->pipeline.common_elements.tracker_bin.tracker,"src",metadata_probe,3) ||
     (!mosaic_preview_labels_enabled() && !attach_probe(app->pipeline.instance_bins[0].osd_bin.nvosd,"sink",labels_probe,0)))return FALSE;
  display_sink=app->pipeline.instance_bins[0].sink_bin.sub_bins[0].sink;
  if (!display_sink || !GST_IS_BASE_SINK (display_sink)) {
    g_printerr ("SIX_LIVE_ATTACH_ERROR stage=display_sink code=NOT_GST_BASE_SINK\n");
    return FALSE;
  }
  /* The mosaic path publishes from a separate post-OSD branch and deliberately
   * configures the ordinary sink as fakesink; it does not require EGL/X11. */
  if (!mosaic_preview_labels_enabled () &&
      !g_str_has_prefix (G_OBJECT_TYPE_NAME (display_sink), "GstEgl")) {
    GstElementFactory *factory = gst_element_get_factory (display_sink);
    if (!factory || g_strcmp0 (
            gst_plugin_feature_get_name (GST_PLUGIN_FEATURE (factory)),
            "nveglglessink")) {
      g_printerr ("SIX_LIVE_ATTACH_ERROR stage=display_sink code=EGL_SINK_REQUIRED\n");
      return FALSE;
    }
  }
  live_pipeline = app->pipeline.pipeline;
  if (telemetry_enabled) {
    six_live_pipeline_state ("starting");
    g_print ("SIX_COMPONENT {\"component\":\"detector\",\"enabled\":%s,\"monotonic_ns\":%" G_GUINT64_FORMAT "}\n",
        app->config.primary_gie_config.enable ? "true" : "false", monotonic_ns ());
    g_print ("SIX_COMPONENT {\"component\":\"tracker\",\"enabled\":%s,\"monotonic_ns\":%" G_GUINT64_FORMAT "}\n",
        app->config.tracker_config.enable ? "true" : "false", monotonic_ns ());
    fflush (stdout);
    GError *thread_error = NULL;
    telemetry_stop = 0;
    telemetry_thread = g_thread_try_new ("six-telemetry-pump", telemetry_pump, NULL, &thread_error);
    if (!telemetry_thread) {
      g_printerr ("SIX_LIVE_ATTACH_ERROR stage=telemetry_pump code=THREAD_CREATE_FAILED\n");
      g_clear_error (&thread_error);
      return FALSE;
    }
  }
  timer=g_timeout_add(1000,tick,NULL);return TRUE;
}
void six_live_stop (void) {
  g_print ("SIX_SHUTDOWN event=six_live_stop_entered\n");
  g_atomic_int_set (&ownership_stopping, 1);
  if(timer){g_source_remove(timer);timer=0;}
  if(enabled)(void)tick(NULL);
  if (telemetry_enabled) {
    six_live_pipeline_state ("stopped");
    g_mutex_lock (&telemetry_lock);
    for (guint i = 0; i < 6; i++)
      set_source_lifecycle_locked (i, "offline");
    g_mutex_unlock (&telemetry_lock);
  }
  if (telemetry_thread) {
    g_atomic_int_set (&telemetry_stop, 1);
    g_print ("SIX_SHUTDOWN event=telemetry_thread_join_started\n");
    g_thread_join (telemetry_thread);
    telemetry_thread = NULL;
    g_print ("SIX_SHUTDOWN event=telemetry_thread_join_completed\n");
  }
  for (guint i = 0; telemetry_enabled && i < 6; i++) {
    SourceLifecycleSnapshot lifecycle;
    g_mutex_lock (&telemetry_lock);
    lifecycle = source_lifecycle[i];
    g_mutex_unlock (&telemetry_lock);
    if (lifecycle.revision > lifecycle_sent_revision[i]) {
      g_print ("SIX_SOURCE_STATE {\"source_id\":%u,\"camera_id\":\"CAM-%02u\","
          "\"state\":\"offline\",\"source_generation\":%" G_GUINT64_FORMAT
          ",\"monotonic_ns\":%" G_GUINT64_FORMAT "}\n",
          i, i + 1, lifecycle.generation, monotonic_ns ());
      lifecycle_sent_revision[i] = lifecycle.revision;
    }
  }
  fflush (stdout);
  enabled=FALSE;
  live_pipeline = NULL;
  telemetry_enabled = FALSE;
  for(guint i=0;i<6;i++){wipe(&users[i]);wipe(&passwords[i]);wipe(&uri_hashes[i]);}
  g_print ("SIX_SHUTDOWN event=six_live_stop_completed\n");
}
