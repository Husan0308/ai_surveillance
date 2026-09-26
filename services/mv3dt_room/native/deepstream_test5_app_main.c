/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gst/gst.h>
#include <glib.h>

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <sys/stat.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/file.h>

#include <time.h>
#include <unistd.h>
#include <errno.h>

#include "deepstream_app.h"
#include "deepstream_config_file_parser.h"
#include <cuda_runtime_api.h>
#include "nvds_version.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include <termios.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta_schema.h"

#include "deepstream_test5_app.h"
#include "bbox_correction.h"
#include "deepstream_crop_encoder.h"

/* External reference to global error capture buffer */
extern gchar *g_nvds_last_error_message;

#define MAX_DISPLAY_LEN (64)
#define MAX_TIME_STAMP_LEN (64)
#define STREAMMUX_BUFFER_POOL_SIZE (16)

#define INOTIFY_EVENT_SIZE    (sizeof (struct inotify_event))
#define INOTIFY_EVENT_BUF_LEN (1024 * ( INOTIFY_EVENT_SIZE + 16))
#define MAX_NAME_LENGTH 32752

#define IS_YAML(file) (g_str_has_suffix(file, ".yml") || g_str_has_suffix(file, ".yaml"))

/** @{
 * Macro's below and corresponding code-blocks are used to demonstrate
 * nvmsgconv + Broker Metadata manipulation possibility
 */

/**
 * IMPORTANT Note 1:
 * The code within the check for model_used == APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE
 * is applicable as sample demo code for
 * configs that use resnet PGIE model
 * with class ID's: {0, 1, 2, 3} for {CAR, BICYCLE, PERSON, ROADSIGN}
 * followed by optional Tracker + 3 X SGIEs (Vehicle-Type,Color,Make)
 * only!
 * Please comment out the code if using any other
 * custom PGIE + SGIE combinations
 * and use the code as reference to write your own
 * NvDsEventMsgMeta generation code in generate_event_msg_meta()
 * function
 */
typedef enum
{
  APP_CONFIG_ANALYTICS_MODELS_UNKNOWN = 0,
  APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE = 1,
} AppConfigAnalyticsModel;

/**
 * IMPORTANT Note 2:
 * GENERATE_DUMMY_META_EXT macro implements code
 * that assumes APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE
 * case discussed above, and generate dummy metadata
 * for other classes like Person class
 *
 * Vehicle class schema meta (NvDsVehicleObject) is filled
 * in properly from Classifier-Metadata;
 * see in-code documentation and usage of
 * schema_fill_sample_sgie_vehicle_metadata()
 */
//#define GENERATE_DUMMY_META_EXT

/** Following class-ID's
 * used for demonstration code
 * assume an ITS detection model
 * which outputs CLASS_ID=0 for Vehicle class
 * and CLASS_ID=2 for Person class
 * and SGIEs X 3 same as the sample DS config for test5-app:
 * configs/test5_config_file_src_infer_tracker_sgie.txt
 */

#define SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID  (4)
#define SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID (5)
#define SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID  (6)

#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_CAR    (0)
#ifdef GENERATE_DUMMY_META_EXT
#define RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_PERSON (2)
#endif
/** @} */

#ifdef EN_DEBUG
#define LOGD(...) printf(__VA_ARGS__)
#else
#define LOGD(...)
#endif

static TestAppCtx *testAppCtx;
GST_DEBUG_CATEGORY (NVDS_APP);


/*
 * Environment-gated frame-path diagnostics. No file is opened and no pad
 * probe is installed unless MV3DT_FRAME_AUDIT_LOG is set. The audit records
 * every frame, including zero-object frames, at each requested boundary.
 */
typedef struct {
  const gchar *stage;
} FrameAuditProbeContext;

/*
 * Per-source startup/readiness/reconnect health. This is deliberately keyed by
 * the source_id carried by NvDsFrameMeta, not by camera-label assumptions. The
 * existing DeepStream source-bin reset path is used for recovery.
 */
typedef struct {
  gboolean configured;
  gboolean seen[3];
  guint64 frames[3];
  guint64 last_frame_num[3];
  guint64 first_pts[3];
  guint64 first_ntp[3];
  gint64 first_wall_us[3];
  gint64 last_progress_us[3];
  gboolean person_seen;
  gboolean stall_logged;
  gboolean recovery_active;
  gboolean recovery_failure_logged;
  gint64 stall_started_us;
  gint64 reconnect_attempt_us;
  guint reconnect_attempts;
  guint sub_bin_index;
} SourceHealth;

static const gchar *frame_audit_camera_id (guint source_id, guint pad_index,
    gchar *mapped, gsize mapped_size);

static FILE *frame_audit_file;
static FILE *source_health_file;
static GMutex source_health_mutex;
static gboolean source_health_initialized;
static gboolean source_health_enabled;
static gchar *source_health_dir;
static gint64 source_health_started_us;
static gint64 source_health_last_status_write_us;
static gboolean source_health_last_ready;
static gboolean source_health_last_degraded;
static SourceHealth source_health[MAX_SOURCE_BINS];
static AppCtx *source_health_app_ctx;
static guint source_health_watchdog_id;

#define SOURCE_HEALTH_MUX 0
#define SOURCE_HEALTH_PGIE 1
#define SOURCE_HEALTH_TRACKER 2
#define SOURCE_HEALTH_STAGE_COUNT 3
#define SOURCE_HEALTH_STALL_SEC 5
#define SOURCE_HEALTH_STARTUP_SEC 10
#define SOURCE_HEALTH_RECOVERY_SEC 15
#define SOURCE_HEALTH_MAX_RECONNECT_ATTEMPTS 3

static gint
source_health_stage_index (const gchar *stage)
{
  if (!g_strcmp0 (stage, "nvstreammux"))
    return SOURCE_HEALTH_MUX;
  if (!g_strcmp0 (stage, "pgie"))
    return SOURCE_HEALTH_PGIE;
  if (!g_strcmp0 (stage, "tracker"))
    return SOURCE_HEALTH_TRACKER;
  return -1;
}

static const gchar *
source_health_stage_name (gint stage)
{
  switch (stage) {
    case SOURCE_HEALTH_MUX: return "nvstreammux";
    case SOURCE_HEALTH_PGIE: return "pgie";
    case SOURCE_HEALTH_TRACKER: return "tracker";
    default: return "unknown";
  }
}

static gboolean
source_health_open (void)
{
  const gchar *directory = g_getenv ("MV3DT_SOURCE_HEALTH_DIR");
  gchar *events_path;

  if (source_health_initialized)
    return source_health_enabled;

  g_mutex_init (&source_health_mutex);
  source_health_initialized = TRUE;
  if (!directory || !*directory)
    return FALSE;

  source_health_dir = g_strdup (directory);
  g_mkdir_with_parents (source_health_dir, 0755);
  events_path = g_build_filename (source_health_dir, "source_health.jsonl", NULL);
  source_health_file = fopen (events_path, "w");
  g_free (events_path);
  if (!source_health_file) {
    g_printerr ("source health: failed to open %s: %s\n",
        directory, g_strerror (errno));
    return FALSE;
  }
  setvbuf (source_health_file, NULL, _IOLBF, 0);
  source_health_enabled = TRUE;
  source_health_started_us = g_get_monotonic_time ();
  return TRUE;
}

static void
source_health_event_locked (const gchar *event, guint source_id,
    const gchar *state, gint64 recovery_us)
{
  gchar mapped[128];

  if (!source_health_enabled || !source_health_file)
    return;
  frame_audit_camera_id (source_id, source_id, mapped, sizeof (mapped));
  fprintf (source_health_file,
      "{\"event\":\"%s\",\"epoch_ms\":%" G_GINT64_FORMAT
      ",\"source_id\":%u,\"mapped_camera_id\":\"%s\"",
      event, g_get_real_time () / 1000, source_id, mapped);
  if (state)
    fprintf (source_health_file, ",\"source_state\":\"%s\"", state);
  if (recovery_us >= 0)
    fprintf (source_health_file, ",\"recovery_duration_ms\":%.3f",
        (double) recovery_us / 1000.0);
  fputs ("}\n", source_health_file);
  fflush (source_health_file);
  g_print ("SOURCE_HEALTH %s source_id=%u camera=%s%s%s%s\n", event,
      source_id, mapped, state ? " state=" : "", state ? state : "",
      recovery_us >= 0 ? " recovery_ms=" : "");
  if (recovery_us >= 0)
    g_print ("%.3f", (double) recovery_us / 1000.0);
  if (recovery_us >= 0)
    g_print ("\n");
}

static gboolean
source_health_ready_locked (gint64 now_us)
{
  guint i;

  for (i = 0; i < MAX_SOURCE_BINS; i++) {
    SourceHealth *health = &source_health[i];
    if (!health->configured)
      continue;
    if (!health->seen[SOURCE_HEALTH_MUX] ||
        !health->seen[SOURCE_HEALTH_PGIE] ||
        !health->seen[SOURCE_HEALTH_TRACKER])
      return FALSE;
    if (now_us - health->last_progress_us[SOURCE_HEALTH_MUX] >
            SOURCE_HEALTH_STALL_SEC * G_USEC_PER_SEC ||
        now_us - health->last_progress_us[SOURCE_HEALTH_PGIE] >
            SOURCE_HEALTH_STALL_SEC * G_USEC_PER_SEC ||
        now_us - health->last_progress_us[SOURCE_HEALTH_TRACKER] >
            SOURCE_HEALTH_STALL_SEC * G_USEC_PER_SEC)
      return FALSE;
  }
  return TRUE;
}

static void
source_health_write_readiness_locked (gint64 now_us, gboolean force)
{
  gboolean ready, degraded = FALSE;
  gchar *path, *tmp_path;
  FILE *out;
  guint i;

  if (!source_health_enabled || !source_health_dir)
    return;
  if (!force && now_us - source_health_last_status_write_us < G_USEC_PER_SEC)
    return;
  source_health_last_status_write_us = now_us;
  ready = source_health_ready_locked (now_us);
  for (i = 0; i < MAX_SOURCE_BINS; i++) {
    if (source_health[i].configured && source_health[i].recovery_active)
      degraded = TRUE;
  }
  path = g_build_filename (source_health_dir, "readiness.json", NULL);
  tmp_path = g_strdup_printf ("%s.tmp", path);
  out = fopen (tmp_path, "w");
  if (!out) {
    g_free (path);
    g_free (tmp_path);
    return;
  }
  fprintf (out, "{\"status\":\"%s\",\"ready\":%s,\"updated_epoch_ms\":%" G_GINT64_FORMAT ",\"sources\":[",
      ready ? "ready" : (degraded ? "degraded" : "not_ready"),
      ready ? "true" : "false", g_get_real_time () / 1000);
  {
    gboolean first = TRUE;
    for (i = 0; i < MAX_SOURCE_BINS; i++) {
      SourceHealth *health = &source_health[i];
      if (!health->configured)
        continue;
      if (!first) fputc (',', out);
      first = FALSE;
      fprintf (out, "{\"source_id\":%u,\"mux_frames\":%" G_GUINT64_FORMAT
          ",\"pgie_frames\":%" G_GUINT64_FORMAT
          ",\"tracker_frames\":%" G_GUINT64_FORMAT
          ",\"mux_seen\":%s,\"pgie_seen\":%s,\"tracker_seen\":%s"
          ",\"last_mux_age_ms\":%.3f,\"last_pgie_age_ms\":%.3f,\"last_tracker_age_ms\":%.3f"
          ",\"person_seen\":%s,\"reconnect_attempts\":%u}",
          i, health->frames[SOURCE_HEALTH_MUX], health->frames[SOURCE_HEALTH_PGIE],
          health->frames[SOURCE_HEALTH_TRACKER],
          health->seen[SOURCE_HEALTH_MUX] ? "true" : "false",
          health->seen[SOURCE_HEALTH_PGIE] ? "true" : "false",
          health->seen[SOURCE_HEALTH_TRACKER] ? "true" : "false",
          health->seen[SOURCE_HEALTH_MUX] ? (double)(now_us - health->last_progress_us[SOURCE_HEALTH_MUX]) / 1000.0 : -1.0,
          health->seen[SOURCE_HEALTH_PGIE] ? (double)(now_us - health->last_progress_us[SOURCE_HEALTH_PGIE]) / 1000.0 : -1.0,
          health->seen[SOURCE_HEALTH_TRACKER] ? (double)(now_us - health->last_progress_us[SOURCE_HEALTH_TRACKER]) / 1000.0 : -1.0,
          health->person_seen ? "true" : "false", health->reconnect_attempts);
    }
  }
  fputs ("]}\n", out);
  fclose (out);
  rename (tmp_path, path);
  if (ready != source_health_last_ready) {
    source_health_last_ready = ready;
    if (ready)
      source_health_event_locked ("startup_ready", 0, "PLAYING", -1);
  }
  if (degraded != source_health_last_degraded)
    source_health_last_degraded = degraded;
  g_free (path);
  g_free (tmp_path);
}

static void
source_health_configure (AppCtx *app_ctx)
{
  guint i;
  if (!source_health_open () || !app_ctx)
    return;
  source_health_app_ctx = app_ctx;
  g_mutex_lock (&source_health_mutex);
  memset (source_health, 0, sizeof (source_health));
  for (i = 0; i < app_ctx->pipeline.multi_src_bin.num_bins; i++) {
    guint source_id = app_ctx->pipeline.multi_src_bin.sub_bins[i].source_id;
    if (source_id < MAX_SOURCE_BINS) {
      source_health[source_id].configured = TRUE;
      source_health[source_id].sub_bin_index = i;
    }
  }
  source_health_write_readiness_locked (g_get_monotonic_time (), TRUE);
  g_mutex_unlock (&source_health_mutex);
}

static void
source_health_update (const gchar *stage, NvDsBatchMeta *batch_meta)
{
  NvDsMetaList *l_frame;
  gint stage_index = source_health_stage_index (stage);
  gint64 now_us = g_get_monotonic_time ();

  if (stage_index < 0 || !source_health_enabled || !batch_meta)
    return;
  g_mutex_lock (&source_health_mutex);
  for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    SourceHealth *health;
    NvDsMetaList *l_obj;
    if (!frame_meta || frame_meta->source_id >= MAX_SOURCE_BINS)
      continue;
    health = &source_health[frame_meta->source_id];
    if (!health->configured)
      continue;
    health->frames[stage_index]++;
    health->last_frame_num[stage_index] = frame_meta->frame_num;
    health->last_progress_us[stage_index] = now_us;
    if (!health->seen[stage_index]) {
      health->seen[stage_index] = TRUE;
      health->first_pts[stage_index] = frame_meta->buf_pts;
      health->first_ntp[stage_index] = frame_meta->ntp_timestamp;
      health->first_wall_us[stage_index] = g_get_real_time ();
      source_health_event_locked ("first_frame", frame_meta->source_id,
          source_health_stage_name (stage_index), -1);
    }
    if (stage_index == SOURCE_HEALTH_PGIE) {
      for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
        NvDsObjectMeta *obj = (NvDsObjectMeta *) l_obj->data;
        if (obj && (!g_ascii_strcasecmp (obj->obj_label, "person") ||
                obj->class_id == 0)) {
          if (!health->person_seen)
            source_health_event_locked ("first_person_detection",
                frame_meta->source_id, source_health_stage_name (stage_index), -1);
          health->person_seen = TRUE;
          break;
        }
      }
    }
    if (stage_index == SOURCE_HEALTH_MUX && health->recovery_active) {
      gint64 recovery_us = now_us - health->stall_started_us;
      health->recovery_active = FALSE;
      health->stall_logged = FALSE;
      health->recovery_failure_logged = FALSE;
      source_health_event_locked ("reconnect_success", frame_meta->source_id,
          "PLAYING", recovery_us);
    }
  }
  source_health_write_readiness_locked (now_us, FALSE);
  g_mutex_unlock (&source_health_mutex);
}

static gboolean
source_health_watchdog_cb (gpointer data)
{
  AppCtx *app_ctx = (AppCtx *) data;
  gint64 now_us = g_get_monotonic_time ();
  guint i;

  if (!source_health_enabled || !app_ctx)
    return G_SOURCE_CONTINUE;
  g_mutex_lock (&source_health_mutex);
  for (i = 0; i < MAX_SOURCE_BINS; i++) {
    SourceHealth *health = &source_health[i];
    NvDsSrcBin *src_bin;
    gboolean mux_stalled;
    if (!health->configured || health->sub_bin_index >=
        app_ctx->pipeline.multi_src_bin.num_bins)
      continue;
    src_bin = &app_ctx->pipeline.multi_src_bin.sub_bins[health->sub_bin_index];
    mux_stalled = !health->seen[SOURCE_HEALTH_MUX] ?
        (now_us - source_health_started_us >= SOURCE_HEALTH_STARTUP_SEC * G_USEC_PER_SEC) :
        (now_us - health->last_progress_us[SOURCE_HEALTH_MUX] >= SOURCE_HEALTH_STALL_SEC * G_USEC_PER_SEC);
    if (mux_stalled && !health->stall_logged && !src_bin->reconfiguring) {
      health->stall_logged = TRUE;
      health->recovery_active = TRUE;
      health->recovery_failure_logged = FALSE;
      health->stall_started_us = now_us;
      health->reconnect_attempts = 0;
      source_health_event_locked ("source_stall_detected", i, "DEGRADED", -1);
    }
    if (health->recovery_active && !src_bin->reconfiguring &&
        (health->reconnect_attempt_us == 0 ||
         now_us - health->reconnect_attempt_us >= G_USEC_PER_SEC)) {
      if (health->reconnect_attempts < SOURCE_HEALTH_MAX_RECONNECT_ATTEMPTS) {
        health->reconnect_attempts++;
        health->reconnect_attempt_us = now_us;
        source_health_event_locked ("reconnect_attempt", i, "RECONNECTING", -1);
        reset_source_pipeline (src_bin);
      } else if (!health->recovery_failure_logged &&
          now_us - health->stall_started_us >= SOURCE_HEALTH_RECOVERY_SEC * G_USEC_PER_SEC) {
        health->recovery_failure_logged = TRUE;
        source_health_event_locked ("reconnect_failure", i, "DEGRADED", -1);
      }
    }
  }
  source_health_write_readiness_locked (now_us, FALSE);
  g_mutex_unlock (&source_health_mutex);
  return G_SOURCE_CONTINUE;
}

static void
source_health_close (void)
{
  if (source_health_watchdog_id) {
    g_source_remove (source_health_watchdog_id);
    source_health_watchdog_id = 0;
  }
  if (source_health_file) {
    fclose (source_health_file);
    source_health_file = NULL;
  }
  g_clear_pointer (&source_health_dir, g_free);
}

/* Latest-only V11 shared-memory preview. It is enabled only when the
 * production UI transport is requested; analytics and identity data do not
 * flow through this path. The writer publishes the post-OSD/output buffer
 * format expected by PreviewFrameReader. */
typedef struct {
  gboolean enabled;
  gchar *path;
  gint fd;
  gpointer map;
  gsize map_size;
  NvBufSurface *target;
  guint width;
  guint height;
  guint stride;
  guint payload_size;
  guint64 sequence;
  gdouble fps_ema;
  gint64 last_publish_us;
  GMutex mutex;
} PreviewWriter;

static PreviewWriter preview_writers[2];
static gboolean preview_initialized;
static GMutex preview_transform_mutex;

#define PREVIEW_MAGIC "V11UI01\0"
#define PREVIEW_VERSION 1
#define PREVIEW_HEADER_SIZE 64
#define PREVIEW_WIDTH 1920
#define PREVIEW_HEIGHT 1080
#define PREVIEW_STRIDE (PREVIEW_WIDTH * 4)
#define PREVIEW_PAYLOAD_SIZE (PREVIEW_STRIDE * PREVIEW_HEIGHT)

static gint
preview_camera_index (guint source_id, guint pad_index)
{
  gchar mapped[128];
  frame_audit_camera_id (source_id, pad_index, mapped, sizeof (mapped));
  if (!g_strcmp0 (mapped, "CAM-01")) return 0;
  if (!g_strcmp0 (mapped, "CAM-04")) return 1;
  return -1;
}

static gboolean
preview_write_header (PreviewWriter *writer, guint object_count, guint fps_milli,
    guint64 timestamp_ns)
{
  guint8 header[PREVIEW_HEADER_SIZE] = {0};
  guint32 version = PREVIEW_VERSION;
  guint64 sequence = writer->sequence;
  guint32 values[6] = {writer->width, writer->height, writer->stride,
      writer->payload_size, object_count, fps_milli};
  memcpy (header, PREVIEW_MAGIC, 8);
  memcpy (header + 8, &version, sizeof (version));
  memcpy (header + 12, &sequence, sizeof (sequence));
  memcpy (header + 20, &timestamp_ns, sizeof (timestamp_ns));
  memcpy (header + 28, values, sizeof (values));
  return pwrite (writer->fd, header, PREVIEW_HEADER_SIZE, 0) == PREVIEW_HEADER_SIZE;
}

static gboolean
preview_open_writer (PreviewWriter *writer, const gchar *directory,
    const gchar *camera_slug)
{
  gchar *path;

  memset (writer, 0, sizeof (*writer));
  writer->fd = -1;
  writer->width = PREVIEW_WIDTH;
  writer->height = PREVIEW_HEIGHT;
  writer->stride = PREVIEW_STRIDE;
  writer->payload_size = PREVIEW_PAYLOAD_SIZE;
  g_mutex_init (&writer->mutex);
  path = g_strdup_printf ("%s/v11_ui_preview_%s_v1.bin", directory, camera_slug);
  writer->path = path;
  writer->fd = open (path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (writer->fd < 0) {
    g_printerr ("preview: open failed camera=%s path=%s error=%s\n", camera_slug, path, g_strerror (errno));
    return FALSE;
  }
  /* The DeepStream container runs as root while the desktop UI runs as the
   * logged-in user; shared-memory preview files are intentionally readable by
   * that UI user, but never writable by it. */
  fchmod (writer->fd, 0644);
  writer->map_size = PREVIEW_HEADER_SIZE + PREVIEW_PAYLOAD_SIZE;
  if (ftruncate (writer->fd, writer->map_size) != 0) {
    g_printerr ("preview: ftruncate failed path=%s error=%s\n", path, g_strerror (errno));
    return FALSE;
  }
  writer->map = mmap (NULL, writer->map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
      writer->fd, 0);
  if (writer->map == MAP_FAILED) {
    g_printerr ("preview: mmap failed path=%s error=%s\n", path, g_strerror (errno));
    writer->map = NULL;
    return FALSE;
  }
  memset (writer->map, 0, writer->map_size);
  writer->enabled = TRUE;
  preview_write_header (writer, 0, 0, 0);
  return TRUE;
}

static gboolean
preview_open (void)
{
  const gchar *directory = g_getenv ("MV3DT_UI_PREVIEW_DIR");
  NvBufSurfaceCreateParams params;

  if (preview_initialized)
    return preview_writers[0].enabled || preview_writers[1].enabled;
  preview_initialized = TRUE;
  g_mutex_init (&preview_transform_mutex);
  if (!directory || !*directory)
    return FALSE;
  g_mkdir_with_parents (directory, 0755);
  preview_open_writer (&preview_writers[0], directory, "cam01");
  preview_open_writer (&preview_writers[1], directory, "cam04");
  memset (&params, 0, sizeof (params));
  params.gpuId = 0;
  params.width = PREVIEW_WIDTH;
  params.height = PREVIEW_HEIGHT;
  params.colorFormat = NVBUF_COLOR_FORMAT_BGRA;
  params.layout = NVBUF_LAYOUT_PITCH;
  params.memType = NVBUF_MEM_CUDA_UNIFIED;
  params.isContiguous = TRUE;
  for (guint i = 0; i < 2; i++) {
    if (!preview_writers[i].enabled ||
        NvBufSurfaceCreate (&preview_writers[i].target, 1, &params) != 0) {
      g_printerr ("preview: target allocation failed camera_index=%u\n", i);
      preview_writers[i].enabled = FALSE;
      continue;
    }
    g_print ("preview: enabled camera_index=%u path=%s\n", i, preview_writers[i].path);
  }
  return preview_writers[0].enabled || preview_writers[1].enabled;
}

static void
preview_publish (NvBufSurface *surface, NvDsFrameMeta *frame_meta,
    guint object_count)
{
  NvBufSurface src;
  NvBufSurfTransformConfigParams config;
  NvBufSurfTransformParams transform;
  NvBufSurfTransformRect src_rect, dst_rect;
  NvBufSurfaceParams *src_params;
  NvBufSurfaceParams *dst_params;
  PreviewWriter *writer;
  gint camera_index;
  gint64 now_us;
  guint8 *dst;
  guint y;

  if (!surface || !frame_meta)
    return;
  camera_index = preview_camera_index (frame_meta->source_id, frame_meta->pad_index);
  if (frame_meta->batch_id >= surface->batchSize) {
    static gboolean logged_invalid_batch[2];
    if (camera_index >= 0 && camera_index < 2 && !logged_invalid_batch[camera_index]) {
      g_printerr ("preview: invalid batch index camera_index=%d source=%u pad=%u batch_id=%u batch_size=%u\n",
          camera_index, frame_meta->source_id, frame_meta->pad_index, frame_meta->batch_id, surface->batchSize);
      logged_invalid_batch[camera_index] = TRUE;
    }
    return;
  }
  if (camera_index < 0 || !preview_writers[camera_index].enabled) {
    static gboolean logged_unmapped[2];
    if (camera_index >= 0 && !logged_unmapped[camera_index]) {
      g_printerr ("preview: writer disabled camera_index=%d source=%u pad=%u\n", camera_index, frame_meta->source_id, frame_meta->pad_index);
      logged_unmapped[camera_index] = TRUE;
    }
    return;
  }
  writer = &preview_writers[camera_index];
  src = *surface;
  src.numFilled = 1;
  src.batchSize = 1;
  src.surfaceList = &surface->surfaceList[frame_meta->batch_id];
  src_params = &src.surfaceList[0];
  dst_params = &writer->target->surfaceList[0];
  memset (&config, 0, sizeof (config));
  config.compute_mode = NvBufSurfTransformCompute_Default;
  config.gpu_id = 0;
  memset (&transform, 0, sizeof (transform));
  memset (&src_rect, 0, sizeof (src_rect));
  memset (&dst_rect, 0, sizeof (dst_rect));
  src_rect.width = src_params->width;
  src_rect.height = src_params->height;
  dst_rect.width = PREVIEW_WIDTH;
  dst_rect.height = PREVIEW_HEIGHT;
  transform.src_rect = &src_rect;
  transform.dst_rect = &dst_rect;
  transform.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
  transform.transform_filter = NvBufSurfTransformInter_Default;
  g_mutex_lock (&preview_transform_mutex);
  {
    NvBufSurfTransform_Error session_error = NvBufSurfTransformSetSessionParams (&config);
    NvBufSurfTransform_Error transform_error = session_error == NvBufSurfTransformError_Success ?
        NvBufSurfTransform (&src, writer->target, &transform) : session_error;
    if (transform_error != NvBufSurfTransformError_Success) {
      static gboolean logged_transform[2];
      if (!logged_transform[camera_index]) {
        g_printerr ("preview: transform failed camera_index=%d source=%u pad=%u batch_id=%u batch_size=%u src=%ux%u format=%d error=%d\n",
            camera_index, frame_meta->source_id, frame_meta->pad_index, frame_meta->batch_id,
            surface->batchSize, src_params->width, src_params->height, src_params->colorFormat, transform_error);
        logged_transform[camera_index] = TRUE;
      }
      g_mutex_unlock (&preview_transform_mutex);
      return;
    }
  }
  if (NvBufSurfaceMap (writer->target, 0, 0, NVBUF_MAP_READ) != 0) {
    g_mutex_unlock (&preview_transform_mutex);
    return;
  }
  NvBufSurfaceSyncForCpu (writer->target, 0, 0);
  dst = (guint8 *) dst_params->mappedAddr.addr[0];
  now_us = g_get_monotonic_time ();
  g_mutex_lock (&writer->mutex);
  if (writer->last_publish_us > 0) {
    gdouble instant = 1.0e6 / (gdouble) MAX (1, now_us - writer->last_publish_us);
    writer->fps_ema = writer->fps_ema > 0.0 ? writer->fps_ema * 0.85 + instant * 0.15 : instant;
  }
  writer->last_publish_us = now_us;
  writer->sequence++;
  flock (writer->fd, LOCK_EX);
  for (y = 0; y < PREVIEW_HEIGHT; y++)
    memcpy ((guint8 *) writer->map + PREVIEW_HEADER_SIZE + y * writer->stride,
        dst + y * dst_params->pitch, writer->stride);
  preview_write_header (writer, object_count,
      (guint) MAX (0, (gint) (writer->fps_ema * 1000.0)),
      (guint64) g_get_monotonic_time () * 1000);
  flock (writer->fd, LOCK_UN);
  g_mutex_unlock (&writer->mutex);
  NvBufSurfaceUnMap (writer->target, 0, 0);
  g_mutex_unlock (&preview_transform_mutex);
}

static GstPadProbeReturn
preview_pad_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  GstBuffer *buffer;
  GstMapInfo map;
  NvBufSurface *surface;
  NvDsBatchMeta *batch_meta;
  NvDsMetaList *l_frame;
  (void) pad; (void) user_data;

  if (!info || !(buffer = GST_PAD_PROBE_INFO_BUFFER (info)))
    return GST_PAD_PROBE_OK;
  memset (&map, 0, sizeof (map));
  if (!gst_buffer_map (buffer, &map, GST_MAP_READ))
    return GST_PAD_PROBE_OK;
  surface = (NvBufSurface *) map.data;
  batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  if (surface && batch_meta) {
    for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      if (frame_meta)
        preview_publish (surface, frame_meta, frame_meta->num_obj_meta);
    }
  }
  gst_buffer_unmap (buffer, &map);
  return GST_PAD_PROBE_OK;
}

static void
preview_attach (AppCtx *app_ctx)
{
  GstElement *element;
  GstPad *pad;

  if (!preview_open () || !app_ctx)
    return;
  /* The configured display sink is source-id=0 and therefore cannot carry
   * CAM-04.  Attach at the existing tracker output so the V11 preview sees
   * every source frame in the production batch before the single-camera EGL
   * sink filter. */
  element = app_ctx->pipeline.common_elements.tracker_bin.tracker;
  pad = element ? gst_element_get_static_pad (element, "src") : NULL;
  if (!pad) {
    element = app_ctx->pipeline.instance_bins[0].osd_bin.nvosd;
    if (element)
      pad = gst_element_get_static_pad (element, "src");
  }
  if (!pad) {
    element = app_ctx->pipeline.instance_bins[0].sink_bin.bin;
    pad = gst_element_get_static_pad (element, "sink");
  }
  if (!pad)
    return;
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, preview_pad_probe, NULL, NULL);
  gst_object_unref (pad);
}

static void
preview_close (void)
{
  for (guint i = 0; i < 2; i++) {
    PreviewWriter *writer = &preview_writers[i];
    if (writer->target) {
      NvBufSurfaceDestroy (writer->target);
      writer->target = NULL;
    }
    if (writer->map) {
      munmap (writer->map, writer->map_size);
      writer->map = NULL;
    }
    if (writer->fd >= 0) {
      close (writer->fd);
      writer->fd = -1;
    }
    if (writer->path) {
      unlink (writer->path);
      g_free (writer->path);
      writer->path = NULL;
    }
  }
}

static FILE *frame_audit_file;
static GMutex frame_audit_mutex;
static gboolean frame_audit_initialized;

static const gchar *
frame_audit_camera_id (guint source_id, guint pad_index,
    gchar *mapped, gsize mapped_size)
{
  const gchar *mapping = g_getenv ("MV3DT_AUDIT_CAMERA_MAP");
  gchar pair_key[64];
  gchar source_key[64];
  gchar **entries;
  guint i;

  g_snprintf (pair_key, sizeof (pair_key), "%u/%u", source_id, pad_index);
  g_snprintf (source_key, sizeof (source_key), "source:%u", source_id);
  g_strlcpy (mapped, "UNMAPPED", mapped_size);
  if (!mapping || !*mapping)
    return mapped;

  entries = g_strsplit_set (mapping, ";,", -1);
  for (i = 0; entries && entries[i]; i++) {
    gchar **kv = g_strsplit (entries[i], "=", 2);
    if (kv[0] && kv[1] &&
        (!g_strcmp0 (kv[0], pair_key) || !g_strcmp0 (kv[0], source_key))) {
      g_strlcpy (mapped, kv[1], mapped_size);
      g_strfreev (kv);
      break;
    }
    g_strfreev (kv);
  }
  g_strfreev (entries);
  return mapped;
}

static gboolean
frame_audit_open (void)
{
  const gchar *path;
  gchar *directory;

  if (frame_audit_initialized)
    return frame_audit_file != NULL;
  g_mutex_init (&frame_audit_mutex);
  frame_audit_initialized = TRUE;
  path = g_getenv ("MV3DT_FRAME_AUDIT_LOG");
  if (!path || !*path)
    return FALSE;

  directory = g_path_get_dirname (path);
  g_mkdir_with_parents (directory, 0755);
  g_free (directory);
  frame_audit_file = fopen (path, "w");
  if (!frame_audit_file) {
    g_printerr ("frame audit: failed to open %s: %s\n", path, g_strerror (errno));
    return FALSE;
  }
  setvbuf (frame_audit_file, NULL, _IOLBF, 0);
  return TRUE;
}

static void
frame_audit_log_batch (const gchar *stage, NvDsBatchMeta *batch_meta)
{
  NvDsMetaList *l_frame;

  if (!frame_audit_open () || !batch_meta)
    return;

  g_mutex_lock (&frame_audit_mutex);
  for (l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    NvDsMetaList *l_obj;
    gchar mapped[128];
    gboolean first_object = TRUE;

    if (!frame_meta)
      continue;
    frame_audit_camera_id (frame_meta->source_id, frame_meta->pad_index,
        mapped, sizeof (mapped));
    fprintf (frame_audit_file,
        "{\"record\":\"frame\",\"stage\":\"%s\","
        "\"frame_num\":%d,\"source_id\":%u,\"pad_index\":%u,"
        "\"batch_id\":%u,\"pts\":\"%" G_GUINT64_FORMAT "\","
        "\"ntp_timestamp\":\"%" G_GUINT64_FORMAT "\","
        "\"bInferDone\":%s,\"num_obj_meta\":%u,"
        "\"mapped_camera_id\":\"%s\",\"objects\":[",
        stage, frame_meta->frame_num, frame_meta->source_id,
        frame_meta->pad_index, frame_meta->batch_id, frame_meta->buf_pts,
        frame_meta->ntp_timestamp, frame_meta->bInferDone ? "true" : "false",
        frame_meta->num_obj_meta, mapped);
    for (l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
      if (!obj_meta)
        continue;
      if (!first_object)
        fputc (',', frame_audit_file);
      first_object = FALSE;
      fprintf (frame_audit_file,
          "{\"object_id\":\"%" G_GUINT64_FORMAT "\","
          "\"class_id\":%d,\"label\":\"%s\","
          "\"confidence\":%.6f,\"tracker_confidence\":%.6f}",
          obj_meta->object_id, obj_meta->class_id, obj_meta->obj_label,
          obj_meta->confidence, obj_meta->tracker_confidence);
    }
    fputs ("]}\n", frame_audit_file);
  }
  fflush (frame_audit_file);
  g_mutex_unlock (&frame_audit_mutex);
}

static GstPadProbeReturn
frame_audit_pad_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  FrameAuditProbeContext *context = (FrameAuditProbeContext *) user_data;
  GstBuffer *buffer;
  NvDsBatchMeta *batch_meta;

  (void) pad;
  if (!info || !(buffer = GST_PAD_PROBE_INFO_BUFFER (info)))
    return GST_PAD_PROBE_OK;
  batch_meta = gst_buffer_get_nvds_batch_meta (buffer);
  source_health_update (context->stage, batch_meta);
  frame_audit_log_batch (context->stage, batch_meta);
  return GST_PAD_PROBE_OK;
}

static void
frame_audit_attach_pad (GstElement *element, const gchar *stage)
{
  GstPad *pad;
  FrameAuditProbeContext *context;

  if (!element)
    return;
  if (g_getenv ("MV3DT_FRAME_AUDIT_LOG") && !frame_audit_open ())
    return;
  pad = gst_element_get_static_pad (element, "src");
  if (!pad) {
    g_printerr ("frame audit: no src pad for %s\n", stage);
    return;
  }
  context = g_new0 (FrameAuditProbeContext, 1);
  context->stage = stage;
  gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, frame_audit_pad_probe,
      context, (GDestroyNotify) g_free);
  gst_object_unref (pad);
}

/** @{ imported from deepstream-app as is */


#define MAX_INSTANCES 128
#define APP_TITLE "DeepStreamTest5App"

#define DEFAULT_X_WINDOW_WIDTH 1920
#define DEFAULT_X_WINDOW_HEIGHT 1080

AppCtx *appCtx[MAX_INSTANCES];
static guint cintr = FALSE;
static GMainLoop *main_loop = NULL;
static gchar **cfg_files = NULL;
static gchar **input_files = NULL;
static gchar **override_cfg_file = NULL;
static gboolean playback_utc = FALSE;
static gboolean print_version = FALSE;
static gboolean show_bbox_text = FALSE;
static gboolean force_tcp = TRUE;
static gboolean print_dependencies_version = FALSE;
static gboolean quit = FALSE;
static gint return_value = 0;
static guint num_instances;
static guint num_input_files;
static GMutex fps_lock;
static gdouble fps[MAX_SOURCE_BINS];
static gdouble fps_avg[MAX_SOURCE_BINS];

static Display *display = NULL;
static Window windows[MAX_INSTANCES] = { 0 };

static GThread *x_event_thread = NULL;
static GMutex disp_lock;

static guint rrow, rcol, rcfg;
static gboolean rrowsel = FALSE, selecting = FALSE;
static AppConfigAnalyticsModel model_used = APP_CONFIG_ANALYTICS_MODELS_UNKNOWN;

static struct timeval ota_request_time;
static struct timeval ota_completion_time;

typedef struct _OTAInfo
{
  AppCtx *appCtx;
  gchar *override_cfg_file;
} OTAInfo;

/** @} imported from deepstream-app as is */
GOptionEntry entries[] = {
  {"version", 'v', 0, G_OPTION_ARG_NONE, &print_version,
      "Print DeepStreamSDK version", NULL}
  ,
  {"tiledtext", 't', 0, G_OPTION_ARG_NONE, &show_bbox_text,
      "Display Bounding box labels in tiled mode", NULL}
  ,
  {"version-all", 0, 0, G_OPTION_ARG_NONE, &print_dependencies_version,
      "Print DeepStreamSDK and dependencies version", NULL}
  ,
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME_ARRAY, &cfg_files,
      "Set the config file", NULL}
  ,
  {"override-cfg-file", 'o', 0, G_OPTION_ARG_FILENAME_ARRAY, &override_cfg_file,
      "Set the override config file, used for on-the-fly model update feature",
        NULL}
  ,
  {"input-file", 'i', 0, G_OPTION_ARG_FILENAME_ARRAY, &input_files,
      "Set the input file", NULL}
  ,
  {"playback-utc", 'p', 0, G_OPTION_ARG_INT, &playback_utc,
        "Playback utc; default=false (base UTC from file-URL or RTCP Sender Report) =true (base UTC from file/rtsp URL)",
      NULL}
  ,
  {"pgie-model-used", 'm', 0, G_OPTION_ARG_INT, &model_used,
        "PGIE Model used; {0 - Unknown [DEFAULT]}, {1: Resnet 4-class [Car, Bicycle, Person, Roadsign]}",
      NULL}
  ,
  {"no-force-tcp", 0, G_OPTION_FLAG_REVERSE, G_OPTION_ARG_NONE, &force_tcp,
      "Do not force TCP for RTP transport", NULL}
  ,
  {NULL}
  ,
};

static void nanoseconds_to_rfc3339(int64_t nanoseconds, char *output, size_t output_size);

/**
 * @brief  Fill NvDsVehicleObject with the NvDsClassifierMetaList
 *         information in NvDsObjectMeta
 *         NOTE: This function assumes the test-application is
 *         run with 3 X SGIEs sample config:
 *         test5_config_file_src_infer_tracker_sgie.txt
 *         or an equivalent config
 *         NOTE: If user is adding custom SGIEs, make sure to
 *         edit this function implementation
 * @param  obj_params [IN] The NvDsObjectMeta as detected and kept
 *         in NvDsBatchMeta->NvDsFrameMeta(List)->NvDsObjectMeta(List)
 * @param  obj [IN/OUT] The NvDSMeta-Schema defined Vehicle metadata
 *         structure
 */
static void schema_fill_sample_sgie_vehicle_metadata (NvDsObjectMeta *
    obj_params, NvDsVehicleObject * obj);

/**
 * @brief  Performs model update OTA operation
 *         Sets "model-engine-file" configuration parameter
 *         on infer plugin to initiate model switch OTA process
 * @param  ota_appCtx [IN] App context pointer
 */
void apply_ota (AppCtx * ota_appCtx);

/**
 * @brief  Thread which handles the model-update OTA functionlity
 *         1) Adds watch on the changes made in the provided ota-override-file,
 *            if changes are detected, validate the model-update change request,
 *            intiate model-update OTA process
 *         2) Frame drops / frames without inference should NOT be detected in
 *            this on-the-fly model update process
 *         3) In case of model update OTA fails, error message will be printed
 *            on the console and pipeline continues to run with older
 *            model configuration
 * @param  gpointer [IN] Pointer to OTAInfo structure
 * @param  gpointer [OUT] Returns NULL in case of thread exits
 */
gpointer ota_handler_thread (gpointer data);

static void
generate_ts_rfc3339 (char *buf, int buf_size)
{
  time_t tloc;
  struct tm tm_log;
  struct timespec ts;
  char strmsec[6];              //.nnnZ\0

  clock_gettime (CLOCK_REALTIME, &ts);
  memcpy (&tloc, (void *) (&ts.tv_sec), sizeof (time_t));
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  int ms = ts.tv_nsec / 1000000;
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
}

static GstClockTime
generate_ts_rfc3339_from_ts (char *buf, int buf_size, GstClockTime ts,
    gchar * src_uri, gint stream_id)
{
  time_t tloc;
  struct tm tm_log;
  char strmsec[6];              //.nnnZ\0
  int ms;

  GstClockTime ts_generated;

  if ((playback_utc
      && (appCtx[0]->config.source_attr_all_config.type !=
          NV_DS_SOURCE_IPC)) || (ts == 0) ) {
    if (testAppCtx->streams[stream_id].meta_number == 0) {
      testAppCtx->streams[stream_id].timespec_first_frame =
          extract_utc_from_uri (src_uri);
      memcpy (&tloc,
          (void *) (&testAppCtx->streams[stream_id].timespec_first_frame.
              tv_sec), sizeof (time_t));
      ms = testAppCtx->streams[stream_id].timespec_first_frame.tv_nsec /
          1000000;
      testAppCtx->streams[stream_id].gst_ts_first_frame = ts;
      ts_generated =
          GST_TIMESPEC_TO_TIME (testAppCtx->streams[stream_id].
          timespec_first_frame);
      if (ts_generated == 0) {
        g_print
            ("WARNING; playback mode used with URI not conforming to timestamp format;"
            " check README; using system-time\n");
        clock_gettime (CLOCK_REALTIME,
            &testAppCtx->streams[stream_id].timespec_first_frame);
        ts_generated =
            GST_TIMESPEC_TO_TIME (testAppCtx->streams[stream_id].
            timespec_first_frame);
      }
    } else {
      GstClockTime ts_current =
          GST_TIMESPEC_TO_TIME (testAppCtx->
          streams[stream_id].timespec_first_frame) + (ts -
          testAppCtx->streams[stream_id].gst_ts_first_frame);
      struct timespec timespec_current;
      GST_TIME_TO_TIMESPEC (ts_current, timespec_current);
      memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
      ms = timespec_current.tv_nsec / 1000000;
      ts_generated = ts_current;
    }
  } else {
    /** ts itself is UTC Time in ns */
    struct timespec timespec_current;
    GST_TIME_TO_TIMESPEC (ts, timespec_current);
    memcpy (&tloc, (void *) (&timespec_current.tv_sec), sizeof (time_t));
    ms = timespec_current.tv_nsec / 1000000;
    ts_generated = ts;
  }
  gmtime_r (&tloc, &tm_log);
  strftime (buf, buf_size, "%Y-%m-%dT%H:%M:%S", &tm_log);
  g_snprintf (strmsec, sizeof (strmsec), ".%.3dZ", ms);
  strncat (buf, strmsec, buf_size);
  LOGD ("ts=%s\n", buf);

  return ts_generated;
}


static gpointer
meta_copy_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  NvDsEventMsgMeta *dstMeta = NULL;

  dstMeta = (NvDsEventMsgMeta *) g_memdup2 (srcMeta, sizeof (NvDsEventMsgMeta));

  if (srcMeta->ts)
    dstMeta->ts = g_strdup (srcMeta->ts);

  if (srcMeta->objSignature.size > 0) {
    dstMeta->objSignature.signature = (gdouble *) g_memdup2 (srcMeta->objSignature.signature,
        srcMeta->objSignature.size);
    dstMeta->objSignature.size = srcMeta->objSignature.size;
  }

  if (srcMeta->objectId) {
    dstMeta->objectId = g_strdup (srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);
  }

  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *srcObj = (NvDsVehicleObject *) srcMeta->extMsg;
      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      if (srcObj->type)
        obj->type = g_strdup (srcObj->type);
      if (srcObj->make)
        obj->make = g_strdup (srcObj->make);
      if (srcObj->model)
        obj->model = g_strdup (srcObj->model);
      if (srcObj->color)
        obj->color = g_strdup (srcObj->color);
      if (srcObj->license)
        obj->license = g_strdup (srcObj->license);
      if (srcObj->region)
        obj->region = g_strdup (srcObj->region);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsVehicleObject);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

      obj->age = srcObj->age;

      if (srcObj->gender)
        obj->gender = g_strdup (srcObj->gender);
      if (srcObj->cap)
        obj->cap = g_strdup (srcObj->cap);
      if (srcObj->hair)
        obj->hair = g_strdup (srcObj->hair);
      if (srcObj->apparel)
        obj->apparel = g_strdup (srcObj->apparel);

      dstMeta->extMsg = obj;
      dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    }
  }

  if (srcMeta->embedding.embedding_length > 0) {
    dstMeta->embedding.embedding_length = srcMeta->embedding.embedding_length;
    dstMeta->embedding.embedding_vector =
        g_memdup2(srcMeta->embedding.embedding_vector,
                 srcMeta->embedding.embedding_length * sizeof(float));
  }

  if(srcMeta->has3DTracking) {
    dstMeta->has3DTracking = true;
    dstMeta->singleView3DTracking.ptWorldFeet[0] = srcMeta->singleView3DTracking.ptWorldFeet[0];
    dstMeta->singleView3DTracking.ptWorldFeet[1] = srcMeta->singleView3DTracking.ptWorldFeet[1];
    dstMeta->singleView3DTracking.ptImgFeet[0] = srcMeta->singleView3DTracking.ptImgFeet[0];
    dstMeta->singleView3DTracking.ptImgFeet[1] = srcMeta->singleView3DTracking.ptImgFeet[1];
    dstMeta->singleView3DTracking.convexHull.numFilled = srcMeta->singleView3DTracking.convexHull.numFilled;
    dstMeta->singleView3DTracking.convexHull.points =
        g_memdup2(srcMeta->singleView3DTracking.convexHull.points,
                    srcMeta->singleView3DTracking.convexHull.numFilled * 2 * sizeof(gint));
    memcpy(dstMeta->singleView3DTracking.bbox3d.boxes_3d,
           srcMeta->singleView3DTracking.bbox3d.boxes_3d,
           sizeof(srcMeta->singleView3DTracking.bbox3d.boxes_3d));
    dstMeta->singleView3DTracking.bbox3d.scores_3d = srcMeta->singleView3DTracking.bbox3d.scores_3d;
  }
  else {
    dstMeta->has3DTracking = false;
  }


  return dstMeta;
}

static void
meta_free_func (gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
  user_meta->user_meta_data = NULL;

  if (srcMeta->ts) {
    g_free (srcMeta->ts);
  }

  if (srcMeta->objSignature.size > 0) {
    g_free (srcMeta->objSignature.signature);
    srcMeta->objSignature.size = 0;
  }

  if (srcMeta->objectId) {
    g_free (srcMeta->objectId);
  }

  if (srcMeta->sensorStr) {
    g_free (srcMeta->sensorStr);
  }

  if (srcMeta->embedding.embedding_length > 0 && srcMeta->embedding.embedding_vector)
  {
    // First check if the embedding vector is from tracker or some other plugin
    // If it's from tracker, then NVDS_TRACKER_BATCH_REID_META will be present
    NvDsMetaList * l_user_meta = NULL;
    NvDsUserMeta *user_meta_local = NULL;
    bool is_from_tracker = false;
    for (l_user_meta = user_meta->base_meta.batch_meta->batch_user_meta_list; l_user_meta != NULL;
      l_user_meta = l_user_meta->next)
    {
      user_meta_local = (NvDsUserMeta *)(l_user_meta->data);
      if (user_meta_local->base_meta.meta_type == NVDS_TRACKER_BATCH_REID_META)
      {
        is_from_tracker = true;
        break;
      }
    }
    // Free the embedding vector only if it is not from tracker
    // In case of tracker, the embedding vector memory is owned by tracker, so we don't need to free it here
    if (!is_from_tracker) {
      // CS : TBD : Enable the following lines once crash is fixed
      // g_free (srcMeta->embedding.embedding_vector);
      // srcMeta->embedding.embedding_vector = NULL;
      // srcMeta->embedding.embedding_length = 0;
    }
  }
  if (srcMeta->has3DTracking && srcMeta->singleView3DTracking.convexHull.points) {
    g_free(srcMeta->singleView3DTracking.convexHull.points);
    srcMeta->singleView3DTracking.convexHull.numFilled = 0;
  }
  if (srcMeta->extMsgSize > 0) {
    if (srcMeta->objType == NVDS_OBJECT_TYPE_VEHICLE) {
      NvDsVehicleObject *obj = (NvDsVehicleObject *) srcMeta->extMsg;
      if (obj->type)
        g_free (obj->type);
      if (obj->color)
        g_free (obj->color);
      if (obj->make)
        g_free (obj->make);
      if (obj->model)
        g_free (obj->model);
      if (obj->license)
        g_free (obj->license);
      if (obj->region)
        g_free (obj->region);
    } else if (srcMeta->objType == NVDS_OBJECT_TYPE_PERSON) {
      NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

      if (obj->gender)
        g_free (obj->gender);
      if (obj->cap)
        g_free (obj->cap);
      if (obj->hair)
        g_free (obj->hair);
      if (obj->apparel)
        g_free (obj->apparel);
    }
    g_free (srcMeta->extMsg);
    srcMeta->extMsg = NULL;
    srcMeta->extMsgSize = 0;
  }
  g_free (srcMeta);
}

#ifdef GENERATE_DUMMY_META_EXT
static void
generate_vehicle_meta (gpointer data)
{
  NvDsVehicleObject *obj = (NvDsVehicleObject *) data;

  obj->type = g_strdup ("sedan-dummy");
  obj->color = g_strdup ("blue");
  obj->make = g_strdup ("Bugatti");
  obj->model = g_strdup ("M");
  obj->license = g_strdup ("XX1234");
  obj->region = g_strdup ("CA");
}

static void
generate_person_meta (gpointer data)
{
  NvDsPersonObject *obj = (NvDsPersonObject *) data;
  obj->age = 45;
  obj->cap = g_strdup ("none-dummy-person-info");
  obj->hair = g_strdup ("black");
  obj->gender = g_strdup ("male");
  obj->apparel = g_strdup ("formal");
}
#endif /**< GENERATE_DUMMY_META_EXT */

static void nanoseconds_to_rfc3339(int64_t nanoseconds, char *output, size_t output_size) {
    time_t seconds = nanoseconds / 1000000000;
    int32_t milliseconds = (nanoseconds % 1000000000) / 1000000;

    struct tm tm_buf;
    struct tm *tm_info = gmtime_r(&seconds, &tm_buf);

    char time_str[MAX_TIME_STAMP_LEN];
    strftime(time_str, MAX_TIME_STAMP_LEN, "%Y-%m-%dT%H:%M:%S", tm_info);

    g_snprintf(output, output_size, "%s.%03dZ", time_str, milliseconds);
}

static void
generate_event_msg_meta (AppCtx * appCtx, gpointer data, gint class_id, gboolean useTs,
    GstClockTime ts, gchar * src_uri, gint stream_id, guint sensor_id,
    NvDsObjectMeta * obj_params, float scaleW, float scaleH,
    NvDsFrameMeta * frame_meta,
    float *embedding_data, int numElements,
    gboolean embedding_on_device) {
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  GstClockTime ts_generated = 0;

  meta->objType = NVDS_OBJECT_TYPE_UNKNOWN; /**< object unknown */
  /* The sensor_id is parsed from the source group name which has the format
   * [source<sensor-id>]. */
  meta->sensorId = sensor_id;
  meta->placeId = sensor_id;
  meta->moduleId = sensor_id;
  meta->frameId = frame_meta->frame_num;
  meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);
  meta->confidence = obj_params->confidence;

  if (embedding_data) {
    // printf("Malloc embedding: %d\n", numElements*4);
    meta->embedding.embedding_vector =
        (float *)g_malloc0(numElements * sizeof(float));
    if (embedding_on_device) {
      cudaMemcpy(meta->embedding.embedding_vector, embedding_data,
               numElements * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
      cudaMemcpy(meta->embedding.embedding_vector, embedding_data,
               numElements * sizeof(float), cudaMemcpyHostToHost);
    }
    meta->embedding.embedding_length = numElements;
  } else {
    meta->embedding.embedding_length = 0;
  }
  strncpy (meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

  if(appCtx->config.custom_ts_to_rfc)
    nanoseconds_to_rfc3339(ts, meta->ts, MAX_TIME_STAMP_LEN);
  else {
    /** INFO: This API is called once for every 30 frames (now) */
    if ((useTs && src_uri) || appCtx->config.source_attr_all_config.type == NV_DS_SOURCE_IPC) {
      ts_generated =
      generate_ts_rfc3339_from_ts (meta->ts, MAX_TIME_STAMP_LEN, ts, src_uri,
      stream_id);
    } else {
      generate_ts_rfc3339 (meta->ts, MAX_TIME_STAMP_LEN);
    }
  }

  meta->has3DTracking = false;
  meta->visibility = 1.0;

  for (NvDsMetaList *l_user = obj_params->obj_user_meta_list;
             l_user != NULL; l_user = l_user->next) {
    NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
    if (user_meta->base_meta.meta_type == NVDS_OBJ_IMAGE_FOOT_LOCATION)
    {
      meta->has3DTracking = true;
      float *pPtFeet = (float *)user_meta->user_meta_data;
      meta->singleView3DTracking.ptImgFeet[0] = pPtFeet[0];
      meta->singleView3DTracking.ptImgFeet[1] = pPtFeet[1];
    }
    else if (user_meta->base_meta.meta_type == NVDS_OBJ_WORLD_FOOT_LOCATION)
    {
      float *pPtFeet = (float *)user_meta->user_meta_data;
      meta->singleView3DTracking.ptWorldFeet[0] = pPtFeet[0];
      meta->singleView3DTracking.ptWorldFeet[1] = pPtFeet[1];
    }
    else if (user_meta->base_meta.meta_type == NVDS_OBJ_VISIBILITY)
    {
      meta->visibility = *(float *)user_meta->user_meta_data;
    }
    else if (user_meta->base_meta.meta_type == NVDS_OBJ_IMAGE_CONVEX_HULL)
    {
      NvDsObjConvexHull *pConvexHull = (NvDsObjConvexHull *)user_meta->user_meta_data;
      meta->singleView3DTracking.convexHull.points = g_malloc0(sizeof(gint) * pConvexHull->numPointsAllocated * 2);
      meta->singleView3DTracking.convexHull.numFilled = pConvexHull->numPoints;
      for (uint32_t i = 0; i < pConvexHull->numPoints; i++)
      {
        meta->singleView3DTracking.convexHull.points[2 * i] = pConvexHull->list[2 * i];
        meta->singleView3DTracking.convexHull.points[2 * i + 1] = pConvexHull->list[2 * i + 1];
      }
    }
    else if (user_meta->base_meta.meta_type == NVDS_OBJ_3D_META)
    {
      meta->has3DTracking = true;
      NvDsObj3DBbox *p3DBbox = (NvDsObj3DBbox *)user_meta->user_meta_data;
      meta->singleView3DTracking.bbox3d.boxes_3d[0] = p3DBbox->xCentre;
      meta->singleView3DTracking.bbox3d.boxes_3d[1] = p3DBbox->yCentre;
      meta->singleView3DTracking.bbox3d.boxes_3d[2] = p3DBbox->zCentre;
      meta->singleView3DTracking.bbox3d.boxes_3d[3] = p3DBbox->xLen;
      meta->singleView3DTracking.bbox3d.boxes_3d[4] = p3DBbox->yLen;
      meta->singleView3DTracking.bbox3d.boxes_3d[5] = p3DBbox->zLen;
      meta->singleView3DTracking.bbox3d.boxes_3d[6] = p3DBbox->xRot;
      meta->singleView3DTracking.bbox3d.boxes_3d[7] = p3DBbox->yRot;
      meta->singleView3DTracking.bbox3d.boxes_3d[8] = p3DBbox->zRot;
      meta->singleView3DTracking.bbox3d.boxes_3d[9] = p3DBbox->xVel;
      meta->singleView3DTracking.bbox3d.boxes_3d[10] = p3DBbox->yVel;
      meta->singleView3DTracking.bbox3d.boxes_3d[11] = p3DBbox->zVel;
      // Set detection confidence
      meta->singleView3DTracking.bbox3d.scores_3d = obj_params->confidence;
    }
  }

  meta->confidence = obj_params->tracker_confidence;
  /**
   * Valid attributes in the metadata sent over nvmsgbroker:
   * a) Sensor ID (shall be configured in nvmsgconv config file)
   * b) bbox info (meta->bbox) <- obj_params->rect_params (attr_info have sgie info)
   * c) tracking ID (meta->trackingId) <- obj_params->object_id
   */

  /** bbox - resolution is scaled by nvinfer back to
   * the resolution provided by streammux
   * We have to scale it back to original stream resolution
    */

  meta->bbox.left = obj_params->rect_params.left * scaleW;
  meta->bbox.top = obj_params->rect_params.top * scaleH;
  meta->bbox.width = obj_params->rect_params.width * scaleW;
  meta->bbox.height = obj_params->rect_params.height * scaleH;

  /** tracking ID */
  meta->trackingId = obj_params->object_id;

  /** sensor ID when streams are added using nvmultiurisrcbin REST API */
  NvDsSensorInfo* sensorInfo = get_sensor_info(appCtx, stream_id);
  if(sensorInfo) {
    /** this stream was added using REST API; we have Sensor Info! */
    LOGD("this stream [%d:%s] was added using REST API; we have Sensor Info\n",
        sensorInfo->source_id, sensorInfo->sensor_id);
    meta->sensorStr = g_strdup (sensorInfo->sensor_id);
  }

  (void) ts_generated;

  /*
   * This demonstrates how to attach custom objects.
   * Any custom object as per requirement can be generated and attached
   * like NvDsVehicleObject / NvDsPersonObject. Then that object should
   * be handled in gst-nvmsgconv component accordingly.
   */
  if (model_used == APP_CONFIG_ANALYTICS_RESNET_PGIE_3SGIE_TYPE_COLOR_MAKE) {
    if (class_id == RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_CAR) {
      meta->type = NVDS_EVENT_MOVING;
      meta->objType = NVDS_OBJECT_TYPE_VEHICLE;
      meta->objClassId = RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_CAR;

      NvDsVehicleObject *obj =
          (NvDsVehicleObject *) g_malloc0 (sizeof (NvDsVehicleObject));
      schema_fill_sample_sgie_vehicle_metadata (obj_params, obj);

      meta->extMsg = obj;
      meta->extMsgSize = sizeof (NvDsVehicleObject);
    }
#ifdef GENERATE_DUMMY_META_EXT
    else if (class_id == RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_PERSON) {
      meta->type = NVDS_EVENT_ENTRY;
      meta->objType = NVDS_OBJECT_TYPE_PERSON;
      meta->objClassId = RESNET10_PGIE_3SGIE_TYPE_COLOR_MAKECLASS_ID_PERSON;

      NvDsPersonObject *obj =
          (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
      generate_person_meta (obj);

      meta->extMsg = obj;
      meta->extMsgSize = sizeof (NvDsPersonObject);
    }
#endif /**< GENERATE_DUMMY_META_EXT */
  }
}

static void
generate_event_msg_meta_dummy (AppCtx * appCtx, gpointer data, gint stream_id,
    NvDsFrameMeta * frame_meta)
{
  NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
  GstClockTime ts_generated = 0;

  meta->objType = NVDS_OBJECT_TYPE_DUMMY; /**< object unknown */
  /* The sensor_id is parsed from the source group name which has the format
   * [source<sensor-id>]. */
  meta->sensorId = appCtx->config.multi_source_config[stream_id].camera_id;
  meta->placeId = appCtx->config.multi_source_config[stream_id].camera_id;
  meta->moduleId = appCtx->config.multi_source_config[stream_id].camera_id;
  meta->frameId = frame_meta->frame_num;
  meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
  nanoseconds_to_rfc3339(frame_meta->ntp_timestamp, meta->ts, MAX_TIME_STAMP_LEN);

  /** sensor ID when streams are added using nvmultiurisrcbin REST API */
  NvDsSensorInfo* sensorInfo = get_sensor_info(appCtx, stream_id);
  if(sensorInfo) {
    /** this stream was added using REST API; we have Sensor Info! */
    LOGD("this stream [%d:%s] was added using REST API; we have Sensor Info\n",
        sensorInfo->source_id, sensorInfo->sensor_id);
    meta->sensorStr = g_strdup (sensorInfo->sensor_id);
  }
  (void) ts_generated;
}

/**
 * Callback function to be called once all inferences (Primary + Secondary)
 * are done. This is opportunity to modify content of the metadata.
 * e.g. Here Person is being replaced with Man/Woman and corresponding counts
 * are being maintained. It should be modified according to network classes
 * or can be removed altogether if not required.
 */
static void
bbox_generated_probe_after_analytics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  guint flag=1;
  char *verbose = getenv("SPARSE4D_DEBUG_TS");

  frame_audit_log_batch ("production_probe", batch_meta);
  mv3dt_crop_encoder_process (appCtx, buf, batch_meta);

  if(batch_meta->batch_user_meta_list) {  //for SPARSE4D model
    for (NvDsMetaList *l = batch_meta->batch_user_meta_list; l; l = l->next) {
      NvDsUserMeta *user_event_meta = (NvDsUserMeta *)(l->data);
      NvDsBbox3dObjectList *bbox3d_list = (NvDsBbox3dObjectList *)user_event_meta->user_meta_data;

      if ((user_event_meta && user_event_meta->base_meta.meta_type == NVDS_CUSTOM_MSG_SPARSE4D)) {
        flag=0;
        for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
          l_frame = l_frame->next) {
          NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
          if (verbose != NULL && atoi(verbose) == 1) {
            if ((bbox3d_list->count < MAX_ENTRIES) && (frame_meta != NULL)) {
              g_strlcpy(bbox3d_list->entries[bbox3d_list->count].source_id,
              frame_meta->sensorInfo_meta.sensor_name,
              MAX_SOURCE_ID_LEN);
              bbox3d_list->entries[bbox3d_list->count].timestamp = frame_meta->ntp_timestamp;
              bbox3d_list->count++;
            } else {
              g_print ("This is either EOS case or some error has happened\n");
              break;
            }
          }
        }
      }
    }
  }
  if (flag==1){
    NvDsObjectMeta *obj_meta = NULL;
    GstClockTime buffer_pts = 0;
    guint32 stream_id = 0;

    for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
      NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
      stream_id = frame_meta->source_id;
      GstClockTime buf_ntp_time = 0;
      if (playback_utc == FALSE) {
        /** Calculate the buffer-NTP-time
         * derived from this stream's RTCP Sender Report here:
         */
        StreamSourceInfo *src_stream = &testAppCtx->streams[stream_id];
        buf_ntp_time = frame_meta->ntp_timestamp;

          if (buf_ntp_time < src_stream->last_ntp_time) {
            GST_WARNING ("Source %d: NTP timestamps are backward in time."
            " Current: %lu previous: %lu \n",stream_id, buf_ntp_time, src_stream->last_ntp_time);
          }
          src_stream->last_ntp_time = buf_ntp_time;
      }

        GList *l;
      if(frame_meta->num_obj_meta) {
        for (l = frame_meta->obj_meta_list; l != NULL; l = l->next) {
        /* Now using above information we need to form a text that should
         * be displayed on top of the bounding box, so lets form it here. */

          obj_meta = (NvDsObjectMeta *) (l->data);
          // Get reid meta from obj user meta
          float *embedding_data = NULL;
          int numElements = 0;
          gboolean embedding_on_device = false;
          int user_meta_count = 0;
          int reid_meta_count = 0;
          //! Attaching Embedding tensor metadata
          for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list;
              l_user != NULL; l_user = l_user->next) {
            user_meta_count++;
            NvDsUserMeta *user_meta = (NvDsUserMeta *)l_user->data;
            if (user_meta->base_meta.meta_type == NVDS_TRACKER_OBJ_REID_META) {
              reid_meta_count++;
              /** Use embedding from tracker reid*/
              NvDsObjReid *pReidObj = (NvDsObjReid *) (user_meta->user_meta_data);
              if (pReidObj != NULL && pReidObj->ptr_host != NULL && pReidObj->featureSize > 0) {
                numElements = pReidObj->featureSize;
                embedding_data = (float *)(pReidObj->ptr_host);
              }
            }
          }
          {
            /**
             * Enable only if this callback is after tiler
             * NOTE: Scaling back code-commented
             * now that bbox_generated_probe_after_analytics() is post analytics
             * (say pgie, tracker or sgie)
             * and before tiler, no plugin shall scale metadata and will be
             * corresponding to the nvstreammux resolution
             */
            float scaleW = 0;
            float scaleH = 0;
            /* Frequency of messages to be send will be based on use case.
             * Here message is being sent for first object every 30 frames.
             */
            buffer_pts = frame_meta->buf_pts;
            if (!appCtx->config.streammux_config.pipeline_width
              || !appCtx->config.streammux_config.pipeline_height) {
              g_print ("invalid pipeline params\n");
              return;
            }
            LOGD ("stream %d==%d [%d X %d]\n", frame_meta->source_id,
              frame_meta->pad_index, frame_meta->source_frame_width,
              frame_meta->source_frame_height);
            scaleW =
              (float) frame_meta->source_frame_width /
              appCtx->config.streammux_config.pipeline_width;
            scaleH =
              (float) frame_meta->source_frame_height /
              appCtx->config.streammux_config.pipeline_height;

            if (playback_utc == FALSE) {
              /** Use the buffer-NTP-time derived from this stream's RTCP Sender
               * Report here:
               */
              buffer_pts = buf_ntp_time;
            }
            /** Generate NvDsEventMsgMeta for every object */
            NvDsEventMsgMeta *msg_meta =
              (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
            generate_event_msg_meta (appCtx, msg_meta, obj_meta->class_id, TRUE,
              /**< useTs NOTE: Pass FALSE for files without base-timestamp in URI */
              buffer_pts,
              appCtx->config.multi_source_config[stream_id].uri, stream_id,
              appCtx->config.multi_source_config[stream_id].camera_id,
              obj_meta, scaleW, scaleH, frame_meta,
              embedding_data, numElements, embedding_on_device);
            testAppCtx->streams[stream_id].meta_number++;
            NvDsUserMeta *user_event_meta =
              nvds_acquire_user_meta_from_pool (batch_meta);
            if (user_event_meta) {
              /*
               * Since generated event metadata has custom objects for
               * Vehicle / Person which are allocated dynamically, we are
               * setting copy and free function to handle those fields when
               * metadata copy happens between two components.
               */
              user_event_meta->user_meta_data = (void *) msg_meta;
              user_event_meta->base_meta.batch_meta = batch_meta;
              user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
              user_event_meta->base_meta.copy_func =
                (NvDsMetaCopyFunc) meta_copy_func;
              user_event_meta->base_meta.release_func =
                (NvDsMetaReleaseFunc) meta_free_func;
              nvds_add_user_meta_to_frame (frame_meta, user_event_meta);
            } else {
              g_print ("Error in attaching event meta to buffer\n");
            }
          }
        }
      }
      else if(appCtx->config.dummy_payload)
      {
        NvDsEventMsgMeta *msg_meta =
          (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
        generate_event_msg_meta_dummy (appCtx, msg_meta, stream_id, frame_meta);
          NvDsUserMeta *user_event_meta =
            nvds_acquire_user_meta_from_pool (batch_meta);
          if (user_event_meta) {
            /*
             * Since generated event metadata has custom objects for
             * Vehicle / Person which are allocated dynamically, we are
             * setting copy and free function to handle those fields when
             * metadata copy happens between two components.
             */
            user_event_meta->user_meta_data = (void *) msg_meta;
            user_event_meta->base_meta.batch_meta = batch_meta;
            user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
            user_event_meta->base_meta.copy_func =
              (NvDsMetaCopyFunc) meta_copy_func;
            user_event_meta->base_meta.release_func =
              (NvDsMetaReleaseFunc) meta_free_func;
            nvds_add_user_meta_to_frame (frame_meta, user_event_meta);
          } else {
            g_print ("Error in attaching event meta to buffer\n");
          }
      }
      testAppCtx->streams[stream_id].frameCount++;
    }
  }
}

/** @{ imported from deepstream-app as is */

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler (int signum)
{
  struct sigaction action;

  NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  cintr = TRUE;
}

/**
 * callback function to print the performance numbers of each stream.
 */
static void
perf_cb (gpointer context, NvDsAppPerfStruct * str)
{
  static guint header_print_cnt = 0;
  guint i;
  AppCtx *appCtx = (AppCtx *) context;
  guint numf = str->num_instances;

  g_mutex_lock (&fps_lock);
  guint active_src_count = 0;

  if (!str->use_nvmultiurisrcbin) {
    for (i = 0; i < numf; i++) {
      fps[i] = str->fps[i];
      if (fps[i]){
        active_src_count++;
      }
      fps_avg[i] = str->fps_avg[i];
    }
    g_print("Active sources : %u\n", active_src_count);
    if (header_print_cnt % 20 == 0) {
      g_print ("\n**PERF:  ");
      for (i = 0; i < numf; i++) {
        g_print ("FPS %d (Avg)\t", i);
      }
      g_print ("\n");
      header_print_cnt = 0;
    }
    header_print_cnt++;

    time_t t = time (NULL);
    struct tm tm_buf;
    struct tm *tm = localtime_r (&t, &tm_buf);
    char time_buf[26];
    printf ("%s", asctime_r (tm, time_buf));
    if (num_instances > 1)
      g_print ("PERF(%d): ", appCtx->index);
    else
      g_print ("**PERF:  ");

    for (i = 0; i < numf; i++) {
      g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
    }
  } else {
    for (guint j = 0; j < str->active_source_size; j++) {
      i = str->source_detail[j].source_id;
      fps[i] = str->fps[i];
      if (fps[i]){
        active_src_count++;
      }
      fps_avg[i] = str->fps_avg[i];
    }
    g_print("Active sources : %u\n", active_src_count);
    if (header_print_cnt % 20 == 0) {
      g_print ("\n**PERF:  ");
      for (guint j = 0; j < str->active_source_size; j++) {
        i = str->source_detail[j].source_id;
        g_print ("FPS %d (Avg)\t", i);
      }
      g_print ("\n");
      header_print_cnt = 0;
    }
    header_print_cnt++;

    time_t t = time (NULL);
    struct tm tm_buf;
    struct tm *tm = localtime_r (&t, &tm_buf);
    char time_buf[26];
    printf ("%s", asctime_r (tm, time_buf));
    if (num_instances > 1)
      g_print ("PERF(%d): ", appCtx->index);
    else
      g_print ("**PERF:  ");

    g_print("\n");
    for (guint j = 0; j < str->active_source_size; j++) {
      i = str->source_detail[j].source_id;
      if (!str->stream_name_display){
        g_print ("%.2f (%.2f)\t", fps[i], fps_avg[i]);
      }
      else {
        g_print("%s[%s] %.2f (%.2f)\t", str->source_detail[j].sensor_id,str->source_detail[j].sensor_name,fps[i], fps_avg[i]);
      }
    }
  }
  g_print ("\n");
  g_mutex_unlock (&fps_lock);
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (quit) {
    return FALSE;
  }

  if (cintr) {
    cintr = FALSE;

    quit = TRUE;
    g_main_loop_quit (main_loop);

    return FALSE;
  }
  return TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/*
 * Function to enable / disable the canonical mode of terminal.
 * In non canonical mode input is available immediately (without the user
 * having to type a line-delimiter character).
 */
static void
changemode (int dir)
{
  static struct termios oldt, newt;

  if (dir == 1) {
    tcgetattr (STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON);
    tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  } else
    tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
}

static void
print_runtime_commands (void)
{
  g_print ("\nRuntime commands:\n"
      "\th: Print this help\n"
      "\tq: Quit\n\n" "\tp: Pause\n" "\tr: Resume\n\n");

  if (appCtx[0]->config.tiled_display_config.enable) {
    g_print
        ("NOTE: To expand a source in the 2D tiled display and view object details,"
        " left-click on the source.\n"
        "      To go back to the tiled display, right-click anywhere on the window.\n\n");
  }
}

/**
 * Loop function to check keyboard inputs and status of each pipeline.
 */
static gboolean
event_thread_func (gpointer arg)
{
  guint i;
  gboolean ret = TRUE;

  // Check if all instances have quit
  for (i = 0; i < num_instances; i++) {
    if (!appCtx[i]->quit)
      break;
  }

  if (i == num_instances) {
    quit = TRUE;
    g_main_loop_quit (main_loop);
    return FALSE;
  }
  // Check for keyboard input
  if (!kbhit ()) {
    //continue;
    return TRUE;
  }
  int c = fgetc (stdin);
  g_print ("\n");

  gint source_id = -1;
  GstElement *tiler = appCtx[rcfg]->pipeline.tiled_display_bin.tiler;

  /* Only access show-source property if tiler is nvmultistreamtiler (not identity) */
  if (appCtx[rcfg]->config.tiled_display_config.enable &&
      (appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
       appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX))
  {
    g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);

    if (selecting) {
      if (rrowsel == FALSE) {
        if (c >= '0' && c <= '9') {
          rrow = c - '0';
          g_print ("--selecting source  row %d--\n", rrow);
          rrowsel = TRUE;
        }
      } else {
        if (c >= '0' && c <= '9') {
          int tile_num_columns = appCtx[rcfg]->config.tiled_display_config.columns;
          rcol = c - '0';
          selecting = FALSE;
          rrowsel = FALSE;
          source_id = tile_num_columns * rrow + rcol;
          g_print ("--selecting source  col %d sou=%d--\n", rcol, source_id);
          if (source_id >= (gint) appCtx[rcfg]->config.num_source_sub_bins) {
            source_id = -1;
          } else if (appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
                     appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX) {
            appCtx[rcfg]->show_bbox_text = TRUE;
            appCtx[rcfg]->active_source_index = source_id;
            g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
          }
        }
      }
    }
  }
  switch (c) {
    case 'h':
      print_runtime_commands ();
      break;
    case 'p':
      for (i = 0; i < num_instances; i++)
        pause_pipeline (appCtx[i]);
      break;
    case 'r':
      for (i = 0; i < num_instances; i++)
        resume_pipeline (appCtx[i]);
      break;
    case 'q':
      quit = TRUE;
      g_main_loop_quit (main_loop);
      ret = FALSE;
      break;
    case 'c':
      if (appCtx[rcfg]->config.tiled_display_config.enable && selecting == FALSE && source_id == -1) {
        g_print("--selecting config file --\n");
        c = fgetc(stdin);
        if (c >= '0' && c <= '9') {
          rcfg = c - '0';
          if (rcfg < num_instances) {
            g_print("--selecting config  %d--\n", rcfg);
          } else {
            g_print("--selected config file %d out of bound, reenter\n", rcfg);
            rcfg = 0;
          }
        }
      }
      break;
    case 'z':
      if (appCtx[rcfg]->config.tiled_display_config.enable && source_id == -1 && selecting == FALSE) {
        g_print ("--selecting source --\n");
        selecting = TRUE;
      } else {
        if (!show_bbox_text) {
          GstElement *nvosd = appCtx[rcfg]->pipeline.instance_bins[0].osd_bin.nvosd;
          g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
          if (appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
              appCtx[rcfg]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX) {
            g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
          }
        }
        appCtx[rcfg]->active_source_index = -1;
        selecting = FALSE;
        rcfg = 0;
        g_print("--tiled mode --\n");
      }
      break;
    default:
      break;
  }
  return ret;
}

static int
get_source_id_from_coordinates (float x_rel, float y_rel, AppCtx *appCtx)
{
  int tile_num_rows = appCtx->config.tiled_display_config.rows;
  int tile_num_columns = appCtx->config.tiled_display_config.columns;

  int source_id = (int) (x_rel * tile_num_columns);
  source_id += ((int) (y_rel * tile_num_rows)) * tile_num_columns;

  /* Don't allow clicks on empty tiles. */
  if (source_id >= (gint) appCtx->config.num_source_sub_bins)
    source_id = -1;

  return source_id;
}

/**
 * Thread to monitor X window events.
 */
static gpointer
nvds_x_event_thread (gpointer data)
{
  g_mutex_lock (&disp_lock);
  while (display) {
    XEvent e;
    guint index;
    memset(&e, 0, sizeof(XEvent));
    while (XPending (display)) {
      XNextEvent (display, &e);
      switch (e.type) {
        case ButtonPress:
        {
          XWindowAttributes win_attr;
          XButtonEvent ev = e.xbutton;
          gint source_id;
          GstElement *tiler;
          memset(&win_attr, 0, sizeof(XWindowAttributes));
          XGetWindowAttributes (display, ev.window, &win_attr);

          for (index = 0; index < MAX_INSTANCES; index++)
            if (ev.window == windows[index])
              break;

          tiler = appCtx[index]->pipeline.tiled_display_bin.tiler;

          /* Only access show-source property if tiler is nvmultistreamtiler (not identity) */
          if (appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
              appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX) {
            g_object_get (G_OBJECT (tiler), "show-source", &source_id, NULL);
          } else {
            source_id = -1;
          }

          if (ev.button == Button1 && source_id == -1 && (index >=0 && index < MAX_INSTANCES )) {
            source_id =
                get_source_id_from_coordinates (ev.x * 1.0 / win_attr.width,
                ev.y * 1.0 / win_attr.height, appCtx[index]);
            if (source_id > -1 &&
                (appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
                 appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX)) {
              g_object_set (G_OBJECT (tiler), "show-source", source_id, NULL);
              appCtx[index]->active_source_index = source_id;
              appCtx[index]->show_bbox_text = TRUE;
              GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
              g_object_set (G_OBJECT (nvosd), "display-text", TRUE, NULL);
            }
          } else if (ev.button == Button3 &&
                     (appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE ||
                      appCtx[index]->config.tiled_display_config.enable == NV_DS_TILED_DISPLAY_ENABLE_WITH_PARALLEL_DEMUX)) {
            g_object_set (G_OBJECT (tiler), "show-source", -1, NULL);
            appCtx[index]->active_source_index = -1;
            if (!show_bbox_text) {
              appCtx[index]->show_bbox_text = FALSE;
              GstElement *nvosd = appCtx[index]->pipeline.instance_bins[0].osd_bin.nvosd;
              g_object_set (G_OBJECT (nvosd), "display-text", FALSE, NULL);
            }
          }
        }
          break;
        case KeyRelease:
        {
          KeySym p, r, q;
          guint i;
          p = XKeysymToKeycode (display, XK_P);
          r = XKeysymToKeycode (display, XK_R);
          q = XKeysymToKeycode (display, XK_Q);
          if (e.xkey.keycode == p) {
            for (i = 0; i < num_instances; i++)
              pause_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == r) {
            for (i = 0; i < num_instances; i++)
              resume_pipeline (appCtx[i]);
            break;
          }
          if (e.xkey.keycode == q) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
        case ClientMessage:
        {
          Atom wm_delete;
          for (index = 0; index < MAX_INSTANCES; index++)
            if (e.xclient.window == windows[index])
              break;

          wm_delete = XInternAtom (display, "WM_DELETE_WINDOW", 1);
          if (wm_delete != None && wm_delete == (Atom) e.xclient.data.l[0]) {
            quit = TRUE;
            g_main_loop_quit (main_loop);
          }
        }
          break;
      }
    }
    g_mutex_unlock (&disp_lock);
    g_usleep (G_USEC_PER_SEC / 20);
    g_mutex_lock (&disp_lock);
  }
  g_mutex_unlock (&disp_lock);
  return NULL;
}

/**
 * callback function to add application specific metadata.
 * Here it demonstrates how to display the URI of source in addition to
 * the text generated after inference.
 */
static gboolean
overlay_graphics (AppCtx * appCtx, GstBuffer * buf,
    NvDsBatchMeta * batch_meta, guint index)
{
  return TRUE;
}

/**
 * Callback function to notify the status of the model update
 */
static void
infer_model_updated_cb (GstElement * gie, gint err, const gchar * config_file)
{
  double otaTime = 0;
  gettimeofday (&ota_completion_time, NULL);

  otaTime = (ota_completion_time.tv_sec - ota_request_time.tv_sec) * 1000.0;
  otaTime += (ota_completion_time.tv_usec - ota_request_time.tv_usec) / 1000.0;

  const char *err_str = (err == 0 ? "ok" : "failed");
  g_print
      ("\nModel Update Status: Updated model : %s, OTATime = %f ms, result: %s \n\n",
      config_file, otaTime, err_str);
}

/**
 * Function to print detected Inotify handler events
 * Used only for debugging purposes
 */
static void
display_inotify_event (struct inotify_event *i_event)
{
  printf ("    watch decriptor =%2d; ", i_event->wd);
  if (i_event->cookie > 0)
    printf ("cookie =%4d; ", i_event->cookie);

  printf ("mask = ");
  if (i_event->mask & IN_ACCESS)
    printf ("IN_ACCESS ");
  if (i_event->mask & IN_ATTRIB)
    printf ("IN_ATTRIB ");
  if (i_event->mask & IN_CLOSE_NOWRITE)
    printf ("IN_CLOSE_NOWRITE ");
  if (i_event->mask & IN_CLOSE_WRITE)
    printf ("IN_CLOSE_WRITE ");
  if (i_event->mask & IN_CREATE)
    printf ("IN_CREATE ");
  if (i_event->mask & IN_DELETE)
    printf ("IN_DELETE ");
  if (i_event->mask & IN_DELETE_SELF)
    printf ("IN_DELETE_SELF ");
  if (i_event->mask & IN_IGNORED)
    printf ("IN_IGNORED ");
  if (i_event->mask & IN_ISDIR)
    printf ("IN_ISDIR ");
  if (i_event->mask & IN_MODIFY)
    printf ("IN_MODIFY ");
  if (i_event->mask & IN_MOVE_SELF)
    printf ("IN_MOVE_SELF ");
  if (i_event->mask & IN_MOVED_FROM)
    printf ("IN_MOVED_FROM ");
  if (i_event->mask & IN_MOVED_TO)
    printf ("IN_MOVED_TO ");
  if (i_event->mask & IN_OPEN)
    printf ("IN_OPEN ");
  if (i_event->mask & IN_Q_OVERFLOW)
    printf ("IN_Q_OVERFLOW ");
  if (i_event->mask & IN_UNMOUNT)
    printf ("IN_UNMOUNT ");

  if (i_event->mask & IN_CLOSE)
    printf ("IN_CLOSE ");
  if (i_event->mask & IN_MOVE)
    printf ("IN_MOVE ");
  if (i_event->mask & IN_UNMOUNT)
    printf ("IN_UNMOUNT ");
  if (i_event->mask & IN_IGNORED)
    printf ("IN_IGNORED ");
  if (i_event->mask & IN_Q_OVERFLOW)
    printf ("IN_Q_OVERFLOW ");
  printf ("\n");

  if (i_event->len > 0)
    printf ("        name = %s mask= %x \n", i_event->name, i_event->mask);
}

/**
 * Perform model-update OTA operation
 */
void
apply_ota (AppCtx * ota_appCtx)
{
  GstElement *primary_gie = NULL;

  if (ota_appCtx->override_config.primary_gie_config.enable) {
    primary_gie =
        ota_appCtx->pipeline.common_elements.primary_gie_bin.primary_gie;
    gchar *model_engine_file_path =
        ota_appCtx->override_config.primary_gie_config.model_engine_file_path;

    gettimeofday (&ota_request_time, NULL);
    if (model_engine_file_path) {
      g_print ("\nNew Model Update Request %s ----> %s\n",
          GST_ELEMENT_NAME (primary_gie), model_engine_file_path);
      g_object_set (G_OBJECT (primary_gie), "model-engine-file",
          model_engine_file_path, NULL);
    } else {
      g_print
          ("\nInvalid New Model Update Request received. Property model-engine-path is not set\n");
    }
  }
}

/**
 * Independent thread to perform model-update OTA process based on the inotify events
 * It handles currently two scenarios
 * 1) Local Model Update Request (e.g. Standalone Appliation)
 *    In this case, notifier handler watches for the ota_override_file changes
 * 2) Cloud Model Update Request (e.g. EGX with Kubernetes)
 *    In this case, notifier handler watches for the ota_override_file changes along with
 *    ..data directory which gets mounted by EGX deployment in Kubernetes environment.
 */
gpointer
ota_handler_thread (gpointer data)
{

  ssize_t length = 0;
  size_t i = 0;
  char buffer[INOTIFY_EVENT_BUF_LEN];
  OTAInfo *ota = (OTAInfo *) data;
  gchar *ota_ds_config_file = ota->override_cfg_file;
  AppCtx *ota_appCtx = ota->appCtx;
  struct stat file_stat = { 0 };
  GstElement *primary_gie = NULL;
  gboolean connect_pgie_signal = FALSE;

  ota_appCtx->ota_inotify_fd = inotify_init ();

  if (ota_appCtx->ota_inotify_fd < 0) {
    perror ("inotify_init");
    return NULL;
  }

  char *real_path_ds_config_file = realpath (ota_ds_config_file, NULL);
  g_print ("REAL PATH = %s\n", real_path_ds_config_file);

  gchar *ota_dir = g_path_get_dirname (real_path_ds_config_file);
  ota_appCtx->ota_watch_desc =
      inotify_add_watch (ota_appCtx->ota_inotify_fd, ota_dir, IN_ALL_EVENTS);

  int ret = lstat (ota_ds_config_file, &file_stat);
  ret = ret;

  if (S_ISLNK (file_stat.st_mode)) {
    printf (" Override File Provided is Soft Link\n");
    gchar *parent_ota_dir = g_strdup_printf ("%s/..", ota_dir);
    ota_appCtx->ota_watch_desc =
        inotify_add_watch (ota_appCtx->ota_inotify_fd, parent_ota_dir,
        IN_ALL_EVENTS);
  }

  while (1) {
    i = 0;
    length = read (ota_appCtx->ota_inotify_fd, buffer, INOTIFY_EVENT_BUF_LEN);

    if (length < 0) {
      perror ("read");
    }

    if (quit == TRUE)
      goto done;

    while (i < (size_t)length) {
      struct inotify_event *event = (struct inotify_event *) &buffer[i];

      // Enable below function to print the inotify events, used for debugging purpose
      if (0) {
        display_inotify_event (event);
      }

      if (connect_pgie_signal == FALSE) {
        primary_gie =
            ota_appCtx->pipeline.common_elements.primary_gie_bin.primary_gie;
        if (primary_gie) {
          g_signal_connect (G_OBJECT (primary_gie), "model-updated",
              G_CALLBACK (infer_model_updated_cb), NULL);
          connect_pgie_signal = TRUE;
        } else {
          printf
              ("Gstreamer pipeline element nvinfer is yet to be created or invalid\n");
          continue;
        }
      }
      // Ensure null termination
      if (event->len < INOTIFY_EVENT_BUF_LEN && event->len < MAX_NAME_LENGTH ) {
        event->name[event->len] = '\0';
      } else if (INOTIFY_EVENT_BUF_LEN < MAX_NAME_LENGTH){
        event->name[INOTIFY_EVENT_BUF_LEN - 1] = '\0';
      } else {
        event->name[MAX_NAME_LENGTH - 1] = '\0';
      }

      if (event->len) {
        if (event->mask & IN_MOVED_TO) {
          if (strstr ("..data", event->name)) {
            memset (&ota_appCtx->override_config, 0,
                sizeof (ota_appCtx->override_config));
            if (!IS_YAML(ota_ds_config_file)) {
              if (!parse_config_file (&ota_appCtx->override_config,
                      ota_ds_config_file)) {
                NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                    ota_ds_config_file);
                g_print
                    ("Error: ota_handler_thread: Failed to parse config file '%s'",
                    ota_ds_config_file);
              } else {
                apply_ota (ota_appCtx);
              }
            } else if (IS_YAML(ota_ds_config_file)) {
                if (!parse_config_file_yaml (&ota_appCtx->override_config,
                      ota_ds_config_file)) {
                NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                    ota_ds_config_file);
                g_print
                    ("Error: ota_handler_thread: Failed to parse config file '%s'",
                    ota_ds_config_file);
              } else {
                apply_ota (ota_appCtx);
              }
            }
          }
        }
        if (event->mask & IN_CLOSE_WRITE) {
          if (!(event->mask & IN_ISDIR)) {
            if (strstr (ota_ds_config_file, event->name)) {
              g_print ("File %s modified.\n", event->name);

              memset (&ota_appCtx->override_config, 0,
                  sizeof (ota_appCtx->override_config));
              if (!IS_YAML(ota_ds_config_file)) {
                if (!parse_config_file (&ota_appCtx->override_config,
                        ota_ds_config_file)) {
                  NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                      ota_ds_config_file);
                  g_print
                      ("Error: ota_handler_thread: Failed to parse config file '%s'",
                      ota_ds_config_file);
                } else {
                  apply_ota (ota_appCtx);
                }
              } else if (IS_YAML(ota_ds_config_file)) {
                  if (!parse_config_file_yaml (&ota_appCtx->override_config,
                        ota_ds_config_file)) {
                  NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'",
                      ota_ds_config_file);
                  g_print
                      ("Error: ota_handler_thread: Failed to parse config file '%s'",
                      ota_ds_config_file);
                } else {
                  apply_ota (ota_appCtx);
                }
              }
            }
          }
        }
      }
      i += INOTIFY_EVENT_SIZE + event->len;
    }
  }
done:
  inotify_rm_watch (ota_appCtx->ota_inotify_fd, ota_appCtx->ota_watch_desc);
  close (ota_appCtx->ota_inotify_fd);

  free (real_path_ds_config_file);
  g_free (ota_dir);

  g_free (ota);
  return NULL;
}

/** @} imported from deepstream-app as is */

int
main (int argc, char *argv[])
{
  testAppCtx = (TestAppCtx *) g_malloc0 (sizeof (TestAppCtx));
  GOptionContext *ctx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;
  guint i;
  OTAInfo *otaInfo = NULL;

  ctx = g_option_context_new ("Nvidia DeepStream Test5");
  group = g_option_group_new ("abc", NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (ctx, group);
  g_option_context_add_group (ctx, gst_init_get_option_group ());

  GST_DEBUG_CATEGORY_INIT (NVDS_APP, "NVDS_APP", 0, NULL);

  if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    g_print ("%s",g_option_context_get_help (ctx, TRUE, NULL));
    return -1;
  }

  if (print_version) {
    g_print ("deepstream-test5-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    return 0;
  }

  if (print_dependencies_version) {
    g_print ("deepstream-test5-app version %d.%d.%d\n",
        NVDS_APP_VERSION_MAJOR, NVDS_APP_VERSION_MINOR, NVDS_APP_VERSION_MICRO);
    return 0;
  }

  if (cfg_files) {
    num_instances = g_strv_length (cfg_files);
  }
  if (input_files) {
    num_input_files = g_strv_length (input_files);
  }

  if (!cfg_files || num_instances == 0) {
    NVGSTDS_ERR_MSG_V ("Specify config file with -c option");
    return_value = -1;
    goto done;
  }

  for (i = 0; i < num_instances; i++) {
    appCtx[i] = (AppCtx *) g_malloc0 (sizeof (AppCtx));
    appCtx[i]->person_class_id = -1;
    appCtx[i]->car_class_id = -1;
    appCtx[i]->index = i;
    appCtx[i]->active_source_index = -1;
    if (show_bbox_text) {
      appCtx[i]->show_bbox_text = TRUE;
    }

    if (input_files && input_files[i]) {
      appCtx[i]->config.multi_source_config[0].uri =
          g_strdup_printf ("file://%s", input_files[i]);
      g_free (input_files[i]);
    }

    /* Initialize msgapi EARLY - before config parsing
     * This allows error reporting even if config parsing fails */
    if (!msgapi_init_early (appCtx[i], cfg_files[i])) {
      g_print("** INFO: Early Message API initialization skipped or failed - error propagation may not be available\n");
    }

    /* Clear previous error messages before parsing */
    if (g_nvds_last_error_message) {
      g_free(g_nvds_last_error_message);
      g_nvds_last_error_message = NULL;
    }

    if(IS_YAML(cfg_files[i])) {
      if (!parse_config_file_yaml (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        // Capture the detailed error from the global error buffer
        if (g_nvds_last_error_message) {
          appCtx[i]->last_error = g_strdup(g_nvds_last_error_message);
          g_free(g_nvds_last_error_message);
          g_nvds_last_error_message = NULL;
        }
        appCtx[i]->return_value = -1;
        goto done;
      }
    } else {
      if (!parse_config_file (&appCtx[i]->config, cfg_files[i])) {
        NVGSTDS_ERR_MSG_V ("Failed to parse config file '%s'", cfg_files[i]);
        // Capture the detailed error from the global error buffer
        if (g_nvds_last_error_message) {
          appCtx[i]->last_error = g_strdup(g_nvds_last_error_message);
          g_free(g_nvds_last_error_message);
          g_nvds_last_error_message = NULL;
        }
        appCtx[i]->return_value = -1;
        goto done;
      }
    }

    if (override_cfg_file && override_cfg_file[i]) {
      if (!g_file_test (override_cfg_file[i],
            (GFileTest)(G_FILE_TEST_IS_REGULAR | G_FILE_TEST_IS_SYMLINK)))
      {
        g_print ("Override file %s does not exist, quitting...\n",
            override_cfg_file[i]);
        appCtx[i]->return_value = -1;
        goto done;
      }
      otaInfo = (OTAInfo *) g_malloc0 (sizeof (OTAInfo));
      otaInfo->appCtx = appCtx[i];
      otaInfo->override_cfg_file = override_cfg_file[i];
      appCtx[i]->ota_handler_thread = g_thread_new ("ota-handler-thread",
          ota_handler_thread, otaInfo);
    }
  }

  for (i = 0; i < num_instances; i++) {
    for (guint j = 0; j < appCtx[i]->config.num_source_sub_bins; j++) {
       /** Force the source (applicable only if RTSP)
        * to use TCP for RTP/RTCP channels.
        * forcing TCP to avoid problems with UDP port usage from within docker-
        * container.
        * The UDP RTCP channel when run within docker had issues receiving
        * RTCP Sender Reports from server
        */
      if (force_tcp)
        appCtx[i]->config.multi_source_config[j].select_rtp_protocol = 0x04;
    }
    if (!create_pipeline (appCtx[i], bbox_generated_probe_after_analytics,
            NULL, perf_cb, overlay_graphics)) {
      NVGSTDS_ERR_MSG_V ("Failed to create pipeline");
      return_value = -1;
      goto done;
    }
    frame_audit_attach_pad (
        appCtx[i]->pipeline.multi_src_bin.streammux, "nvstreammux");
    frame_audit_attach_pad (
        appCtx[i]->pipeline.common_elements.primary_gie_bin.primary_gie,
        "pgie");
    frame_audit_attach_pad (
        appCtx[i]->pipeline.common_elements.tracker_bin.tracker, "tracker");
    preview_attach (appCtx[i]);
    source_health_configure (appCtx[i]);
    if (!pn263_bbox_correction_attach (
            appCtx[i]->pipeline.common_elements.primary_gie_bin.primary_gie,
            appCtx[i]->pipeline.common_elements.tracker_bin.tracker)) {
      NVGSTDS_ERR_MSG_V ("Failed to attach experiment bbox correction probes");
      return_value = -1;
      goto done;
    }
    /** Now add probe to RTPSession plugin src pad */
    for (guint j = 0; j < appCtx[i]->pipeline.multi_src_bin.num_bins; j++) {
      testAppCtx->streams[j].id = j;
    }
    /** In test5 app, as we could have several sources connected
     * for a typical IoT use-case, raising the nvstreammux's
     * buffer-pool-size to 16 */
    g_object_set (appCtx[i]->pipeline.multi_src_bin.streammux,
        "buffer-pool-size", STREAMMUX_BUFFER_POOL_SIZE, NULL);
  }

  if (num_instances > 0 &&
      !mv3dt_crop_encoder_init (appCtx[0]->config.global_gpu_id)) {
    NVGSTDS_ERR_MSG_V ("Failed to initialize in-pipeline crop encoder");
    return_value = -1;
    goto done;
  }

  main_loop = g_main_loop_new (NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, NULL);

  g_mutex_init (&disp_lock);
  display = XOpenDisplay (NULL);
  for (i = 0; i < num_instances; i++) {
    guint j;

    if (!show_bbox_text) {
      GstElement *nvosd = appCtx[i]->pipeline.instance_bins[0].osd_bin.nvosd;
      if (nvosd) {
        g_object_set(G_OBJECT(nvosd), "display-text", FALSE, NULL);
      }
    }
#if defined(__aarch64__)
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
            GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
        return_value = -1;
        goto done;
      }
#endif
    for (j = 0; j < appCtx[i]->config.num_sink_sub_bins; j++) {
      XTextProperty xproperty;
      gchar *title;
      guint width, height;
      XSizeHints hints = {0};

      if (!GST_IS_VIDEO_OVERLAY (appCtx[i]->pipeline.instance_bins[0].sink_bin.
              sub_bins[j].sink)) {
        continue;
      }

      if (!display) {
        NVGSTDS_ERR_MSG_V ("Could not open X Display");
        return_value = -1;
        goto done;
      }

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width)
        width =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.width;
      else
        width = appCtx[i]->config.tiled_display_config.width;

      if (appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height)
        height =
            appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.height;
      else
        height = appCtx[i]->config.tiled_display_config.height;

      width = (width) ? width : DEFAULT_X_WINDOW_WIDTH;
      height = (height) ? height : DEFAULT_X_WINDOW_HEIGHT;

      hints.flags = PPosition | PSize;
      hints.x = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_x;
      hints.y = appCtx[i]->config.sink_bin_sub_bin_config[j].render_config.offset_y;
      hints.width = width;
      hints.height = height;

      windows[i] =
          XCreateSimpleWindow (display, RootWindow (display,
              DefaultScreen (display)), hints.x, hints.y, width, height, 2,
              0x00000000, 0x00000000);

      XSetNormalHints(display, windows[i], &hints);

      if (num_instances > 1)
        title = g_strdup_printf (APP_TITLE "-%d", i);
      else
        title = g_strdup (APP_TITLE);
      if (XStringListToTextProperty ((char **) &title, 1, &xproperty) != 0) {
        XSetWMName (display, windows[i], &xproperty);
        XFree (xproperty.value);
      }

      XSetWindowAttributes attr = { 0 };
      if ((appCtx[i]->config.tiled_display_config.enable &&
              appCtx[i]->config.tiled_display_config.rows *
              appCtx[i]->config.tiled_display_config.columns == 1) ||
          (appCtx[i]->config.tiled_display_config.enable == 0)) {
        attr.event_mask = KeyRelease;
      } else if (appCtx[i]->config.tiled_display_config.enable) {
        attr.event_mask = ButtonPress | KeyRelease;
      }
      XChangeWindowAttributes (display, windows[i], CWEventMask, &attr);

      Atom wmDeleteMessage = XInternAtom (display, "WM_DELETE_WINDOW", False);
      if (wmDeleteMessage != None) {
        XSetWMProtocols (display, windows[i], &wmDeleteMessage, 1);
      }
      XMapRaised (display, windows[i]);
      XSync (display, 1);       //discard the events for now
      gst_video_overlay_set_window_handle (GST_VIDEO_OVERLAY (appCtx
              [i]->pipeline.instance_bins[0].sink_bin.sub_bins[j].sink),
          (gulong) windows[i]);
      gst_video_overlay_expose (GST_VIDEO_OVERLAY (appCtx[i]->pipeline.
              instance_bins[0].sink_bin.sub_bins[j].sink));
      if (!x_event_thread)
        x_event_thread = g_thread_new ("nvds-window-event-thread",
            nvds_x_event_thread, NULL);
    }
#if !defined(__aarch64__)
    int is_nvgpu = 0;
    NvBufSurfaceDeviceInfo dev_info;
    if (NvBufSurfaceGetDeviceInfo(&dev_info) == 0) {
      if (dev_info.driverType == NVBUF_DRIVER_TYPE_NVGPU) {
        is_nvgpu = 1;
      }
    }

    if (!is_nvgpu) {
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PAUSED) == GST_STATE_CHANGE_FAILURE) {
        NVGSTDS_ERR_MSG_V ("Failed to set pipeline to PAUSED");
        return_value = -1;
        goto done;
      }
    }
#endif
  }

  /* Dont try to set playing state if error is observed */
  if (return_value != -1) {
    for (i = 0; i < num_instances; i++) {
      if (gst_element_set_state (appCtx[i]->pipeline.pipeline,
              GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {

        g_print ("\ncan't set pipeline to playing state.\n");
        return_value = -1;
        goto done;
      }
    }
  }

  if (source_health_enabled && !source_health_watchdog_id)
    source_health_watchdog_id = g_timeout_add (1000, source_health_watchdog_cb, appCtx[0]);

  print_runtime_commands ();

  changemode (1);

  g_timeout_add (40, event_thread_func, NULL);
  g_main_loop_run (main_loop);

  changemode (0);
  preview_close ();
  source_health_close ();

done:
  mv3dt_crop_encoder_shutdown ();

  g_print ("Quitting\n");

  /* GENERIC error reporting via msgapi - catches ALL failures
   * Check each appCtx individually since return_value might not be set yet */
  for (i = 0; i < num_instances; i++) {
    if (appCtx[i] && appCtx[i]->return_value == -1) {
      // Use the captured error message if available, otherwise construct a generic one
      gchar *error_msg = NULL;
      if (appCtx[i]->last_error) {
        error_msg = g_strdup(appCtx[i]->last_error);
      } else if (i < num_instances && cfg_files && cfg_files[i]) {
        error_msg = g_strdup_printf("Application failed to start with config file: %s", cfg_files[i]);
      }

      msgapi_report_error_and_cleanup(appCtx[i], error_msg);

      if (error_msg) {
        g_free(error_msg);
      }
      if (appCtx[i]->last_error) {
        g_free(appCtx[i]->last_error);
        appCtx[i]->last_error = NULL;
      }
    }
  }

  for (i = 0; i < num_instances; i++) {
    if (appCtx[i] == NULL)
      continue;

    if (appCtx[i]->return_value == -1)
      return_value = -1;

    destroy_pipeline (appCtx[i]);

    if (appCtx[i]->ota_handler_thread && override_cfg_file[i]) {
      inotify_rm_watch (appCtx[i]->ota_inotify_fd, appCtx[i]->ota_watch_desc);
      g_thread_join (appCtx[i]->ota_handler_thread);
    }

    g_mutex_lock (&disp_lock);
    if (windows[i])
      XDestroyWindow (display, windows[i]);
    windows[i] = 0;
    g_mutex_unlock (&disp_lock);

    g_free (appCtx[i]);
  }

  g_mutex_lock (&disp_lock);
  if (display)
    XCloseDisplay (display);
  display = NULL;
  g_mutex_unlock (&disp_lock);
  g_mutex_clear (&disp_lock);

  if (main_loop) {
    g_main_loop_unref (main_loop);
  }

  if (ctx) {
    g_option_context_free (ctx);
  }

  if (return_value == 0) {
    g_print ("App run successful\n");
  } else {
    g_print ("App run failed\n");
  }

  gst_deinit ();

  return return_value;

  g_free (testAppCtx);

  return 0;
}

static gchar *
get_first_result_label (NvDsClassifierMeta * classifierMeta)
{
  GList *n;
  for (n = classifierMeta->label_info_list; n != NULL; n = n->next) {
    NvDsLabelInfo *labelInfo = (NvDsLabelInfo *) (n->data);
    if (labelInfo->result_label[0] != '\0') {
      return g_strdup (labelInfo->result_label);
    }
  }
  return NULL;
}

static void
schema_fill_sample_sgie_vehicle_metadata (NvDsObjectMeta * obj_params,
    NvDsVehicleObject * obj)
{
  if (!obj_params || !obj) {
    return;
  }

  /** The JSON obj->classification, say type, color, or make
   * according to the schema shall have null (unknown)
   * classifications (if the corresponding sgie failed to provide a label)
   */
  obj->type = NULL;
  obj->make = NULL;
  obj->model = NULL;
  obj->color = NULL;
  obj->license = NULL;
  obj->region = NULL;

  GList *l;
  for (l = obj_params->classifier_meta_list; l != NULL; l = l->next) {
    NvDsClassifierMeta *classifierMeta = (NvDsClassifierMeta *) (l->data);
    switch (classifierMeta->unique_component_id) {
      case SECONDARY_GIE_VEHICLE_TYPE_UNIQUE_ID:
        obj->type = get_first_result_label (classifierMeta);
        break;
      case SECONDARY_GIE_VEHICLE_COLOR_UNIQUE_ID:
        obj->color = get_first_result_label (classifierMeta);
        break;
      case SECONDARY_GIE_VEHICLE_MAKE_UNIQUE_ID:
        obj->make = get_first_result_label (classifierMeta);
        break;
      default:
        break;
    }
  }
}
