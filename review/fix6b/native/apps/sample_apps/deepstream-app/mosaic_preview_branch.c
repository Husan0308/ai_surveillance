/* Optional post-OSD DeepStream mosaic -> existing V11 shared-memory protocol.
 * Compiled only into the isolated DeepStream 9.1 mosaic viewer build. */
#define _GNU_SOURCE
#include "mosaic_preview_branch.h"
#include "six_live_monitor.h"

#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MOSAIC_WIDTH 1920u
#define MOSAIC_HEIGHT 720u
#define MOSAIC_STRIDE (MOSAIC_WIDTH * 4u)
#define MOSAIC_PAYLOAD ((gsize) MOSAIC_STRIDE * MOSAIC_HEIGHT)
#define MOSAIC_COLUMNS 3u
#define MOSAIC_ROWS 2u
#define MOSAIC_TILE_WIDTH (MOSAIC_WIDTH / MOSAIC_COLUMNS)
#define MOSAIC_TILE_HEIGHT (MOSAIC_HEIGHT / MOSAIC_ROWS)
#define MOSAIC_LABEL_COUNT 6u
#define IPC_HEADER_SIZE 104u
#define IPC_FILE_SIZE (IPC_HEADER_SIZE + MOSAIC_PAYLOAD)
#define IPC_PATH "/run/mosaic-preview/preview-v1.bin"
#define IPC_LOCK_PATH "/run/mosaic-preview/preview-v1.producer-lock"
#define FULLSCREEN_WIDTH 1920u
#define FULLSCREEN_HEIGHT 1080u
#define FULLSCREEN_STRIDE (FULLSCREEN_WIDTH * 4u)
#define FULLSCREEN_PAYLOAD ((gsize) FULLSCREEN_STRIDE * FULLSCREEN_HEIGHT)
#define FULLSCREEN_HEADER_SIZE 65536u
#define FULLSCREEN_FILE_SIZE (FULLSCREEN_HEADER_SIZE + FULLSCREEN_PAYLOAD)
#define FULLSCREEN_PATH "/run/mosaic-preview/fullscreen-v1.bin"
#define FULLSCREEN_LOCK_PATH "/run/mosaic-preview/fullscreen-v1.producer-lock"
#define FULLSCREEN_REQUEST_PATH "/run/mosaic-preview/fullscreen-request-v1"
#define FULLSCREEN_IDENTITY_QUEUE_CAPACITY 8u
#define FULLSCREEN_BATCH_COPY_QUEUE_CAPACITY 2u
#define FULLSCREEN_METADATA_OBJECT_LIMIT 24u
#define LATENCY_RING_SIZE 128u
#define MOSAIC_IDENTITY_RING_SIZE 64u

typedef struct {
  guint source_id;
  guint frame_num;
  guint64 source_pts;
  guint64 source_generation;
  gboolean source_pts_valid;
  gboolean source_generation_valid;
} MosaicSourceIdentity;

typedef struct {
  gboolean valid;
  GstClockTime composite_pts;
  guint source_count;
  MosaicSourceIdentity sources[MOSAIC_LABEL_COUNT];
} MosaicFrameIdentity;

typedef struct {
  guint source_id;
  guint32 source_frame_num;
  guint64 source_pts;
  guint64 source_frame_sequence;
  guint selection_generation;
  GstClockTime branch_pts;
  guint32 object_count;
  guint source_frame_width;
  guint source_frame_height;
  guint pipeline_width;
  guint pipeline_height;
  gboolean source_pts_valid;
  gchar *source_objects_json;
  gchar *fullscreen_objects_json;
  gboolean fullscreen_metadata_matched;
  guint fullscreen_metadata_generation;
  guint fullscreen_object_count;
  guint fullscreen_frame_object_count;
  guint fullscreen_num_frames_in_batch;
  guint fullscreen_meta_source_id;
  gint fullscreen_meta_frame_num;
  guint64 fullscreen_meta_frame_pts;
  gboolean fullscreen_meta_pts_valid;
  guint fullscreen_meta_pad_index;
} FullscreenFrameIdentity;

typedef struct {
  GstBuffer *original; /* Held alive by the copy's GstParentBufferMeta. */
  GstBuffer *copy;
  guint source_id;
  guint selection_generation;
} FullscreenBatchCopy;

typedef enum {
  FULLSCREEN_COPY_OK = 0,
  FULLSCREEN_COPY_SOURCE_BUFFER_MISSING,
  FULLSCREEN_COPY_SOURCE_PTS_INVALID,
  FULLSCREEN_COPY_SOURCE_BATCH_META_MISSING,
  FULLSCREEN_COPY_SOURCE_BATCH_META_DUPLICATE,
  FULLSCREEN_COPY_SOURCE_META_CALLBACK_MISSING,
  FULLSCREEN_COPY_SOURCE_META_RELEASE_MISSING,
  FULLSCREEN_COPY_SOURCE_META_ACCESSOR_MISMATCH,
  FULLSCREEN_COPY_SOURCE_BATCH_CAPACITY_ZERO,
  FULLSCREEN_COPY_SOURCE_BATCH_EXCEEDS_CAPACITY,
  FULLSCREEN_COPY_SOURCE_MEMORY_MISSING,
  FULLSCREEN_COPY_GST_BUFFER_COPY_FAILED,
  FULLSCREEN_COPY_COPIED_BATCH_META_MISSING,
  FULLSCREEN_COPY_COPIED_BATCH_META_DUPLICATE,
  FULLSCREEN_COPY_COPIED_META_CALLBACK_CHANGED,
  FULLSCREEN_COPY_COPIED_META_RELEASE_CHANGED,
  FULLSCREEN_COPY_COPIED_BATCH_META_ALIAS,
  FULLSCREEN_COPY_COPIED_FRAME_COUNT_CHANGED,
  FULLSCREEN_COPY_COPIED_PTS_CHANGED,
  FULLSCREEN_COPY_COPIED_MEMORY_COUNT_CHANGED,
  FULLSCREEN_COPY_VIDEO_MEMORY_NOT_SHARED,
  FULLSCREEN_COPY_FRAME_LIST_LENGTH_CHANGED,
  FULLSCREEN_COPY_SOURCE_FRAME_LIST_COUNT_MISMATCH,
  FULLSCREEN_COPY_COPIED_FRAME_LIST_COUNT_MISMATCH,
  FULLSCREEN_COPY_FRAME_META_MISSING,
  FULLSCREEN_COPY_FRAME_META_ALIAS,
  FULLSCREEN_COPY_FRAME_SOURCE_ID_CHANGED,
  FULLSCREEN_COPY_FRAME_PAD_INDEX_CHANGED,
  FULLSCREEN_COPY_FRAME_BATCH_ID_CHANGED,
  FULLSCREEN_COPY_FRAME_NUMBER_CHANGED,
  FULLSCREEN_COPY_FRAME_PTS_CHANGED,
  FULLSCREEN_COPY_FRAME_NTP_TIMESTAMP_CHANGED,
  FULLSCREEN_COPY_FRAME_GEOMETRY_CHANGED,
  FULLSCREEN_COPY_FRAME_OBJECT_COUNT_CHANGED,
  FULLSCREEN_COPY_SOURCE_OBJECT_LIST_COUNT_MISMATCH,
  FULLSCREEN_COPY_COPIED_OBJECT_LIST_COUNT_MISMATCH,
  FULLSCREEN_COPY_OBJECT_LIST_LENGTH_CHANGED,
  FULLSCREEN_COPY_OBJECT_META_MISSING,
  FULLSCREEN_COPY_OBJECT_META_ALIAS,
  FULLSCREEN_COPY_OBJECT_CORRESPONDENCE_MISSING,
  FULLSCREEN_COPY_OBJECT_CORRESPONDENCE_AMBIGUOUS,
  FULLSCREEN_COPY_OBJECT_MATCH_ALLOCATION_FAILED,
  FULLSCREEN_COPY_OBJECT_ID_CHANGED,
  FULLSCREEN_COPY_OBJECT_CLASS_ID_CHANGED,
  FULLSCREEN_COPY_OBJECT_COMPONENT_ID_CHANGED,
  FULLSCREEN_COPY_DETECTOR_CONFIDENCE_CHANGED,
  FULLSCREEN_COPY_TRACKER_CONFIDENCE_CHANGED,
  FULLSCREEN_COPY_OBJECT_RECT_CHANGED,
  FULLSCREEN_COPY_DETECTOR_RECT_CHANGED,
  FULLSCREEN_COPY_TRACKER_RECT_CHANGED,
  FULLSCREEN_COPY_DISPLAY_TEXT_ALIAS,
  FULLSCREEN_COPY_DISPLAY_TEXT_CONTENT_CHANGED,
  FULLSCREEN_COPY_PARENT_META_ATTACH_FAILED,
  FULLSCREEN_COPY_FAILURE_COUNT
} FullscreenCopyFailure;

typedef enum {
  FULLSCREEN_GEOMETRY_SOURCE_WIDTH_CHANGED = 1u << 0,
  FULLSCREEN_GEOMETRY_SOURCE_HEIGHT_CHANGED = 1u << 1,
  FULLSCREEN_GEOMETRY_PIPELINE_WIDTH_CHANGED = 1u << 2,
  FULLSCREEN_GEOMETRY_PIPELINE_HEIGHT_CHANGED = 1u << 3
} FullscreenGeometryMismatchField;

typedef struct {
  gboolean valid;
  guint mismatch_fields;
  guint selection_generation;
  gpointer source_buffer_identity;
  guint64 buffer_pts;
  guint frame_source_id;
  gint frame_num;
  guint batch_id;
  guint original_source_frame_width;
  guint original_source_frame_height;
  guint original_pipeline_width;
  guint original_pipeline_height;
  guint copied_source_frame_width;
  guint copied_source_frame_height;
  guint copied_pipeline_width;
  guint copied_pipeline_height;
  guint64 frame_pts;
} FullscreenGeometryMismatch;

typedef struct {
  gboolean valid;
  gboolean copied_position_valid;
  guint selection_generation;
  gpointer source_buffer_identity;
  guint64 buffer_pts;
  guint source_id;
  gint frame_num;
  guint batch_id;
  guint64 frame_pts;
  guint original_position;
  guint copied_position;
  guint64 original_object_id;
  guint64 copied_object_id;
} FullscreenObjectMismatch;

typedef struct {
  AppCtx *app_ctx;
  GstElement *tee;
  GstElement *queue;
  GstElement *converter;
  GstElement *caps_filter;
  GstElement *app_sink;
  GstElement *fullscreen_queue;
  GstElement *fullscreen_valve;
  GstElement *fullscreen_tiler;
  GstElement *fullscreen_convert;
  GstElement *fullscreen_caps_filter;
  GstElement *fullscreen_app_sink;
  NvDsOSDBin fullscreen_osd_bin;
  GstPad *tee_main_pad;
  GstPad *tee_preview_pad;
  GstPad *queue_sink_pad;
  GstPad *fullscreen_tee_pad;
  GstPad *fullscreen_queue_sink_pad;
  GstPad *fullscreen_valve_sink_pad;
  GstPad *fullscreen_tiler_sink_pad;
  GstPad *fullscreen_analytics_sink_pad;
  GstPad *fullscreen_osd_src_pad;
  GstPad *sink_bin_pad;
  GstPad *osd_src_pad;
  GstPad *osd_sink_pad;
  gulong label_probe_id;
  gulong fullscreen_label_probe_id;
  gulong fullscreen_osd_src_probe_id;
  guint stats_timer_id;
  guint fullscreen_request_timer_id;
  int fd;
  int producer_lock_fd;
  gboolean owns_producer_lock;
  guint8 *mapping;
  gsize mapping_size;
  guint rate_fps;
  guint64 sequence;
  gint64 last_publish_us;
  gint64 stats_previous_us;
  guint64 stats_previous_published;
  guint64 published;
  guint64 rate_drops;
  guint64 queue_overruns;
  guint64 lock_drops;
  guint64 ipc_failures;
  guint latency_count;
  guint latency_next;
  guint64 latency_ns[LATENCY_RING_SIZE];
  gboolean label_layout_logged;
  int fullscreen_fd;
  int fullscreen_lock_fd;
  gboolean owns_fullscreen_lock;
  guint8 *fullscreen_mapping;
  gsize fullscreen_mapping_size;
  guint64 fullscreen_sequence;
  GQueue fullscreen_identities;
  GQueue fullscreen_batch_copies;
  guint64 fullscreen_source_frame_sequence;
  guint64 fullscreen_last_source_pts;
  gint fullscreen_last_source_frame_num;
  guint fullscreen_last_source_id;
  gboolean fullscreen_have_source_identity;
  gint fullscreen_selected_source;
  gint fullscreen_pending_source;
  gint fullscreen_tiler_primed_source;
  gint fullscreen_prime_requested_source;
  gint fullscreen_selection_generation;
  GstClockTime fullscreen_prime_input_pts;
  gboolean fullscreen_prime_input_pts_valid;
  gint fullscreen_pending_misses;
  guint64 fullscreen_batch_copy_queue_drops;
  guint64 fullscreen_batch_copy_generation_misses;
  guint64 fullscreen_batch_copy_identity_misses;
  guint64 fullscreen_batch_copy_source_misses;
  gint fullscreen_input_metadata_logged;
  gint fullscreen_selection_input_logged;
  gulong fullscreen_input_probe_id;
  gulong fullscreen_tiler_input_probe_id;
  gulong fullscreen_analytics_probe_id;
  gint fullscreen_copy_failure_logged[FULLSCREEN_COPY_FAILURE_COUNT];
  gint fullscreen_copy_miss_logged;
  gint fullscreen_copy_queue_drop_logged;
  gint fullscreen_copy_selection_race_logged;
  gint fullscreen_active_missing_logged;
  gint fullscreen_tiler_input_logged;
  gint fullscreen_prime_mismatch_logged;
  gint fullscreen_identity_fail_logged;
  gint fullscreen_tiler_output_logged;
  gint fullscreen_metadata_logged;
  gint fullscreen_metadata_fail_logged;
  gint fullscreen_label_logged;
  gint fullscreen_osd_output_logged;
  gint fullscreen_appsink_logged;
  gint fullscreen_publication_logged;
  guint64 fullscreen_input_matches;
  guint64 fullscreen_tiler_inputs;
  guint64 fullscreen_tiler_outputs;
  guint64 fullscreen_osd_outputs;
  guint64 fullscreen_appsink_samples;
  guint64 fullscreen_correlation_failures;
  guint64 fullscreen_publications;
  gint64 fullscreen_last_publish_us;
  gdouble fullscreen_fps_ema;
  MosaicFrameIdentity mosaic_identities[MOSAIC_IDENTITY_RING_SIZE];
  guint mosaic_identity_next;
  guint64 mosaic_identity_matches;
  guint64 mosaic_identity_misses;
  GMutex mutex;
} MosaicPreview;

static MosaicPreview *active_preview;
static const gchar *const camera_labels[MOSAIC_LABEL_COUNT] = {
  "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"
};

static void
remember_mosaic_frame_identity (MosaicPreview *preview, GstBuffer *buffer,
    NvDsBatchMeta *batch)
{
  if (!buffer || !batch || !GST_BUFFER_PTS_IS_VALID (buffer))
    return;
  MosaicFrameIdentity identity;
  memset (&identity, 0, sizeof identity);
  identity.valid = TRUE;
  identity.composite_pts = GST_BUFFER_PTS (buffer);
  guint seen = 0;
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    NvDsFrameMeta *meta = node->data;
    if (!meta || meta->source_id >= MOSAIC_LABEL_COUNT || meta->pad_index != meta->source_id)
      continue;
    guint bit = 1u << meta->source_id;
    if (seen & bit) {
      identity.valid = FALSE;
      break;
    }
    seen |= bit;
    MosaicSourceIdentity *source = &identity.sources[identity.source_count++];
    source->source_id = meta->source_id;
    source->frame_num = meta->frame_num;
    source->source_pts_valid = GST_CLOCK_TIME_IS_VALID (meta->buf_pts);
    source->source_pts = source->source_pts_valid ? meta->buf_pts : 0;
    source->source_generation_valid = source->source_pts_valid &&
        six_live_preview_frame_generation (meta, &source->source_generation);
  }
  if (!identity.source_count)
    identity.valid = FALSE;
  g_mutex_lock (&preview->mutex);
  preview->mosaic_identities[preview->mosaic_identity_next++ % MOSAIC_IDENTITY_RING_SIZE] = identity;
  g_mutex_unlock (&preview->mutex);
}

static void
note_published_mosaic_sources (MosaicPreview *preview, GstBuffer *buffer)
{
  if (!buffer || !GST_BUFFER_PTS_IS_VALID (buffer))
    return;
  MosaicSourceIdentity sources[MOSAIC_LABEL_COUNT];
  guint source_count = 0;
  gboolean matched = FALSE;
  GstClockTime pts = GST_BUFFER_PTS (buffer);
  g_mutex_lock (&preview->mutex);
  for (guint i = 0; i < MOSAIC_IDENTITY_RING_SIZE; i++) {
    MosaicFrameIdentity *identity = &preview->mosaic_identities[i];
    if (identity->valid && identity->composite_pts == pts) {
      source_count = identity->source_count;
      memcpy (sources, identity->sources, source_count * sizeof sources[0]);
      identity->valid = FALSE;
      preview->mosaic_identity_matches++;
      matched = TRUE;
      break;
    }
  }
  if (!matched)
    preview->mosaic_identity_misses++;
  g_mutex_unlock (&preview->mutex);
  if (!matched)
    return; /* No per-source freshness is inferred from a composite alone. */
  for (guint i = 0; i < source_count; i++) {
    if (sources[i].source_pts_valid && sources[i].source_generation_valid)
      six_live_preview_frame (sources[i].source_id, sources[i].frame_num,
          sources[i].source_pts, sources[i].source_generation);
  }
}

static gchar *make_fullscreen_snapshot (
    const FullscreenFrameIdentity *identity);

static guint64
monotonic_ns (void)
{
  struct timespec now;
  if (clock_gettime (CLOCK_MONOTONIC, &now) != 0)
    return 0;
  return (guint64) now.tv_sec * 1000000000ULL + (guint64) now.tv_nsec;
}

static void
put_le32 (guint8 *out, guint32 value)
{
  for (guint i = 0; i < 4; i++)
    out[i] = (guint8) (value >> (i * 8));
}

static void
put_le64 (guint8 *out, guint64 value)
{
  for (guint i = 0; i < 8; i++)
    out[i] = (guint8) (value >> (i * 8));
}

static int
compare_u64 (const void *left, const void *right)
{
  guint64 a = *(const guint64 *) left;
  guint64 b = *(const guint64 *) right;
  return (a > b) - (a < b);
}

/* Header-v4 bytes exactly match HEADER_V4/HEADER_SIZE in ui_preview_ipc_v1.py. */
static void
write_header (MosaicPreview *preview, guint64 timestamp_ns, guint fps_milli)
{
  guint8 *header = preview->mapping;
  memset (header, 0, IPC_HEADER_SIZE);
  memcpy (header, "V11UI01\0", 8);
  put_le32 (header + 8, 4);
  put_le64 (header + 12, preview->sequence);
  put_le64 (header + 20, timestamp_ns);
  put_le32 (header + 28, MOSAIC_WIDTH);
  put_le32 (header + 32, MOSAIC_HEIGHT);
  put_le32 (header + 36, MOSAIC_STRIDE);
  put_le32 (header + 40, MOSAIC_PAYLOAD);
  put_le32 (header + 44, 0); /* Composite object counts have no single-source meaning. */
  put_le32 (header + 48, fps_milli);
  /* v4 timing/source fields are intentionally zero: a mosaic has six PTSs. */
}

static gboolean
validate_slot (MosaicPreview *preview)
{
  struct stat info;
  if (fstat (preview->fd, &info) != 0 || !S_ISREG (info.st_mode) ||
      info.st_uid != getuid () || (info.st_mode & 077) != 0 ||
      info.st_size != (off_t) IPC_FILE_SIZE)
    return FALSE;
  preview->mapping_size = IPC_FILE_SIZE;
  preview->mapping = mmap (NULL, preview->mapping_size, PROT_READ | PROT_WRITE,
      MAP_SHARED, preview->fd, 0);
  return preview->mapping != MAP_FAILED;
}

static gboolean
publish_frame (MosaicPreview *preview, const guint8 *plane, gint source_stride,
    gsize plane_offset, gsize buffer_size, guint64 pts, guint64 copy_start_ns)
{
  const gsize row_bytes = MOSAIC_STRIDE;
  gsize stride, last_row, end_offset;
  guint64 timestamp_ns;
  gint lock_result;

  if (source_stride <= 0 || (gsize) source_stride < row_bytes || !plane ||
      MOSAIC_HEIGHT - 1 > (G_MAXSIZE - row_bytes) / (gsize) source_stride)
    goto invalid;
  stride = (gsize) source_stride;
  last_row = ((gsize) MOSAIC_HEIGHT - 1) * stride;
  if (last_row > G_MAXSIZE - row_bytes)
    goto invalid;
  end_offset = plane_offset + last_row + row_bytes;
  if (end_offset < plane_offset || end_offset > buffer_size)
    goto invalid;

  lock_result = flock (preview->fd, LOCK_EX | LOCK_NB);
  if (lock_result != 0) {
    g_mutex_lock (&preview->mutex);
    if (errno == EWOULDBLOCK || errno == EAGAIN)
      preview->lock_drops++;
    else
      preview->ipc_failures++;
    g_mutex_unlock (&preview->mutex);
    return FALSE;
  }

  guint8 *destination = preview->mapping + IPC_HEADER_SIZE;
  for (guint row = 0; row < MOSAIC_HEIGHT; row++)
    memcpy (destination + (gsize) row * row_bytes,
        plane + (gsize) row * stride, row_bytes);
  __sync_synchronize ();
  timestamp_ns = monotonic_ns ();

  gint64 now_us = g_get_monotonic_time ();
  g_mutex_lock (&preview->mutex);
  preview->sequence++;
  preview->published++;
  gdouble fps = 0.0;
  if (preview->stats_previous_us > 0 && now_us > preview->stats_previous_us)
    fps = (preview->published - preview->stats_previous_published) * 1000000.0 /
        (now_us - preview->stats_previous_us);
  write_header (preview, timestamp_ns, (guint) MAX (0, (gint) round (fps * 1000.0)));
  guint slot = preview->latency_next++ % LATENCY_RING_SIZE;
  preview->latency_ns[slot] = timestamp_ns >= copy_start_ns ? timestamp_ns - copy_start_ns : 0;
  preview->latency_count = MIN (preview->latency_count + 1, LATENCY_RING_SIZE);
  preview->last_publish_us = now_us;
  g_mutex_unlock (&preview->mutex);
  if (flock (preview->fd, LOCK_UN) != 0) {
    g_mutex_lock (&preview->mutex);
    preview->ipc_failures++;
    g_mutex_unlock (&preview->mutex);
    return FALSE;
  }
  (void) pts; /* Mosaic PTS is deliberately not presented as a source timestamp. */
  return TRUE;

invalid:
  g_mutex_lock (&preview->mutex);
  preview->ipc_failures++;
  g_mutex_unlock (&preview->mutex);
  return FALSE;
}

static GstFlowReturn
new_sample (GstAppSink *sink, gpointer user_data)
{
  MosaicPreview *preview = user_data;
  GstSample *sample = gst_app_sink_pull_sample (sink);
  if (!sample)
    return GST_FLOW_EOS;

  gint64 now_us = g_get_monotonic_time ();
  g_mutex_lock (&preview->mutex);
  gboolean due = preview->last_publish_us == 0 ||
      now_us - preview->last_publish_us >= (gint64) (1000000 / preview->rate_fps);
  if (!due)
    preview->rate_drops++;
  g_mutex_unlock (&preview->mutex);
  if (!due) {
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  guint64 copy_start_ns = monotonic_ns ();

  GstCaps *caps = gst_sample_get_caps (sample);
  GstBuffer *buffer = gst_sample_get_buffer (sample);
  GstVideoInfo info;
  gboolean ok = FALSE;
  gboolean publish_attempted = FALSE;
  if (!caps || !buffer || !gst_video_info_from_caps (&info, caps) ||
      GST_VIDEO_INFO_FORMAT (&info) != GST_VIDEO_FORMAT_BGRx ||
      GST_VIDEO_INFO_WIDTH (&info) != MOSAIC_WIDTH ||
      GST_VIDEO_INFO_HEIGHT (&info) != MOSAIC_HEIGHT ||
      gst_buffer_get_size (buffer) < GST_VIDEO_INFO_SIZE (&info))
    goto done;

  const GstCapsFeatures *features = gst_caps_get_features (caps, 0);
  if (features && gst_caps_features_contains (features, "memory:NVMM"))
    goto done; /* CPU mapping is accepted only after explicit system-memory caps. */

  GstVideoFrame frame;
  memset (&frame, 0, sizeof frame);
  if (!gst_video_frame_map (&frame, &info, buffer, GST_MAP_READ))
    goto done;
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (&frame, 0);
  gsize offset = GST_VIDEO_INFO_PLANE_OFFSET (&info, 0);
  GstVideoMeta *video_meta = gst_buffer_get_video_meta (buffer);
  if (video_meta) {
    if (video_meta->format != GST_VIDEO_FORMAT_BGRx ||
        video_meta->width != MOSAIC_WIDTH || video_meta->height != MOSAIC_HEIGHT ||
        video_meta->n_planes < 1 || video_meta->stride[0] != stride) {
      gst_video_frame_unmap (&frame);
      goto done;
    }
    offset = video_meta->offset[0];
  }
  const guint8 *plane = GST_VIDEO_FRAME_PLANE_DATA (&frame, 0);
  guint64 pts = GST_BUFFER_PTS_IS_VALID (buffer) ? GST_BUFFER_PTS (buffer) : 0;
  publish_attempted = TRUE;
  ok = publish_frame (preview, plane, stride, offset, gst_buffer_get_size (buffer),
      pts, copy_start_ns);
  gst_video_frame_unmap (&frame);
  if (ok)
    note_published_mosaic_sources (preview, buffer);

done:
  if (!ok && !publish_attempted) {
    g_mutex_lock (&preview->mutex);
    preview->ipc_failures++;
    g_mutex_unlock (&preview->mutex);
  }
  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

static void
queue_overrun (GstElement *queue, gpointer user_data)
{
  (void) queue;
  MosaicPreview *preview = user_data;
  g_mutex_lock (&preview->mutex);
  preview->queue_overruns++;
  g_mutex_unlock (&preview->mutex);
}

static GstPadProbeReturn
camera_labels_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  if (!batch || !batch->frame_meta_list)
    return GST_PAD_PROBE_OK;
  /* Source identities were captured before the tiler at the analytics tee. */

  /* nvmultistreamtiler produces one composite video buffer and transformed
   * batch metadata. Attach one canvas-coordinate label pack once per output
   * buffer; do not treat each retained source-frame meta as a separate tile
   * canvas. The app config is checked against the source mapping before this
   * probe is installed. */
  NvDsFrameMeta *composite_frame_meta = NULL;
  guint frame_meta_count = 0;
  guint observed_source_mask = 0;
  gboolean invalid_source_id = FALSE;
  GString *observed = g_string_new (NULL);
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
    if (!frame_meta)
      continue;
    if (!composite_frame_meta)
      composite_frame_meta = frame_meta;
    frame_meta_count++;
    guint source_id = frame_meta->source_id;
    if (source_id >= MOSAIC_LABEL_COUNT) {
      invalid_source_id = TRUE;
      continue;
    }
    observed_source_mask |= 1u << source_id;
    if (observed->len < 96) {
      if (observed->len)
        g_string_append_c (observed, ',');
      g_string_append_printf (observed, "%u/%u", source_id, frame_meta->pad_index);
    }
  }

  if (!preview->label_layout_logged) {
    g_print ("MOSAIC_LABEL_METADATA {\"frame_meta_count\":%u,\"num_frames_in_batch\":%u,"
        "\"observed_source_mask\":%u,\"source_id_pad_index\":[%s]}\n",
        frame_meta_count, batch->num_frames_in_batch, observed_source_mask,
        observed->len ? observed->str : "");
    preview->label_layout_logged = TRUE;
  }
  g_string_free (observed, TRUE);

  if (!composite_frame_meta || invalid_source_id ||
      G_N_ELEMENTS (camera_labels) != MOSAIC_LABEL_COUNT ||
      G_N_ELEMENTS (((NvDsDisplayMeta *) 0)->text_params) < MOSAIC_LABEL_COUNT)
    return GST_PAD_PROBE_OK;

  NvDsDisplayMeta *display = nvds_acquire_display_meta_from_pool (batch);
  if (!display)
    return GST_PAD_PROBE_OK;
  display->num_labels = MOSAIC_LABEL_COUNT;
  for (guint source_id = 0; source_id < MOSAIC_LABEL_COUNT; source_id++) {
    guint column = source_id % MOSAIC_COLUMNS;
    guint row = source_id / MOSAIC_COLUMNS;
    NvOSD_TextParams *label = &display->text_params[source_id];
    label->display_text = g_strdup (camera_labels[source_id]);
    label->x_offset = (gint) (column * MOSAIC_TILE_WIDTH + 12);
    label->y_offset = (gint) (row * MOSAIC_TILE_HEIGHT + 10);
    label->font_params.font_name = "Sans";
    label->font_params.font_size = 18;
    label->font_params.font_color = (NvOSD_ColorParams) { 1.0, 1.0, 1.0, 1.0 };
    label->set_bg_clr = 1;
    label->text_bg_clr = (NvOSD_ColorParams) { 0.0, 0.0, 0.0, 0.8 };
  }
  nvds_add_display_meta_to_frame (composite_frame_meta, display);
  return GST_PAD_PROBE_OK;
}

static gboolean
stats_tick (gpointer user_data)
{
  MosaicPreview *preview = user_data;
  guint64 published, rate_drops, queue_overruns, lock_drops, failures;
  guint64 identity_matches, identity_misses;
  guint64 latency[LATENCY_RING_SIZE];
  guint latency_count;
  gint64 now_us = g_get_monotonic_time ();
  g_mutex_lock (&preview->mutex);
  published = preview->published;
  rate_drops = preview->rate_drops;
  queue_overruns = preview->queue_overruns;
  lock_drops = preview->lock_drops;
  failures = preview->ipc_failures;
  identity_matches = preview->mosaic_identity_matches;
  identity_misses = preview->mosaic_identity_misses;
  latency_count = preview->latency_count;
  memcpy (latency, preview->latency_ns, sizeof latency);
  gdouble fps = preview->stats_previous_us > 0 && now_us > preview->stats_previous_us ?
      (published - preview->stats_previous_published) * 1000000.0 /
      (now_us - preview->stats_previous_us) : 0.0;
  preview->stats_previous_us = now_us;
  preview->stats_previous_published = published;
  g_mutex_unlock (&preview->mutex);

  qsort (latency, latency_count, sizeof latency[0], compare_u64);
  guint64 p50 = latency_count ? latency[(latency_count - 1) / 2] : 0;
  guint64 p95 = latency_count ? latency[(latency_count - 1) * 95 / 100] : 0;
  g_print ("MOSAIC_PREVIEW_STATS {\"publish_fps\":%.3f,\"published\":%" G_GUINT64_FORMAT
      ",\"rate_drops\":%" G_GUINT64_FORMAT ",\"queue_overruns\":%" G_GUINT64_FORMAT
      ",\"ipc_lock_drops\":%" G_GUINT64_FORMAT ",\"ipc_failures\":%" G_GUINT64_FORMAT
      ",\"identity_matches\":%" G_GUINT64_FORMAT ",\"identity_misses\":%" G_GUINT64_FORMAT
      ",\"copy_latency_p50_ns\":%" G_GUINT64_FORMAT ",\"copy_latency_p95_ns\":%" G_GUINT64_FORMAT "}\n",
      fps, published, rate_drops, queue_overruns, lock_drops, failures,
      identity_matches, identity_misses, p50, p95);
  fflush (stdout);
  return G_SOURCE_CONTINUE;
}

static gboolean
valid_rate (guint *rate)
{
  const gchar *value = g_getenv ("V11_UI_MOSAIC_PREVIEW_RATE_FPS");
  if (!value || !*value) {
    *rate = 10;
    return TRUE;
  }
  gchar *end = NULL;
  guint64 parsed = g_ascii_strtoull (value, &end, 10);
  if (!end || *end || parsed < 1 || parsed > 20)
    return FALSE;
  *rate = (guint) parsed;
  return TRUE;
}

static gboolean
open_slot (MosaicPreview *preview)
{
  struct stat lock_info;
  preview->producer_lock_fd = open (IPC_LOCK_PATH,
      O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (preview->producer_lock_fd < 0 || fstat (preview->producer_lock_fd, &lock_info) != 0 ||
      !S_ISREG (lock_info.st_mode) || lock_info.st_uid != getuid () ||
      (lock_info.st_mode & 077) != 0 || flock (preview->producer_lock_fd, LOCK_EX | LOCK_NB) != 0)
    return FALSE;
  preview->owns_producer_lock = TRUE;

  preview->fd = open (IPC_PATH, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  if (preview->fd < 0 || !validate_slot (preview))
    return FALSE;
  memset (preview->mapping, 0, IPC_HEADER_SIZE);
  write_header (preview, 0, 0);
  return TRUE;
}

/* Fullscreen v5 uses the already supported preview IPC reader. The attached
 * tracker snapshot identifies the selected source/frame; bbox and ID pixels
 * themselves are rendered by NVIDIA OSD on this same buffer. */
static void
write_fullscreen_header (MosaicPreview *preview, guint64 timestamp_ns,
    guint64 pts, guint32 source_frame_num, guint32 object_count,
    guint32 fps_milli, const gchar *snapshot, guint32 snapshot_size)
{
  guint8 *header = preview->fullscreen_mapping;
  memset (header, 0, FULLSCREEN_HEADER_SIZE);
  memcpy (header, "V11UI01\0", 8);
  put_le32 (header + 8, 5);
  put_le64 (header + 12, preview->fullscreen_sequence);
  put_le64 (header + 20, timestamp_ns);
  put_le32 (header + 28, FULLSCREEN_WIDTH);
  put_le32 (header + 32, FULLSCREEN_HEIGHT);
  put_le32 (header + 36, FULLSCREEN_STRIDE);
  put_le32 (header + 40, FULLSCREEN_PAYLOAD);
  put_le32 (header + 44, object_count);
  put_le32 (header + 48, fps_milli);
  put_le64 (header + 68, pts);
  put_le32 (header + 84, source_frame_num);
  put_le64 (header + 92, 0); /* no decoder timing is claimed by this branch */
  put_le64 (header + 100, 0);
  put_le64 (header + 108, 0);
  put_le32 (header + 116, source_frame_num);
  put_le32 (header + 120, snapshot_size);
  if (snapshot_size && snapshot && snapshot_size <= FULLSCREEN_HEADER_SIZE - 128)
    memcpy (header + 128, snapshot, snapshot_size);
}

static gboolean
open_fullscreen_slot (MosaicPreview *preview)
{
  struct stat lock_info, slot_info;
  preview->fullscreen_lock_fd = open (FULLSCREEN_LOCK_PATH,
      O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (preview->fullscreen_lock_fd < 0 ||
      fstat (preview->fullscreen_lock_fd, &lock_info) != 0 ||
      !S_ISREG (lock_info.st_mode) || lock_info.st_uid != getuid () ||
      (lock_info.st_mode & 077) != 0 ||
      flock (preview->fullscreen_lock_fd, LOCK_EX | LOCK_NB) != 0)
    return FALSE;
  preview->owns_fullscreen_lock = TRUE;

  preview->fullscreen_fd = open (FULLSCREEN_PATH,
      O_RDWR | O_CLOEXEC | O_NOFOLLOW);
  if (preview->fullscreen_fd < 0 ||
      fstat (preview->fullscreen_fd, &slot_info) != 0 ||
      !S_ISREG (slot_info.st_mode) || slot_info.st_uid != getuid () ||
      (slot_info.st_mode & 077) != 0 ||
      slot_info.st_size != (off_t) FULLSCREEN_FILE_SIZE)
    return FALSE;
  preview->fullscreen_mapping_size = FULLSCREEN_FILE_SIZE;
  preview->fullscreen_mapping = mmap (NULL, preview->fullscreen_mapping_size,
      PROT_READ | PROT_WRITE, MAP_SHARED, preview->fullscreen_fd, 0);
  if (preview->fullscreen_mapping == MAP_FAILED) {
    preview->fullscreen_mapping = NULL;
    return FALSE;
  }
  preview->fullscreen_sequence = 0;
  write_fullscreen_header (preview, 0, 0, 0, 0, 0, NULL, 0);
  return TRUE;
}

static gboolean
publish_fullscreen_frame (MosaicPreview *preview, const guint8 *plane,
    gint source_stride, gsize plane_offset, gsize buffer_size,
    const FullscreenFrameIdentity *identity)
{
  gsize row_bytes = FULLSCREEN_STRIDE;
  gsize stride, last_row, end_offset;
  gsize snapshot_len;
  gint64 now_us;
  guint64 now_ns;
  gchar *snapshot = NULL;

  if (!plane || !identity || !identity->source_objects_json ||
      identity->source_frame_sequence == 0 || source_stride <= 0 ||
      (gsize) source_stride < row_bytes ||
      FULLSCREEN_HEIGHT - 1 > (G_MAXSIZE - row_bytes) / (gsize) source_stride)
    return FALSE;
  stride = (gsize) source_stride;
  last_row = ((gsize) FULLSCREEN_HEIGHT - 1) * stride;
  if (last_row > G_MAXSIZE - row_bytes)
    return FALSE;
  end_offset = plane_offset + last_row + row_bytes;
  if (end_offset < plane_offset || end_offset > buffer_size)
    return FALSE;

  if (identity->source_id >= MOSAIC_LABEL_COUNT ||
      identity->source_id != (guint) g_atomic_int_get (
          &preview->fullscreen_selected_source) ||
      identity->selection_generation != (guint) g_atomic_int_get (
          &preview->fullscreen_selection_generation))
    return FALSE;
  snapshot = make_fullscreen_snapshot (identity);
  if (!snapshot)
    return FALSE;
  snapshot_len = strlen (snapshot);
  if (snapshot_len == 0 || snapshot_len > G_MAXUINT32 ||
      snapshot_len > FULLSCREEN_HEADER_SIZE - 128) {
    g_free (snapshot);
    return FALSE;
  }

  if (flock (preview->fullscreen_fd, LOCK_EX | LOCK_NB) != 0) {
    g_free (snapshot);
    return FALSE;
  }
  guint8 *destination = preview->fullscreen_mapping + FULLSCREEN_HEADER_SIZE;
  for (guint row = 0; row < FULLSCREEN_HEIGHT; row++)
    memcpy (destination + (gsize) row * row_bytes,
        plane + (gsize) row * stride, row_bytes);
  if (identity->source_id != (guint) g_atomic_int_get (
          &preview->fullscreen_selected_source) ||
      identity->selection_generation != (guint) g_atomic_int_get (
          &preview->fullscreen_selection_generation)) {
    __sync_synchronize ();
    (void) flock (preview->fullscreen_fd, LOCK_UN);
    g_free (snapshot);
    return FALSE; /* selection changed while pixels were being copied */
  }
  __sync_synchronize ();
  now_ns = monotonic_ns ();
  now_us = g_get_monotonic_time ();
  if (preview->fullscreen_last_publish_us > 0 && now_us > preview->fullscreen_last_publish_us) {
    gdouble instant = 1000000.0 / (now_us - preview->fullscreen_last_publish_us);
    preview->fullscreen_fps_ema = preview->fullscreen_fps_ema <= 0.0 ? instant :
        preview->fullscreen_fps_ema * 0.85 + instant * 0.15;
  }
  preview->fullscreen_last_publish_us = now_us;
  preview->fullscreen_sequence++;
  preview->fullscreen_publications++;
  write_fullscreen_header (preview, now_ns, identity->source_pts,
      identity->source_frame_num, identity->object_count,
      (guint32) MAX (0.0, MIN (4294967.0, preview->fullscreen_fps_ema * 1000.0)),
      snapshot, (guint32) snapshot_len);
  __sync_synchronize ();
  (void) flock (preview->fullscreen_fd, LOCK_UN);
  g_free (snapshot);
  return TRUE;
}

static void
fullscreen_identity_free (FullscreenFrameIdentity *identity)
{
  if (!identity)
    return;
  g_free (identity->source_objects_json);
  g_free (identity->fullscreen_objects_json);
  g_free (identity);
}

static void
clear_fullscreen_identities_locked (MosaicPreview *preview)
{
  FullscreenFrameIdentity *identity;
  while ((identity = g_queue_pop_head (&preview->fullscreen_identities)) != NULL)
    fullscreen_identity_free (identity);
}

static void
fullscreen_batch_copy_free (FullscreenBatchCopy *entry)
{
  if (!entry)
    return;
  if (entry->copy)
    gst_buffer_unref (entry->copy);
  g_free (entry);
}

static void
clear_fullscreen_batch_copies_locked (MosaicPreview *preview)
{
  FullscreenBatchCopy *entry;
  while ((entry = g_queue_pop_head (&preview->fullscreen_batch_copies)) != NULL)
    fullscreen_batch_copy_free (entry);
}

static gboolean
same_bbox_coords (const NvBbox_Coords *left, const NvBbox_Coords *right)
{
  return left->left == right->left && left->top == right->top &&
      left->width == right->width && left->height == right->height;
}

/* DeepStream 9.1's object-meta copy callback can emit the copied object list
 * in a different order. Match a copied object by its complete non-ID content;
 * if that signature is not unique, fail closed instead of pairing by index. */
static gboolean
fullscreen_object_copy_signature_matches (const NvDsObjectMeta *original,
    const NvDsObjectMeta *copied)
{
  return original && copied &&
      original->class_id == copied->class_id &&
      original->unique_component_id == copied->unique_component_id &&
      original->confidence == copied->confidence &&
      original->tracker_confidence == copied->tracker_confidence &&
      original->rect_params.left == copied->rect_params.left &&
      original->rect_params.top == copied->rect_params.top &&
      original->rect_params.width == copied->rect_params.width &&
      original->rect_params.height == copied->rect_params.height &&
      same_bbox_coords (&original->detector_bbox_info.org_bbox_coords,
          &copied->detector_bbox_info.org_bbox_coords) &&
      same_bbox_coords (&original->tracker_bbox_info.org_bbox_coords,
          &copied->tracker_bbox_info.org_bbox_coords) &&
      g_strcmp0 (original->text_params.display_text,
          copied->text_params.display_text) == 0;
}

static gboolean
fullscreen_object_pointer_in_list (const NvDsMetaList *list,
    const NvDsObjectMeta *object)
{
  for (const NvDsMetaList *node = list; node; node = node->next)
    if (node->data == object)
      return TRUE;
  return FALSE;
}

static void
record_fullscreen_object_mismatch (FullscreenObjectMismatch *mismatch,
    GstBuffer *source, guint selection_generation, const NvDsFrameMeta *frame,
    guint original_position, gboolean copied_position_valid,
    guint copied_position, const NvDsObjectMeta *original,
    const NvDsObjectMeta *copied)
{
  if (!mismatch)
    return;
  mismatch->valid = TRUE;
  mismatch->copied_position_valid = copied_position_valid;
  mismatch->selection_generation = selection_generation;
  mismatch->source_buffer_identity = source;
  mismatch->buffer_pts = source ? GST_BUFFER_PTS (source) : GST_CLOCK_TIME_NONE;
  mismatch->source_id = frame ? frame->source_id : G_MAXUINT;
  mismatch->frame_num = frame ? frame->frame_num : -1;
  mismatch->batch_id = frame ? frame->batch_id : G_MAXUINT;
  mismatch->frame_pts = frame ? frame->buf_pts : GST_CLOCK_TIME_NONE;
  mismatch->original_position = original_position;
  mismatch->copied_position = copied_position;
  mismatch->original_object_id = original ? (guint64) original->object_id : 0;
  mismatch->copied_object_id = copied ? (guint64) copied->object_id : 0;
}

static void set_fullscreen_copy_failure (const gchar **failure_code,
    FullscreenCopyFailure failure);

static gboolean
compare_fullscreen_frame_objects (GstBuffer *source,
    guint selection_generation, const NvDsFrameMeta *original_frame,
    const NvDsFrameMeta *copied_frame, const gchar **failure_code,
    FullscreenObjectMismatch *object_mismatch)
{
  const guint object_count = original_frame->num_obj_meta;
  gboolean *copied_positions_used = object_count ?
      g_try_new0 (gboolean, object_count) : NULL;
  if (object_count && !copied_positions_used) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_OBJECT_MATCH_ALLOCATION_FAILED);
    return FALSE;
  }

  guint original_position = 0;
  for (const NvDsMetaList *old_node = original_frame->obj_meta_list;
       old_node; old_node = old_node->next, original_position++) {
    const NvDsObjectMeta *original = old_node->data;
    if (!original) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_META_MISSING);
      g_free (copied_positions_used);
      return FALSE;
    }

    guint matches = 0;
    guint copied_position = 0;
    const NvDsObjectMeta *copied = NULL;
    guint candidate_position = 0;
    for (const NvDsMetaList *new_node = copied_frame->obj_meta_list;
         new_node; new_node = new_node->next, candidate_position++) {
      const NvDsObjectMeta *candidate = new_node->data;
      if (!candidate) {
        set_fullscreen_copy_failure (failure_code,
            FULLSCREEN_COPY_OBJECT_META_MISSING);
        g_free (copied_positions_used);
        return FALSE;
      }
      if (fullscreen_object_pointer_in_list (original_frame->obj_meta_list,
              candidate)) {
        set_fullscreen_copy_failure (failure_code,
            FULLSCREEN_COPY_OBJECT_META_ALIAS);
        g_free (copied_positions_used);
        return FALSE;
      }
      if (!fullscreen_object_copy_signature_matches (original, candidate))
        continue;
      matches++;
      copied = candidate;
      copied_position = candidate_position;
    }

    if (matches == 0) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_CORRESPONDENCE_MISSING);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (matches != 1 || copied_positions_used[copied_position]) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_CORRESPONDENCE_AMBIGUOUS);
      g_free (copied_positions_used);
      return FALSE;
    }
    copied_positions_used[copied_position] = TRUE;

    if (original->object_id != copied->object_id) {
      record_fullscreen_object_mismatch (object_mismatch, source,
          selection_generation, original_frame, original_position, TRUE,
          copied_position, original, copied);
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_ID_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->class_id != copied->class_id) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_CLASS_ID_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->unique_component_id != copied->unique_component_id) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_COMPONENT_ID_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->confidence != copied->confidence) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_DETECTOR_CONFIDENCE_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->tracker_confidence != copied->tracker_confidence) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_TRACKER_CONFIDENCE_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->rect_params.left != copied->rect_params.left ||
        original->rect_params.top != copied->rect_params.top ||
        original->rect_params.width != copied->rect_params.width ||
        original->rect_params.height != copied->rect_params.height) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_RECT_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (!same_bbox_coords (&original->detector_bbox_info.org_bbox_coords,
            &copied->detector_bbox_info.org_bbox_coords)) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_DETECTOR_RECT_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (!same_bbox_coords (&original->tracker_bbox_info.org_bbox_coords,
            &copied->tracker_bbox_info.org_bbox_coords)) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_TRACKER_RECT_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (original->text_params.display_text &&
        original->text_params.display_text == copied->text_params.display_text) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_DISPLAY_TEXT_ALIAS);
      g_free (copied_positions_used);
      return FALSE;
    }
    if (g_strcmp0 (original->text_params.display_text,
            copied->text_params.display_text) != 0) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_DISPLAY_TEXT_CONTENT_CHANGED);
      g_free (copied_positions_used);
      return FALSE;
    }
  }

  g_free (copied_positions_used);
  return TRUE;
}

static const gchar *const fullscreen_copy_failure_names[] = {
  "OK",
  "SOURCE_BUFFER_MISSING",
  "SOURCE_PTS_INVALID",
  "SOURCE_BATCH_META_MISSING",
  "SOURCE_BATCH_META_DUPLICATE",
  "SOURCE_META_COPY_CALLBACK_MISSING",
  "SOURCE_META_RELEASE_CALLBACK_MISSING",
  "SOURCE_META_ACCESSOR_MISMATCH",
  "SOURCE_BATCH_CAPACITY_ZERO",
  "SOURCE_BATCH_EXCEEDS_CAPACITY",
  "SOURCE_MEMORY_MISSING",
  "GST_BUFFER_COPY_FAILED",
  "COPIED_BATCH_META_MISSING",
  "COPIED_BATCH_META_DUPLICATE",
  "COPIED_META_COPY_CALLBACK_CHANGED",
  "COPIED_META_RELEASE_CALLBACK_CHANGED",
  "COPIED_BATCH_META_ALIAS",
  "COPIED_FRAME_COUNT_CHANGED",
  "COPIED_PTS_CHANGED",
  "COPIED_MEMORY_COUNT_CHANGED",
  "VIDEO_MEMORY_NOT_SHARED",
  "FRAME_LIST_LENGTH_CHANGED",
  "SOURCE_FRAME_LIST_COUNT_MISMATCH",
  "COPIED_FRAME_LIST_COUNT_MISMATCH",
  "FRAME_META_MISSING",
  "FRAME_META_ALIAS",
  "FRAME_SOURCE_ID_CHANGED",
  "FRAME_PAD_INDEX_CHANGED",
  "FRAME_BATCH_ID_CHANGED",
  "FRAME_NUMBER_CHANGED",
  "FRAME_PTS_CHANGED",
  "FRAME_NTP_TIMESTAMP_CHANGED",
  "FRAME_GEOMETRY_CHANGED",
  "FRAME_OBJECT_COUNT_CHANGED",
  "SOURCE_OBJECT_LIST_COUNT_MISMATCH",
  "COPIED_OBJECT_LIST_COUNT_MISMATCH",
  "OBJECT_LIST_LENGTH_CHANGED",
  "OBJECT_META_MISSING",
  "OBJECT_META_ALIAS",
  "OBJECT_CORRESPONDENCE_MISSING",
  "OBJECT_CORRESPONDENCE_AMBIGUOUS",
  "OBJECT_MATCH_ALLOCATION_FAILED",
  "OBJECT_ID_CHANGED",
  "OBJECT_CLASS_ID_CHANGED",
  "OBJECT_COMPONENT_ID_CHANGED",
  "DETECTOR_CONFIDENCE_CHANGED",
  "TRACKER_CONFIDENCE_CHANGED",
  "OBJECT_RECT_CHANGED",
  "DETECTOR_RECT_CHANGED",
  "TRACKER_RECT_CHANGED",
  "DISPLAY_TEXT_ALIAS",
  "DISPLAY_TEXT_CONTENT_CHANGED",
  "PARENT_META_ATTACH_FAILED"
};

G_STATIC_ASSERT (G_N_ELEMENTS (fullscreen_copy_failure_names) ==
    FULLSCREEN_COPY_FAILURE_COUNT);

static guint
count_nvds_batch_attachments (GstBuffer *buffer, NvDsMeta **found)
{
  guint count = 0;
  gpointer state = NULL;
  GstMeta *gst_meta;
  if (found)
    *found = NULL;
  while ((gst_meta = gst_buffer_iterate_meta (buffer, &state)) != NULL) {
    if (gst_meta->info->api != NVDS_META_API_TYPE)
      continue;
    NvDsMeta *nvds_meta = (NvDsMeta *) gst_meta;
    if (nvds_meta->meta_type != NVDS_BATCH_GST_META)
      continue;
    count++;
    if (found)
      *found = nvds_meta;
  }
  return count;
}

static guint
count_nvds_meta_list (const NvDsMetaList *list)
{
  guint count = 0;
  for (const NvDsMetaList *node = list; node; node = node->next)
    count++;
  return count;
}

static const gchar *
fullscreen_copy_failure_name (FullscreenCopyFailure failure)
{
  if (failure < FULLSCREEN_COPY_FAILURE_COUNT)
    return fullscreen_copy_failure_names[failure];
  return "UNKNOWN_COPY_FAILURE";
}

static void
set_fullscreen_copy_failure (const gchar **failure_code,
    FullscreenCopyFailure failure)
{
  if (failure_code)
    *failure_code = fullscreen_copy_failure_name (failure);
}

static gchar *
fullscreen_geometry_mismatch_field_names (guint mismatch_fields)
{
  GString *fields = g_string_new (NULL);
  if (mismatch_fields & FULLSCREEN_GEOMETRY_SOURCE_WIDTH_CHANGED)
    g_string_append (fields, "source_frame_width");
  if (mismatch_fields & FULLSCREEN_GEOMETRY_SOURCE_HEIGHT_CHANGED) {
    if (fields->len)
      g_string_append_c (fields, ',');
    g_string_append (fields, "source_frame_height");
  }
  if (mismatch_fields & FULLSCREEN_GEOMETRY_PIPELINE_WIDTH_CHANGED) {
    if (fields->len)
      g_string_append_c (fields, ',');
    g_string_append (fields, "pipeline_width");
  }
  if (mismatch_fields & FULLSCREEN_GEOMETRY_PIPELINE_HEIGHT_CHANGED) {
    if (fields->len)
      g_string_append_c (fields, ',');
    g_string_append (fields, "pipeline_height");
  }
  return g_string_free (fields, FALSE);
}

static void
restore_copied_pipeline_geometry (const NvDsFrameMeta *original,
    NvDsFrameMeta *copied)
{
  /* DeepStream 9.1's batch-meta copy callback omits these streammux fields.
   * Restore only zero-valued omissions; leave unexpected nonzero values for
   * the strict comparison below to reject. */
  if (copied->pipeline_width == 0 && original->pipeline_width != 0)
    copied->pipeline_width = original->pipeline_width;
  if (copied->pipeline_height == 0 && original->pipeline_height != 0)
    copied->pipeline_height = original->pipeline_height;
}

static guint
fullscreen_geometry_mismatch_fields (const NvDsFrameMeta *original,
    const NvDsFrameMeta *copied)
{
  guint mismatch_fields = 0;
  if (original->source_frame_width != copied->source_frame_width)
    mismatch_fields |= FULLSCREEN_GEOMETRY_SOURCE_WIDTH_CHANGED;
  if (original->source_frame_height != copied->source_frame_height)
    mismatch_fields |= FULLSCREEN_GEOMETRY_SOURCE_HEIGHT_CHANGED;
  if (original->pipeline_width != copied->pipeline_width)
    mismatch_fields |= FULLSCREEN_GEOMETRY_PIPELINE_WIDTH_CHANGED;
  if (original->pipeline_height != copied->pipeline_height)
    mismatch_fields |= FULLSCREEN_GEOMETRY_PIPELINE_HEIGHT_CHANGED;
  return mismatch_fields;
}

static gboolean
first_fullscreen_copy_failure_for_selection (MosaicPreview *preview,
    const gchar *failure_code)
{
  for (guint i = 1; i < FULLSCREEN_COPY_FAILURE_COUNT; i++) {
    if (g_strcmp0 (failure_code, fullscreen_copy_failure_names[i]) == 0)
      return g_atomic_int_compare_and_exchange (
          &preview->fullscreen_copy_failure_logged[i], 0, 1);
  }
  return FALSE;
}

/* The two tilers must not mutate the same NvDsObjectMeta. gst_buffer_copy()
 * invokes NVIDIA's metadata-copy callbacks; reject a shallow/incomplete copy.
 * The video GstMemory must remain shared, and a parent reference prevents the
 * upstream pool from recycling it while the fullscreen branch is using it. */
static GstBuffer *
clone_fullscreen_batch_buffer (GstBuffer *source, guint selection_generation,
    const gchar **failure_code, FullscreenGeometryMismatch *geometry_mismatch,
    FullscreenObjectMismatch *object_mismatch)
{
  if (geometry_mismatch)
    memset (geometry_mismatch, 0, sizeof (*geometry_mismatch));
  if (object_mismatch)
    memset (object_mismatch, 0, sizeof (*object_mismatch));
  if (failure_code)
    *failure_code = "UNKNOWN_COPY_FAILURE";
  if (!source) {
    set_fullscreen_copy_failure (failure_code, FULLSCREEN_COPY_SOURCE_BUFFER_MISSING);
    return NULL;
  }
  if (!GST_BUFFER_PTS_IS_VALID (source)) {
    set_fullscreen_copy_failure (failure_code, FULLSCREEN_COPY_SOURCE_PTS_INVALID);
    return NULL;
  }

  NvDsMeta *original_nvds_meta = NULL;
  guint original_meta_count = count_nvds_batch_attachments (source,
      &original_nvds_meta);
  if (original_meta_count == 0 || !original_nvds_meta) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_BATCH_META_MISSING);
    return NULL;
  }
  if (original_meta_count != 1) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_BATCH_META_DUPLICATE);
    return NULL;
  }
  if (!original_nvds_meta->copyfunc) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_META_CALLBACK_MISSING);
    return NULL;
  }
  if (!original_nvds_meta->freefunc) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_META_RELEASE_MISSING);
    return NULL;
  }
  NvDsBatchMeta *original = (NvDsBatchMeta *) original_nvds_meta->meta_data;
  if (!original || gst_buffer_get_nvds_batch_meta (source) != original) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_META_ACCESSOR_MISMATCH);
    return NULL;
  }
  if (!GST_BUFFER_PTS_IS_VALID (source)) {
    set_fullscreen_copy_failure (failure_code, FULLSCREEN_COPY_SOURCE_PTS_INVALID);
    return NULL;
  }
  if (original->max_frames_in_batch == 0) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_BATCH_CAPACITY_ZERO);
    return NULL;
  }
  if (original->num_frames_in_batch > original->max_frames_in_batch) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_BATCH_EXCEEDS_CAPACITY);
    return NULL;
  }
  if (count_nvds_meta_list (original->frame_meta_list) !=
      original->num_frames_in_batch) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_FRAME_LIST_COUNT_MISMATCH);
    return NULL;
  }
  if (gst_buffer_n_memory (source) == 0) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_SOURCE_MEMORY_MISSING);
    return NULL;
  }

  GstBuffer *copy = gst_buffer_copy (source);
  if (!copy) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_GST_BUFFER_COPY_FAILED);
    return NULL;
  }

  NvDsMeta *copied_nvds_meta = NULL;
  guint copied_meta_count = count_nvds_batch_attachments (copy,
      &copied_nvds_meta);
  if (copied_meta_count == 0 || !copied_nvds_meta) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_BATCH_META_MISSING);
    goto fail;
  }
  if (copied_meta_count != 1) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_BATCH_META_DUPLICATE);
    goto fail;
  }
  if (!copied_nvds_meta->copyfunc ||
      copied_nvds_meta->copyfunc != original_nvds_meta->copyfunc) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_META_CALLBACK_CHANGED);
    goto fail;
  }
  if (!copied_nvds_meta->freefunc ||
      copied_nvds_meta->freefunc != original_nvds_meta->freefunc) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_META_RELEASE_CHANGED);
    goto fail;
  }
  NvDsBatchMeta *copied = (NvDsBatchMeta *) copied_nvds_meta->meta_data;
  if (!copied || copied == original ||
      gst_buffer_get_nvds_batch_meta (copy) != copied) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_BATCH_META_ALIAS);
    goto fail;
  }
  if (copied->num_frames_in_batch != original->num_frames_in_batch) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_FRAME_COUNT_CHANGED);
    goto fail;
  }
  if (count_nvds_meta_list (copied->frame_meta_list) !=
      copied->num_frames_in_batch) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_FRAME_LIST_COUNT_MISMATCH);
    goto fail;
  }
  if (GST_BUFFER_PTS (copy) != GST_BUFFER_PTS (source)) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_PTS_CHANGED);
    goto fail;
  }
  if (gst_buffer_n_memory (copy) != gst_buffer_n_memory (source)) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_COPIED_MEMORY_COUNT_CHANGED);
    goto fail;
  }
  for (guint i = 0; i < gst_buffer_n_memory (source); i++)
    if (gst_buffer_peek_memory (copy, i) != gst_buffer_peek_memory (source, i)) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_VIDEO_MEMORY_NOT_SHARED);
      goto fail;
    }

  NvDsMetaList *old_frame = original->frame_meta_list;
  NvDsMetaList *new_frame = copied->frame_meta_list;
  for (; old_frame && new_frame; old_frame = old_frame->next, new_frame = new_frame->next) {
    NvDsFrameMeta *old = old_frame->data;
    NvDsFrameMeta *new = new_frame->data;
    if (!old || !new) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_META_MISSING);
      goto fail;
    }
    if (old == new) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_META_ALIAS);
      goto fail;
    }
    if (old->source_id != new->source_id) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_SOURCE_ID_CHANGED);
      goto fail;
    }
    if (old->pad_index != new->pad_index) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_PAD_INDEX_CHANGED);
      goto fail;
    }
    if (old->batch_id != new->batch_id) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_BATCH_ID_CHANGED);
      goto fail;
    }
    if (old->frame_num != new->frame_num) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_NUMBER_CHANGED);
      goto fail;
    }
    if (old->buf_pts != new->buf_pts) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_PTS_CHANGED);
      goto fail;
    }
    if (old->ntp_timestamp != new->ntp_timestamp) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_NTP_TIMESTAMP_CHANGED);
      goto fail;
    }
    /* Preserve omitted streammux values from this exact source frame, then
     * retain strict equality checks for all four geometry fields. */
    restore_copied_pipeline_geometry (old, new);
    guint geometry_mismatch_fields = fullscreen_geometry_mismatch_fields (old,
        new);
    if (geometry_mismatch_fields) {
      if (geometry_mismatch) {
        geometry_mismatch->valid = TRUE;
        geometry_mismatch->mismatch_fields = geometry_mismatch_fields;
        geometry_mismatch->selection_generation = selection_generation;
        geometry_mismatch->source_buffer_identity = source;
        geometry_mismatch->buffer_pts = GST_BUFFER_PTS (source);
        geometry_mismatch->frame_source_id = old->source_id;
        geometry_mismatch->frame_num = old->frame_num;
        geometry_mismatch->batch_id = old->batch_id;
        geometry_mismatch->frame_pts = old->buf_pts;
        geometry_mismatch->original_source_frame_width = old->source_frame_width;
        geometry_mismatch->original_source_frame_height = old->source_frame_height;
        geometry_mismatch->original_pipeline_width = old->pipeline_width;
        geometry_mismatch->original_pipeline_height = old->pipeline_height;
        geometry_mismatch->copied_source_frame_width = new->source_frame_width;
        geometry_mismatch->copied_source_frame_height = new->source_frame_height;
        geometry_mismatch->copied_pipeline_width = new->pipeline_width;
        geometry_mismatch->copied_pipeline_height = new->pipeline_height;
      }
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_GEOMETRY_CHANGED);
      goto fail;
    }
    if (old->num_obj_meta != new->num_obj_meta) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_FRAME_OBJECT_COUNT_CHANGED);
      goto fail;
    }
    if (count_nvds_meta_list (old->obj_meta_list) != old->num_obj_meta) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_SOURCE_OBJECT_LIST_COUNT_MISMATCH);
      goto fail;
    }
    if (count_nvds_meta_list (new->obj_meta_list) != new->num_obj_meta) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_COPIED_OBJECT_LIST_COUNT_MISMATCH);
      goto fail;
    }
    if (!compare_fullscreen_frame_objects (source, selection_generation, old,
            new, failure_code, object_mismatch))
      goto fail;
    if (count_nvds_meta_list (old->obj_meta_list) !=
        count_nvds_meta_list (new->obj_meta_list)) {
      set_fullscreen_copy_failure (failure_code,
          FULLSCREEN_COPY_OBJECT_LIST_LENGTH_CHANGED);
      goto fail;
    }
  }
  if (old_frame || new_frame) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_FRAME_LIST_LENGTH_CHANGED);
    goto fail;
  }
  if (!gst_buffer_add_parent_buffer_meta (copy, source)) {
    set_fullscreen_copy_failure (failure_code,
        FULLSCREEN_COPY_PARENT_META_ATTACH_FAILED);
    goto fail;
  }
  if (failure_code)
    *failure_code = fullscreen_copy_failure_name (FULLSCREEN_COPY_OK);
  return copy;
fail:
  gst_buffer_unref (copy);
  return NULL;
}

static gboolean
fullscreen_target_is_current (MosaicPreview *preview, guint source_id, guint generation)
{
  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint target = pending >= 0 ? pending : selected;
  return target >= 0 && (guint) target == source_id &&
      generation == (guint) g_atomic_int_get (&preview->fullscreen_selection_generation);
}

/* Snapshot metadata before the analytics tee dispatches the batch to either
 * tiler. Only a selected fullscreen request incurs this metadata copy. */
static GstPadProbeReturn
fullscreen_analytics_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint target = pending >= 0 ? pending : selected;
  guint generation = (guint) g_atomic_int_get (&preview->fullscreen_selection_generation);
  GstBuffer *source = GST_PAD_PROBE_INFO_BUFFER (info);
  /* The analytics tee is after NvDCF and before both tilers. Capture only
   * source frames actually present in this batch; cached tiles are excluded. */
  if (source)
    remember_mosaic_frame_identity (preview, source,
        gst_buffer_get_nvds_batch_meta (source));
  if (!source || target < 0 || target >= (gint) MOSAIC_LABEL_COUNT || generation == 0)
    return GST_PAD_PROBE_OK;

  const gchar *failure_code = NULL;
  FullscreenGeometryMismatch geometry_mismatch = { 0 };
  FullscreenObjectMismatch object_mismatch = { 0 };
  GstBuffer *copy = clone_fullscreen_batch_buffer (source, generation,
      &failure_code, &geometry_mismatch, &object_mismatch);
  if (!copy) {
    NvDsMeta *source_nvds_meta = NULL;
    guint source_nvds_meta_count = source ?
        count_nvds_batch_attachments (source, &source_nvds_meta) : 0;
    NvDsBatchMeta *source_batch = source ?
        gst_buffer_get_nvds_batch_meta (source) : NULL;
    if (first_fullscreen_copy_failure_for_selection (preview, failure_code)) {
      if (object_mismatch.valid) {
        g_printerr ("FULLSCREEN_STAGE_FAIL stage=metadata_isolation "
            "code=%s selected_source_id=%d frame_source_id=%u frame_num=%d "
            "batch_id=%u buffer_identity=%p buffer_pts=%" G_GUINT64_FORMAT " "
            "frame_pts=%" G_GUINT64_FORMAT " selection_generation=%u "
            "original_object_pos=%u copied_object_pos_valid=%s "
            "copied_object_pos=%u "
            "original_object_id=%" G_GUINT64_FORMAT " "
            "copied_object_id=%" G_GUINT64_FORMAT "\n", failure_code,
            target, object_mismatch.source_id, object_mismatch.frame_num,
            object_mismatch.batch_id, object_mismatch.source_buffer_identity,
            object_mismatch.buffer_pts, object_mismatch.frame_pts,
            object_mismatch.selection_generation,
            object_mismatch.original_position,
            object_mismatch.copied_position_valid ? "true" : "false",
            object_mismatch.copied_position,
            object_mismatch.original_object_id,
            object_mismatch.copied_object_id);
      } else if (geometry_mismatch.valid) {
        gchar *failed_fields = fullscreen_geometry_mismatch_field_names (
            geometry_mismatch.mismatch_fields);
        g_printerr ("FULLSCREEN_STAGE_FAIL stage=metadata_isolation "
            "code=%s source_id=%d frame_source_id=%u frame_num=%d batch_id=%u "
            "buffer_identity=%p buffer_pts=%" G_GUINT64_FORMAT " "
            "frame_pts=%" G_GUINT64_FORMAT " selection_generation=%u "
            "failed_fields=%s original={source_frame_width:%u,"
            "source_frame_height:%u,pipeline_width:%u,pipeline_height:%u} "
            "copied={source_frame_width:%u,source_frame_height:%u,"
            "pipeline_width:%u,pipeline_height:%u} pts_valid=%s "
            "batch_frames=%u max_frames=%u nvds_batch_meta_count=%u "
            "memory_count=%u\n", failure_code, target,
            geometry_mismatch.frame_source_id, geometry_mismatch.frame_num,
            geometry_mismatch.batch_id, geometry_mismatch.source_buffer_identity,
            geometry_mismatch.buffer_pts, geometry_mismatch.frame_pts,
            geometry_mismatch.selection_generation, failed_fields,
            geometry_mismatch.original_source_frame_width,
            geometry_mismatch.original_source_frame_height,
            geometry_mismatch.original_pipeline_width,
            geometry_mismatch.original_pipeline_height,
            geometry_mismatch.copied_source_frame_width,
            geometry_mismatch.copied_source_frame_height,
            geometry_mismatch.copied_pipeline_width,
            geometry_mismatch.copied_pipeline_height,
            source && GST_BUFFER_PTS_IS_VALID (source) ? "true" : "false",
            source_batch ? source_batch->num_frames_in_batch : 0,
            source_batch ? source_batch->max_frames_in_batch : 0,
            source_nvds_meta_count, source ? gst_buffer_n_memory (source) : 0);
        g_free (failed_fields);
      } else {
        g_printerr ("FULLSCREEN_STAGE_FAIL stage=metadata_isolation "
            "code=%s source_id=%d generation=%u pts_valid=%s "
            "batch_frames=%u max_frames=%u nvds_batch_meta_count=%u "
            "memory_count=%u\n", failure_code, target, generation,
            source && GST_BUFFER_PTS_IS_VALID (source) ? "true" : "false",
            source_batch ? source_batch->num_frames_in_batch : 0,
            source_batch ? source_batch->max_frames_in_batch : 0,
            source_nvds_meta_count, source ? gst_buffer_n_memory (source) : 0);
      }
    }
    return GST_PAD_PROBE_OK; /* The normal grid keeps the original buffer. */
  }
  FullscreenBatchCopy *entry = g_new0 (FullscreenBatchCopy, 1);
  entry->original = source;
  entry->copy = copy;
  entry->source_id = (guint) target;
  entry->selection_generation = generation;
  g_mutex_lock (&preview->mutex);
  if (!fullscreen_target_is_current (preview, entry->source_id, generation)) {
    gint current_generation = g_atomic_int_get (
        &preview->fullscreen_selection_generation);
    g_mutex_unlock (&preview->mutex);
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_copy_selection_race_logged, 0, 1))
      g_printerr ("FULLSCREEN_COPY_SKIP code=SELECTION_CHANGED_DURING_COPY "
          "source_id=%u copy_generation=%u current_generation=%d\n",
          entry->source_id, generation, current_generation);
    fullscreen_batch_copy_free (entry);
    return GST_PAD_PROBE_OK;
  }
  while (g_queue_get_length (&preview->fullscreen_batch_copies) >=
      FULLSCREEN_BATCH_COPY_QUEUE_CAPACITY) {
    preview->fullscreen_batch_copy_queue_drops++;
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_copy_queue_drop_logged, 0, 1))
      g_printerr ("FULLSCREEN_COPY_QUEUE_DROP source_id=%u generation=%u "
          "capacity=%u\n", entry->source_id, generation,
          FULLSCREEN_BATCH_COPY_QUEUE_CAPACITY);
    fullscreen_batch_copy_free (g_queue_pop_head (&preview->fullscreen_batch_copies));
  }
  g_queue_push_tail (&preview->fullscreen_batch_copies, entry);
  g_mutex_unlock (&preview->mutex);
  return GST_PAD_PROBE_OK;
}

/* The parent pointer is an exact GstBuffer identity, not a timestamp guess. */
static GstBuffer *
take_fullscreen_batch_copy (MosaicPreview *preview, GstBuffer *original,
    guint source_id, guint generation, const gchar **miss_code)
{
  if (miss_code)
    *miss_code = "COPY_QUEUE_EMPTY";
  gint current_generation = g_atomic_int_get (
      &preview->fullscreen_selection_generation);
  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint current_source = pending >= 0 ? pending : selected;
  if (current_generation < 0 || generation != (guint) current_generation) {
    if (miss_code)
      *miss_code = "SELECTION_GENERATION_MISMATCH";
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_batch_copy_generation_misses++;
    g_mutex_unlock (&preview->mutex);
    return NULL;
  }
  if (current_source < 0 || source_id != (guint) current_source) {
    if (miss_code)
      *miss_code = "SELECTED_SOURCE_MISMATCH";
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_batch_copy_source_misses++;
    g_mutex_unlock (&preview->mutex);
    return NULL;
  }
  FullscreenBatchCopy *match = NULL;
  GstBuffer *copy = NULL;
  g_mutex_lock (&preview->mutex);
  for (GList *node = preview->fullscreen_batch_copies.head; node; node = node->next) {
    FullscreenBatchCopy *entry = node->data;
    if (entry->original == original && entry->source_id == source_id &&
        entry->selection_generation != generation) {
      preview->fullscreen_batch_copy_generation_misses++;
      if (miss_code)
        *miss_code = "SELECTION_GENERATION_MISMATCH";
    } else if (entry->original == original &&
        entry->selection_generation == generation && entry->source_id != source_id) {
      preview->fullscreen_batch_copy_source_misses++;
      if (miss_code)
        *miss_code = "SELECTED_SOURCE_MISMATCH";
    } else if (entry->source_id == source_id &&
        entry->selection_generation == generation && entry->original != original) {
      preview->fullscreen_batch_copy_identity_misses++;
      if (miss_code && g_strcmp0 (*miss_code, "COPY_QUEUE_EMPTY") == 0)
        *miss_code = "EXACT_GST_BUFFER_IDENTITY_MISMATCH";
    }
    if (entry->original == original && entry->source_id == source_id &&
        entry->selection_generation == generation) {
      match = entry;
      break;
    }
  }
  if (match) {
    while (preview->fullscreen_batch_copies.head &&
        preview->fullscreen_batch_copies.head->data != match)
      fullscreen_batch_copy_free (g_queue_pop_head (&preview->fullscreen_batch_copies));
    (void) g_queue_pop_head (&preview->fullscreen_batch_copies);
    copy = match->copy;
    match->copy = NULL;
    fullscreen_batch_copy_free (match);
  }
  g_mutex_unlock (&preview->mutex);
  if (copy && !fullscreen_target_is_current (preview, source_id, generation)) {
    current_generation = g_atomic_int_get (
        &preview->fullscreen_selection_generation);
    if (miss_code)
      *miss_code = current_generation != (gint) generation ?
          "SELECTION_GENERATION_MISMATCH" : "SELECTED_SOURCE_MISMATCH";
    g_mutex_lock (&preview->mutex);
    if (current_generation != (gint) generation)
      preview->fullscreen_batch_copy_generation_misses++;
    else
      preview->fullscreen_batch_copy_source_misses++;
    g_mutex_unlock (&preview->mutex);
    gst_buffer_unref (copy);
    copy = NULL;
  }
  return copy;
}

static void
append_json_string (GString *json, const gchar *value, gsize max_bytes)
{
  if (!value) {
    g_string_append (json, "null");
    return;
  }
  g_string_append_c (json, '"');
  for (gsize i = 0; value[i] && i < max_bytes; i++) {
    const guchar ch = (guchar) value[i];
    switch (ch) {
      case '"': g_string_append (json, "\\\""); break;
      case '\\': g_string_append (json, "\\\\"); break;
      case '\b': g_string_append (json, "\\b"); break;
      case '\f': g_string_append (json, "\\f"); break;
      case '\n': g_string_append (json, "\\n"); break;
      case '\r': g_string_append (json, "\\r"); break;
      case '\t': g_string_append (json, "\\t"); break;
      default:
        if (ch < 0x20)
          g_string_append_printf (json, "\\u%04x", ch);
        else
          g_string_append_c (json, (gchar) ch);
        break;
    }
  }
  g_string_append_c (json, '"');
}

static void
append_json_double (GString *json, gdouble value)
{
  if (!isfinite (value)) {
    g_string_append (json, "null");
    return;
  }
  gchar number[G_ASCII_DTOSTR_BUF_SIZE];
  g_ascii_formatd (number, sizeof number, "%.6f", value);
  g_string_append (json, number);
}

static void
append_bbox_info_json (GString *json, const NvBbox_Coords *bbox)
{
  g_string_append (json, "{\"left\":");
  append_json_double (json, bbox->left);
  g_string_append (json, ",\"top\":");
  append_json_double (json, bbox->top);
  g_string_append (json, ",\"width\":");
  append_json_double (json, bbox->width);
  g_string_append (json, ",\"height\":");
  append_json_double (json, bbox->height);
  g_string_append_c (json, '}');
}

static void
append_rect_params_json (GString *json, const NvOSD_RectParams *rect)
{
  g_string_append (json, "{\"left\":");
  append_json_double (json, rect->left);
  g_string_append (json, ",\"top\":");
  append_json_double (json, rect->top);
  g_string_append (json, ",\"width\":");
  append_json_double (json, rect->width);
  g_string_append (json, ",\"height\":");
  append_json_double (json, rect->height);
  g_string_append_c (json, '}');
}

static gchar *
serialize_fullscreen_objects (const NvDsFrameMeta *frame_meta,
    guint source_id, guint32 source_frame_num, guint64 source_pts,
    guint bbox_width, guint bbox_height, const gchar *metadata_stage,
    guint *serialized_count)
{
  GString *json = g_string_new ("[");
  guint count = 0;
  if (frame_meta) {
    for (NvDsMetaList *node = frame_meta->obj_meta_list; node; node = node->next) {
      const NvDsObjectMeta *object = (const NvDsObjectMeta *) node->data;
      if (!object)
        continue;
      if (count >= FULLSCREEN_METADATA_OBJECT_LIMIT)
        continue;
      if (count)
        g_string_append_c (json, ',');

      const NvOSD_RectParams *rect = &object->rect_params;
      const gboolean tracked = object->object_id != UNTRACKED_OBJECT_ID;
      g_string_append_printf (json,
          "{\"camera_id\":\"CAM-%02u\",\"source_id\":%u,"
          "\"frame_num\":%u,\"source_frame_num\":%u,"
          "\"pts_ns\":%" G_GUINT64_FORMAT ",\"class_id\":%d,"
          "\"object_id\":%" G_GUINT64_FORMAT ",\"native_track_id\":",
          source_id + 1, source_id, source_frame_num, source_frame_num,
          source_pts, object->class_id, (guint64) object->object_id);
      if (tracked)
        g_string_append_printf (json, "%" G_GUINT64_FORMAT,
            (guint64) object->object_id);
      else
        g_string_append (json, "null");
      g_string_append_printf (json,
          ",\"tracked\":%s,\"unique_component_id\":%d,"
          "\"obj_label\":", tracked ? "true" : "false",
          object->unique_component_id);
      append_json_string (json, object->obj_label, sizeof object->obj_label);
      g_string_append (json, ",\"osd_text\":{\"display_text\":");
      append_json_string (json, object->text_params.display_text, 127u);
      g_string_append (json, ",\"x_offset\":");
      append_json_double (json, object->text_params.x_offset);
      g_string_append (json, ",\"y_offset\":");
      append_json_double (json, object->text_params.y_offset);
      g_string_append_c (json, '}');
      g_string_append (json, ",\"detector_confidence\":");
      append_json_double (json, object->confidence);
      g_string_append (json, ",\"tracker_confidence\":");
      append_json_double (json, object->tracker_confidence);
      g_string_append (json, ",\"bbox\":[");
      append_json_double (json, rect->left);
      g_string_append_c (json, ',');
      append_json_double (json, rect->top);
      g_string_append_c (json, ',');
      append_json_double (json, rect->left + rect->width);
      g_string_append_c (json, ',');
      append_json_double (json, rect->top + rect->height);
      g_string_append_printf (json,
          "],\"bbox_source\":\"%s\",\"bbox_coordinate_space\":\"pipeline\","
          "\"bbox_coordinate_width\":%u,\"bbox_coordinate_height\":%u,"
          "\"metadata_stage\":\"%s\",\"detector_bbox_info\":",
          metadata_stage, bbox_width, bbox_height, metadata_stage);
      append_bbox_info_json (json,
          &object->detector_bbox_info.org_bbox_coords);
      g_string_append (json, ",\"tracker_bbox_info\":");
      append_bbox_info_json (json,
          &object->tracker_bbox_info.org_bbox_coords);
      g_string_append (json, ",\"rect_params\":");
      append_rect_params_json (json, rect);
      g_string_append_c (json, '}');
      count++;
    }
  }
  g_string_append_c (json, ']');
  if (serialized_count)
    *serialized_count = count;
  return g_string_free (json, FALSE);
}

static gchar *
make_fullscreen_snapshot (const FullscreenFrameIdentity *identity)
{
  return g_strdup_printf (
      "{\"camera_id\":\"CAM-%02u\",\"source_id\":%u,"
      "\"pts_ns\":%" G_GUINT64_FORMAT ",\"source_pts_valid\":%s,"
      "\"frame_num\":%u,\"source_frame_num\":%u,"
      "\"source_frame_num_valid\":true,\"source_frame_sequence\":%"
      G_GUINT64_FORMAT ",\"selection_generation\":%u,"
      "\"source_identity_stage\":\"post_tracker_pre_tiler\","
      "\"identity_correlation\":\"gst_buffer_pts_exact\","
      "\"source_frame_width\":%u,\"source_frame_height\":%u,"
      "\"pipeline_width\":%u,\"pipeline_height\":%u,"
      "\"bbox_coordinate_space\":\"pipeline\",\"bbox_coordinate_width\":%u,"
      "\"bbox_coordinate_height\":%u,\"source_object_count\":%u,"
      "\"source_objects_truncated\":%s,\"objects\":%s,"
      "\"fullscreen_metadata\":{\"matched\":%s,"
      "\"selection_generation\":%u,\"branch_pts_valid\":%s,"
      "\"branch_pts_ns\":%" G_GUINT64_FORMAT ",\"num_frames_in_batch\":%u,"
      "\"frame_meta_source_id\":%u,"
      "\"frame_meta_frame_num\":%d,\"frame_meta_pts_ns\":%"
      G_GUINT64_FORMAT ",\"frame_meta_pts_valid\":%s,"
      "\"frame_meta_pad_index\":%u},"
      "\"fullscreen_object_count\":%u,\"fullscreen_objects_truncated\":%s,"
      "\"fullscreen_objects\":%s}",
      identity->source_id + 1, identity->source_id, identity->source_pts,
      identity->source_pts_valid ? "true" : "false",
      identity->source_frame_num, identity->source_frame_num,
      identity->source_frame_sequence, identity->selection_generation,
      identity->source_frame_width,
      identity->source_frame_height, identity->pipeline_width,
      identity->pipeline_height, identity->pipeline_width,
      identity->pipeline_height, identity->object_count,
      identity->object_count > FULLSCREEN_METADATA_OBJECT_LIMIT ? "true" : "false",
      identity->source_objects_json ? identity->source_objects_json : "[]",
      identity->fullscreen_metadata_matched ? "true" : "false",
      identity->fullscreen_metadata_generation,
      GST_CLOCK_TIME_IS_VALID (identity->branch_pts) ? "true" : "false",
      identity->branch_pts, identity->fullscreen_num_frames_in_batch,
      identity->fullscreen_meta_source_id, identity->fullscreen_meta_frame_num,
      identity->fullscreen_meta_frame_pts,
      identity->fullscreen_meta_pts_valid ? "true" : "false",
      identity->fullscreen_meta_pad_index,
      identity->fullscreen_metadata_matched ?
          (identity->fullscreen_objects_json ? identity->fullscreen_object_count : 0u) : 0u,
      identity->fullscreen_metadata_matched &&
          identity->fullscreen_frame_object_count > FULLSCREEN_METADATA_OBJECT_LIMIT ? "true" : "false",
      identity->fullscreen_objects_json ? identity->fullscreen_objects_json : "[]");
}

/* The tiler output is a selected/zoomed image with transformed metadata. Its
 * frame number and PTS are not used as source identity. Instead, the selected
 * post-tracker source metadata is copied while the exact input GstBuffer is at
 * the valve sink; the output sample must carry that same valid GstBuffer PTS. */
static gboolean
queue_fullscreen_identity (MosaicPreview *preview, const NvDsFrameMeta *frame_meta,
    GstBuffer *buffer, guint expected_generation)
{
  if (!frame_meta || !buffer || !GST_BUFFER_PTS_IS_VALID (buffer))
    return FALSE;
  guint source_id = frame_meta->source_id;
  if (source_id >= MOSAIC_LABEL_COUNT || frame_meta->frame_num < 0)
    return FALSE;

  const gboolean source_pts_valid = GST_CLOCK_TIME_IS_VALID (frame_meta->buf_pts);
  const guint64 source_pts = source_pts_valid ? frame_meta->buf_pts : 0;
  const guint32 frame_num = (guint32) frame_meta->frame_num;
  const guint generation = (guint) g_atomic_int_get (
      &preview->fullscreen_selection_generation);
  if (generation != expected_generation)
    return FALSE;
  FullscreenFrameIdentity *identity = g_new0 (FullscreenFrameIdentity, 1);
  identity->source_id = source_id;
  identity->source_frame_num = frame_num;
  identity->source_pts = source_pts;
  identity->selection_generation = generation;
  identity->branch_pts = GST_BUFFER_PTS (buffer);
  identity->object_count = frame_meta->num_obj_meta;
  identity->source_frame_width = frame_meta->source_frame_width;
  identity->source_frame_height = frame_meta->source_frame_height;
  identity->pipeline_width = frame_meta->pipeline_width;
  identity->pipeline_height = frame_meta->pipeline_height;
  identity->source_pts_valid = source_pts_valid;
  identity->source_objects_json = serialize_fullscreen_objects (frame_meta,
      source_id, frame_num, source_pts, frame_meta->pipeline_width,
      frame_meta->pipeline_height, "post_tracker_rect_params", NULL);
  if (!identity->source_objects_json) {
    fullscreen_identity_free (identity);
    return FALSE;
  }

  g_mutex_lock (&preview->mutex);
  if (generation != (guint) g_atomic_int_get (
          &preview->fullscreen_selection_generation) ||
      source_id != (guint) g_atomic_int_get (&preview->fullscreen_selected_source)) {
    g_mutex_unlock (&preview->mutex);
    fullscreen_identity_free (identity);
    return FALSE;
  }

  const gboolean changed = !preview->fullscreen_have_source_identity ||
      preview->fullscreen_last_source_id != source_id ||
      preview->fullscreen_last_source_frame_num != (gint) frame_num ||
      (source_pts_valid && preview->fullscreen_last_source_pts != source_pts);
  if (!changed) {
    g_mutex_unlock (&preview->mutex);
    fullscreen_identity_free (identity);
    return FALSE;
  }
  preview->fullscreen_source_frame_sequence++;
  identity->source_frame_sequence = preview->fullscreen_source_frame_sequence;
  preview->fullscreen_have_source_identity = TRUE;
  preview->fullscreen_last_source_id = source_id;
  preview->fullscreen_last_source_frame_num = (gint) frame_num;
  preview->fullscreen_last_source_pts = source_pts_valid ? source_pts : 0;

  while (g_queue_get_length (&preview->fullscreen_identities) >=
      FULLSCREEN_IDENTITY_QUEUE_CAPACITY) {
    fullscreen_identity_free (g_queue_pop_head (&preview->fullscreen_identities));
    preview->fullscreen_correlation_failures++;
  }
  g_queue_push_tail (&preview->fullscreen_identities, identity);
  g_mutex_unlock (&preview->mutex);
  return TRUE;
}

static FullscreenFrameIdentity *
take_fullscreen_identity (MosaicPreview *preview, GstClockTime output_pts,
    guint selected_source)
{
  if (!GST_CLOCK_TIME_IS_VALID (output_pts))
    return NULL;
  FullscreenFrameIdentity *match = NULL;
  g_mutex_lock (&preview->mutex);
  for (GList *node = preview->fullscreen_identities.head; node; node = node->next) {
    FullscreenFrameIdentity *candidate = node->data;
    if (candidate->branch_pts != output_pts)
      continue;
    if (match) {
      match = NULL; /* duplicate PTS is ambiguous; never guess. */
      preview->fullscreen_correlation_failures++;
      goto done;
    }
    match = candidate;
  }
  if (match) {
    while (preview->fullscreen_identities.head &&
        preview->fullscreen_identities.head->data != match) {
      fullscreen_identity_free (g_queue_pop_head (&preview->fullscreen_identities));
      preview->fullscreen_correlation_failures++;
    }
    (void) g_queue_pop_head (&preview->fullscreen_identities);
  }
done:
  g_mutex_unlock (&preview->mutex);
  if (!match || match->source_id != selected_source ||
      match->selection_generation != (guint) g_atomic_int_get (
          &preview->fullscreen_selection_generation)) {
    fullscreen_identity_free (match);
    return NULL;
  }
  return match;
}

/* Correlate the actual rectangle metadata immediately before fullscreen OSD
 * with the already captured post-NvDCF source metadata. This never guesses
 * from tiler frame_num/source_id; the branch GstBuffer PTS and selection
 * generation must both match. */
static gboolean
record_fullscreen_tiler_metadata (MosaicPreview *preview,
    GstClockTime branch_pts, guint selected_source,
    const NvDsBatchMeta *output_batch,
    const NvDsFrameMeta *output_frame_meta)
{
  if (!output_batch || !output_frame_meta ||
      !GST_CLOCK_TIME_IS_VALID (branch_pts))
    return FALSE;

  const guint generation = (guint) g_atomic_int_get (
      &preview->fullscreen_selection_generation);
  gboolean matched = FALSE;
  g_mutex_lock (&preview->mutex);
  for (GList *node = preview->fullscreen_identities.head; node; node = node->next) {
    FullscreenFrameIdentity *identity = node->data;
    if (identity->branch_pts != branch_pts ||
        identity->source_id != selected_source ||
        identity->selection_generation != generation)
      continue;

    gchar *objects = serialize_fullscreen_objects (output_frame_meta,
        identity->source_id, identity->source_frame_num, identity->source_pts,
        FULLSCREEN_WIDTH, FULLSCREEN_HEIGHT,
        "post_fullscreen_tiler_pre_osd_rect_params",
        &identity->fullscreen_object_count);
    if (objects) {
      g_free (identity->fullscreen_objects_json);
      identity->fullscreen_objects_json = objects;
      identity->fullscreen_metadata_matched = TRUE;
      identity->fullscreen_metadata_generation = generation;
      identity->fullscreen_num_frames_in_batch = output_batch->num_frames_in_batch;
      identity->fullscreen_frame_object_count = output_frame_meta->num_obj_meta;
      identity->fullscreen_meta_source_id = output_frame_meta->source_id;
      identity->fullscreen_meta_frame_num = output_frame_meta->frame_num;
      identity->fullscreen_meta_pts_valid =
          GST_CLOCK_TIME_IS_VALID (output_frame_meta->buf_pts);
      identity->fullscreen_meta_frame_pts =
          identity->fullscreen_meta_pts_valid ?
              output_frame_meta->buf_pts : 0;
      identity->fullscreen_meta_pad_index = output_frame_meta->pad_index;
      matched = TRUE;
    }
    break;
  }
  g_mutex_unlock (&preview->mutex);
  return matched;
}

static void
format_source_mask (gchar *output, gsize output_size, guint32 mask)
{
  GString *ids = g_string_new (NULL);
  for (guint i = 0; i < MOSAIC_LABEL_COUNT; i++) {
    if (!(mask & (1u << i)))
      continue;
    if (ids->len)
      g_string_append_c (ids, ',');
    g_string_append_printf (ids, "%u", i);
  }
  g_strlcpy (output, ids->str, output_size);
  g_string_free (ids, TRUE);
}

/* Probe after the bounded branch queue and immediately before the valve. This
 * sees the exact batch about to be accepted/dropped by the fullscreen branch. */
static GstPadProbeReturn
fullscreen_input_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  gint pending_before = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected_before = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint target_before = pending_before >= 0 ? pending_before : selected_before;
  guint input_generation = (guint) g_atomic_int_get (
      &preview->fullscreen_selection_generation);
  if (target_before < 0 || target_before >= (gint) MOSAIC_LABEL_COUNT) {
    if (preview->fullscreen_valve)
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    return GST_PAD_PROBE_OK;
  }
  if (!buffer)
    return GST_PAD_PROBE_OK;
  const gchar *copy_miss_code = NULL;
  GstBuffer *isolated = take_fullscreen_batch_copy (preview, buffer,
      (guint) target_before, input_generation, &copy_miss_code);
  if (!isolated) {
    if (preview->fullscreen_valve)
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    if (g_atomic_int_compare_and_exchange (&preview->fullscreen_copy_miss_logged, 0, 1)) {
      g_mutex_lock (&preview->mutex);
      guint queued = g_queue_get_length (&preview->fullscreen_batch_copies);
      g_mutex_unlock (&preview->mutex);
      gint current_pending = g_atomic_int_get (&preview->fullscreen_pending_source);
      gint current_selected = g_atomic_int_get (&preview->fullscreen_selected_source);
      gint current_source = current_pending >= 0 ? current_pending : current_selected;
      gint current_generation = g_atomic_int_get (
          &preview->fullscreen_selection_generation);
      g_printerr ("FULLSCREEN_STAGE_FAIL stage=metadata_isolation "
          "code=%s source_id=%d generation=%u current_source_id=%d "
          "current_generation=%d queued=%u\n",
          copy_miss_code ? copy_miss_code : "UNKNOWN_COPY_MISS",
          target_before, input_generation, current_source, current_generation,
          queued);
    }
    return GST_PAD_PROBE_DROP;
  }
  GST_PAD_PROBE_INFO_DATA (info) = isolated;
  gst_buffer_unref (buffer);
  buffer = isolated;
  if (!fullscreen_target_is_current (preview, (guint) target_before, input_generation))
    return GST_PAD_PROBE_DROP;
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  if (!batch || !preview->fullscreen_valve) {
    gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
    gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
    gint target = pending >= 0 ? pending : selected;
    if (!batch && target >= 0 && target < (gint) MOSAIC_LABEL_COUNT &&
        g_atomic_int_compare_and_exchange (
            &preview->fullscreen_active_missing_logged, 0, 1))
      g_print ("FULLSCREEN_STAGE_FAIL stage=valve_input "
          "code=BATCH_META_MISSING source_id=%d camera_id=CAM-%02d "
          "buffer=%s\n", target, target + 1,
          buffer ? "present" : "missing");
    return GST_PAD_PROBE_OK;
  }

  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint target = pending >= 0 ? pending : selected;
  if (target < 0 || target >= (gint) MOSAIC_LABEL_COUNT) {
    g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    return GST_PAD_PROBE_OK;
  }

  guint32 source_mask = 0, pad_mask = 0;
  guint frame_meta_count = 0, target_count = 0;
  NvDsFrameMeta *target_meta = NULL;
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
    if (!frame_meta)
      continue;
    frame_meta_count++;
    if (frame_meta->source_id < 32)
      source_mask |= (1u << frame_meta->source_id);
    if (frame_meta->pad_index < 32)
      pad_mask |= (1u << frame_meta->pad_index);
    if (frame_meta->source_id == (guint) target) {
      target_count++;
      target_meta = frame_meta;
    }
  }

  gchar sources[64], pads[64];
  format_source_mask (sources, sizeof sources, source_mask);
  format_source_mask (pads, sizeof pads, pad_mask);

  if (g_atomic_int_compare_and_exchange (
          &preview->fullscreen_input_metadata_logged, 0, 1)) {
    g_print ("FULLSCREEN_STAGE stage=valve_input status=metadata_observed "
        "batch_frames=%u frame_meta_count=%u source_ids=%s pad_indexes=%s\n",
        batch->num_frames_in_batch,
        frame_meta_count, sources, pads);
  }

  if (target_count == 0) {
    g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    guint distinct_sources = 0;
    for (guint i = 0; i < MOSAIC_LABEL_COUNT; i++)
      if (source_mask & (1u << i))
        distinct_sources++;
    gint absent_batches = g_atomic_int_add (
        &preview->fullscreen_pending_misses, 1) + 1;
    if (absent_batches == 1 || absent_batches % 40 == 0)
      g_print ("FULLSCREEN_STAGE stage=valve_input status=%s "
          "source_id=%d camera_id=CAM-%02d generation=%u "
          "selection_pending=%s absent_batches=%d batch_frames=%u "
          "frame_meta_count=%u distinct_source_count=%u source_ids=%s "
          "pad_indexes=%s\n",
          distinct_sources < MOSAIC_LABEL_COUNT ?
              "partial_batch_target_absent" : "full_batch_target_absent",
          target, target + 1, input_generation, pending >= 0 ? "true" : "false",
          absent_batches, batch->num_frames_in_batch, frame_meta_count,
          distinct_sources, sources, pads);
    return GST_PAD_PROBE_OK;
  }
  if (target_count > 1 || !target_meta) {
    g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_active_missing_logged, 0, 1))
      g_printerr ("FULLSCREEN_STAGE_FAIL stage=valve_input "
          "code=DUPLICATE_TARGET_FRAME_META source_id=%d camera_id=CAM-%02d "
          "generation=%u target_meta_count=%u batch_frames=%u\n", target,
          target + 1, input_generation, target_count, batch->num_frames_in_batch);
    return GST_PAD_PROBE_OK;
  }
  gint previous_absent_batches = g_atomic_int_get (
      &preview->fullscreen_pending_misses);
  if (previous_absent_batches > 0)
    g_print ("FULLSCREEN_STAGE stage=valve_input "
        "status=selected_source_observed_after_absence source_id=%d "
        "camera_id=CAM-%02d generation=%u absent_batches=%d "
        "batch_frames=%u source_ids=%s\n", target, target + 1,
        input_generation, previous_absent_batches, batch->num_frames_in_batch,
        sources);
  g_atomic_int_set (&preview->fullscreen_pending_misses, 0);

  /* The selected-source tiler must first consume a batch containing this
   * source. Open the valve for exactly that priming batch. The output probe
   * acknowledges it by matching the real GstBuffer PTS. */
  if (pending >= 0 &&
      g_atomic_int_get (&preview->fullscreen_tiler_primed_source) != pending) {
    /* Only one priming buffer may be in flight. The tiler acknowledgement is
     * asynchronous; passing later batches would overwrite the PTS correlation
     * key before the first output can be matched. */
    if (g_atomic_int_get (&preview->fullscreen_prime_requested_source) == pending) {
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
      return GST_PAD_PROBE_OK;
    }
    if (!GST_BUFFER_PTS_IS_VALID (buffer)) {
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
      g_print ("FULLSCREEN_STAGE_FAIL stage=valve_input "
          "code=PRIME_BUFFER_PTS_INVALID source_id=%d camera_id=CAM-%02d\n",
          pending, pending + 1);
      return GST_PAD_PROBE_OK;
    }
    if (!g_atomic_int_compare_and_exchange (
            &preview->fullscreen_prime_requested_source, -1, pending)) {
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
      return GST_PAD_PROBE_OK;
    }
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_prime_input_pts = GST_BUFFER_PTS (buffer);
    preview->fullscreen_prime_input_pts_valid = TRUE;
    g_mutex_unlock (&preview->mutex);
    g_object_set (preview->fullscreen_valve, "drop", FALSE, NULL);
    g_print ("FULLSCREEN_STAGE stage=valve_input status=prime_pass "
        "source_id=%d camera_id=CAM-%02d generation=%d frame_num=%d "
        "source_pts_ns=%" G_GUINT64_FORMAT " branch_pts_ns=%" G_GUINT64_FORMAT
        " pad_index=%u batch_frames=%u source_ids=%s pad_indexes=%s\n",
        pending, pending + 1,
        g_atomic_int_get (&preview->fullscreen_selection_generation),
        target_meta->frame_num, target_meta->buf_pts,
        GST_BUFFER_PTS (buffer), target_meta->pad_index,
        batch->num_frames_in_batch, sources, pads);
    return GST_PAD_PROBE_OK;
  }

  if (pending >= 0) {
    gint actual_source = -1;
    g_object_set (preview->fullscreen_tiler, "show-source", pending, NULL);
    g_object_get (preview->fullscreen_tiler, "show-source", &actual_source, NULL);
    if (actual_source != pending) {
      g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
      g_print ("FULLSCREEN_STAGE_FAIL stage=show_source "
          "code=SELECTION_NOT_APPLIED requested_source_id=%d actual_source_id=%d\n",
          pending, actual_source);
      return GST_PAD_PROBE_OK;
    }
    g_atomic_int_set (&preview->fullscreen_selected_source, pending);
    g_atomic_int_set (&preview->fullscreen_pending_source, -1);
    g_atomic_int_set (&preview->fullscreen_active_missing_logged, 0);
    g_atomic_int_set (&preview->fullscreen_osd_output_logged, 0);
    g_atomic_int_set (&preview->fullscreen_appsink_logged, 0);
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_source_frame_sequence = 0;
    preview->fullscreen_have_source_identity = FALSE;
    preview->fullscreen_last_source_id = (guint) pending;
    preview->fullscreen_last_source_frame_num = -1;
    preview->fullscreen_last_source_pts = 0;
    g_mutex_unlock (&preview->mutex);
    g_print ("FULLSCREEN_STAGE stage=show_source status=applied "
        "requested_source_id=%d actual_source_id=%d camera_id=CAM-%02d\n",
        pending, actual_source, pending + 1);
    selected = pending;
  }

  if (!queue_fullscreen_identity (preview, target_meta, buffer, input_generation)) {
    g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_correlation_failures++;
    g_mutex_unlock (&preview->mutex);
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_identity_fail_logged, 0, 1))
      g_print ("FULLSCREEN_STAGE_FAIL stage=valve_input code=%s source_id=%d "
          "camera_id=CAM-%02d frame_num=%d source_pts_valid=%s "
          "branch_pts_valid=%s\n",
          GST_BUFFER_PTS_IS_VALID (buffer) ?
              "SOURCE_FRAME_IDENTITY_NOT_PROGRESSING" : "INPUT_BUFFER_PTS_INVALID",
          selected, selected + 1, target_meta->frame_num,
          GST_CLOCK_TIME_IS_VALID (target_meta->buf_pts) ? "true" : "false",
          GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false");
    return GST_PAD_PROBE_OK;
  }
  g_object_set (preview->fullscreen_valve, "drop", FALSE, NULL);
  preview->fullscreen_input_matches++;
  if (g_atomic_int_compare_and_exchange (
          &preview->fullscreen_selection_input_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=source_selection status=input_correlated "
        "source_id=%d camera_id=CAM-%02d generation=%d frame_num=%d "
        "source_pts_ns=%" G_GUINT64_FORMAT " branch_pts_ns=%" G_GUINT64_FORMAT
        " pad_index=%u\n", selected, selected + 1,
        g_atomic_int_get (&preview->fullscreen_selection_generation),
        target_meta->frame_num, target_meta->buf_pts, GST_BUFFER_PTS (buffer),
        target_meta->pad_index);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
fullscreen_tiler_input_probe (GstPad *pad, GstPadProbeInfo *info,
    gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  if (!batch) {
    gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
    gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
    gint target = pending >= 0 ? pending : selected;
    if (target >= 0 && target < (gint) MOSAIC_LABEL_COUNT &&
        g_atomic_int_compare_and_exchange (
            &preview->fullscreen_active_missing_logged, 0, 1))
      g_print ("FULLSCREEN_STAGE_FAIL stage=tiler_input "
          "code=BATCH_META_MISSING source_id=%d camera_id=CAM-%02d "
          "buffer=%s\n", target, target + 1,
          buffer ? "present" : "missing");
    return GST_PAD_PROBE_OK;
  }

  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint target = pending >= 0 ? pending : selected;
  guint32 source_mask = 0, pad_mask = 0;
  guint frame_meta_count = 0, target_count = 0;
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
    if (!frame_meta)
      continue;
    frame_meta_count++;
    if (frame_meta->source_id < 32)
      source_mask |= 1u << frame_meta->source_id;
    if (frame_meta->pad_index < 32)
      pad_mask |= 1u << frame_meta->pad_index;
    if ((gint) frame_meta->source_id == target)
      target_count++;
  }

  g_mutex_lock (&preview->mutex);
  preview->fullscreen_tiler_inputs++;
  g_mutex_unlock (&preview->mutex);
  if (target >= 0 && target < (gint) MOSAIC_LABEL_COUNT &&
      g_atomic_int_compare_and_exchange (
          &preview->fullscreen_tiler_input_logged, 0, 1)) {
    gchar sources[64], pads[64];
    format_source_mask (sources, sizeof sources, source_mask);
    format_source_mask (pads, sizeof pads, pad_mask);
    g_print ("FULLSCREEN_STAGE stage=tiler_input status=buffer "
        "source_id=%d camera_id=CAM-%02d batch_frames=%u frame_meta_count=%u "
        "target_meta_count=%u source_ids=%s pad_indexes=%s "
        "pts_valid=%s pts_ns=%" G_GUINT64_FORMAT "\n", target,
        target + 1, batch->num_frames_in_batch, frame_meta_count, target_count,
        sources, pads, GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false",
        GST_BUFFER_PTS_IS_VALID (buffer) ? GST_BUFFER_PTS (buffer) : 0);
  }
  guint distinct_sources = 0;
  for (guint i = 0; i < MOSAIC_LABEL_COUNT; i++)
    if (source_mask & (1u << i))
      distinct_sources++;
  if (target >= 0 && target < (gint) MOSAIC_LABEL_COUNT && target_count == 0 &&
      g_atomic_int_compare_and_exchange (
          &preview->fullscreen_active_missing_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=tiler_input "
        "status=%s "
        "source_id=%d camera_id=CAM-%02d generation=%d batch_frames=%u "
        "frame_meta_count=%u source_mask=%u pad_mask=%u\n",
        distinct_sources < MOSAIC_LABEL_COUNT ?
            "partial_batch_target_absent" : "full_batch_target_absent",
        target,
        target + 1, g_atomic_int_get (&preview->fullscreen_selection_generation),
        batch->num_frames_in_batch, frame_meta_count, source_mask, pad_mask);
  if (target >= 0 && target < (gint) MOSAIC_LABEL_COUNT && target_count > 1 &&
      g_atomic_int_compare_and_exchange (
          &preview->fullscreen_active_missing_logged, 0, 1))
    g_printerr ("FULLSCREEN_STAGE_FAIL stage=tiler_input "
        "code=DUPLICATE_TARGET_FRAME_META source_id=%d camera_id=CAM-%02d "
        "generation=%d batch_frames=%u frame_meta_count=%u "
        "target_meta_count=%u\n", target, target + 1,
        g_atomic_int_get (&preview->fullscreen_selection_generation),
        batch->num_frames_in_batch, frame_meta_count, target_count);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
fullscreen_label_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  gint prime_requested = g_atomic_int_get (
      &preview->fullscreen_prime_requested_source);
  GstClockTime output_pts = buffer ? GST_BUFFER_PTS (buffer) : GST_CLOCK_TIME_NONE;
  GstClockTime prime_pts = GST_CLOCK_TIME_NONE;
  gboolean prime_pts_valid;
  g_mutex_lock (&preview->mutex);
  prime_pts_valid = preview->fullscreen_prime_input_pts_valid;
  if (prime_pts_valid)
    prime_pts = preview->fullscreen_prime_input_pts;
  g_mutex_unlock (&preview->mutex);
  if (batch) {
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_tiler_outputs++;
    g_mutex_unlock (&preview->mutex);
    if (pending >= 0 && prime_requested == pending && prime_pts_valid &&
        GST_CLOCK_TIME_IS_VALID (output_pts) &&
        output_pts == prime_pts) {
      g_atomic_int_set (&preview->fullscreen_tiler_primed_source, pending);
      g_mutex_lock (&preview->mutex);
      preview->fullscreen_prime_input_pts_valid = FALSE;
      g_mutex_unlock (&preview->mutex);
      g_print ("FULLSCREEN_STAGE stage=tiler_output status=prime_ack "
          "source_id=%d camera_id=CAM-%02d branch_pts_ns=%" G_GUINT64_FORMAT "\n",
          pending, pending + 1, output_pts);
    } else if (pending >= 0 && prime_requested == pending && prime_pts_valid &&
        g_atomic_int_compare_and_exchange (
            &preview->fullscreen_prime_mismatch_logged, 0, 1)) {
      g_print ("FULLSCREEN_STAGE_FAIL stage=tiler_output "
          "code=PRIME_BUFFER_PTS_MISMATCH source_id=%d "
          "input_pts_ns=%" G_GUINT64_FORMAT " output_pts_valid=%s "
          "output_pts_ns=%" G_GUINT64_FORMAT "\n", pending, prime_pts,
          GST_CLOCK_TIME_IS_VALID (output_pts) ? "true" : "false",
          GST_CLOCK_TIME_IS_VALID (output_pts) ? output_pts : 0);
    }
  } else if (pending >= 0 && prime_requested == pending &&
      g_atomic_int_compare_and_exchange (
          &preview->fullscreen_prime_mismatch_logged, 0, 1)) {
    g_print ("FULLSCREEN_STAGE_FAIL stage=tiler_output "
        "code=OUTPUT_BATCH_META_MISSING source_id=%d camera_id=CAM-%02d "
        "output_buffer=%s output_pts_valid=%s\n", pending, pending + 1,
        buffer ? "present" : "missing",
        buffer && GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false");
  }
  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  if (!batch || selected < 0 || selected >= (gint) MOSAIC_LABEL_COUNT)
    return GST_PAD_PROBE_OK;
  NvDsFrameMeta *frame_meta = batch->frame_meta_list ?
      (NvDsFrameMeta *) batch->frame_meta_list->data : NULL;
  if (!frame_meta) {
    g_print ("FULLSCREEN_STAGE_FAIL stage=tiler_output "
        "code=OUTPUT_FRAME_META_MISSING source_id=%d camera_id=CAM-%02d\n",
        selected, selected + 1);
    return GST_PAD_PROBE_OK;
  }

  /* Record the selected tiler's actual transformed object rectangles before
   * NvDsOSD changes or renders text. The identity must correlate to the
   * original post-NvDCF frame by exact branch PTS and selection generation. */
  if (pending < 0 && GST_CLOCK_TIME_IS_VALID (output_pts)) {
    if (record_fullscreen_tiler_metadata (preview, output_pts,
            (guint) selected, batch, frame_meta)) {
      if (g_atomic_int_compare_and_exchange (
              &preview->fullscreen_metadata_logged, 0, 1))
        g_print ("FULLSCREEN_STAGE stage=tiler_metadata status=correlated "
            "source_id=%d camera_id=CAM-%02d generation=%d branch_pts_ns=%"
            G_GUINT64_FORMAT " output_source_id=%u output_frame_num=%d "
            "output_frame_pts_valid=%s output_frame_pts_ns=%"
            G_GUINT64_FORMAT " output_pad_index=%u batch_frames=%u "
            "objects=%u\n", selected, selected + 1,
            g_atomic_int_get (&preview->fullscreen_selection_generation),
            output_pts, frame_meta->source_id, frame_meta->frame_num,
            GST_CLOCK_TIME_IS_VALID (frame_meta->buf_pts) ? "true" : "false",
            GST_CLOCK_TIME_IS_VALID (frame_meta->buf_pts) ? frame_meta->buf_pts : 0,
            frame_meta->pad_index, batch->num_frames_in_batch,
            frame_meta->num_obj_meta);
    } else {
      g_mutex_lock (&preview->mutex);
      preview->fullscreen_correlation_failures++;
      g_mutex_unlock (&preview->mutex);
      if (g_atomic_int_compare_and_exchange (
              &preview->fullscreen_metadata_fail_logged, 0, 1))
        g_print ("FULLSCREEN_STAGE_FAIL stage=tiler_metadata "
            "code=NO_EXACT_POST_TRACKER_FRAME_MATCH source_id=%d "
            "camera_id=CAM-%02d generation=%d branch_pts_ns=%"
            G_GUINT64_FORMAT " output_source_id=%u output_frame_num=%d\n",
            selected, selected + 1,
            g_atomic_int_get (&preview->fullscreen_selection_generation),
            output_pts, frame_meta->source_id, frame_meta->frame_num);
    }
  }

  /* show-source renders one full-resolution stream. The transformed output
   * frame metadata may describe the tiler output rather than the original
   * source ID, so attach its human label to the output canvas metadata. */
  if (g_atomic_int_compare_and_exchange (
          &preview->fullscreen_tiler_output_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=tiler_output status=metadata_observed "
        "selected_source_id=%d camera_id=CAM-%02d num_frames_in_batch=%u "
        "output_frame_source_id=%u output_frame_num=%d output_frame_pts=%"
        G_GUINT64_FORMAT " branch_pts_ns=%" G_GUINT64_FORMAT "\n",
        selected, selected + 1, batch->num_frames_in_batch,
        frame_meta->source_id, frame_meta->frame_num, frame_meta->buf_pts,
        output_pts);

  NvDsDisplayMeta *display = nvds_acquire_display_meta_from_pool (batch);
  if (!display) {
    g_print ("FULLSCREEN_STAGE_FAIL stage=osd_input "
        "code=DISPLAY_META_POOL_EMPTY source_id=%d camera_id=CAM-%02d\n",
        selected, selected + 1);
    return GST_PAD_PROBE_OK;
  }
  display->num_labels = 1;
  NvOSD_TextParams *label = &display->text_params[0];
  label->display_text = g_strdup (camera_labels[selected]);
  label->x_offset = 16;
  label->y_offset = 12;
  label->font_params.font_name = "Serif";
  label->font_params.font_size = 24;
  label->font_params.font_color = (NvOSD_ColorParams) { 1.0, 1.0, 1.0, 1.0 };
  label->set_bg_clr = 1;
  label->text_bg_clr = (NvOSD_ColorParams) { 0.0, 0.0, 0.0, 0.8 };
  nvds_add_display_meta_to_frame (frame_meta, display);
  if (g_atomic_int_compare_and_exchange (
          &preview->fullscreen_label_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=osd_input status=label_attached "
        "source_id=%d camera_id=CAM-%02d\n", selected, selected + 1);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
fullscreen_osd_output_probe (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  MosaicPreview *preview = user_data;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  if (!buffer)
    return GST_PAD_PROBE_OK;
  g_mutex_lock (&preview->mutex);
  preview->fullscreen_osd_outputs++;
  g_mutex_unlock (&preview->mutex);
  if (g_atomic_int_compare_and_exchange (
          &preview->fullscreen_osd_output_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=osd_output status=buffer "
        "source_id=%d branch_pts_valid=%s branch_pts_ns=%" G_GUINT64_FORMAT "\n",
        g_atomic_int_get (&preview->fullscreen_selected_source),
        GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false",
        GST_BUFFER_PTS_IS_VALID (buffer) ? GST_BUFFER_PTS (buffer) : 0);
  return GST_PAD_PROBE_OK;
}

static GstFlowReturn
fullscreen_new_sample (GstAppSink *sink, gpointer user_data)
{
  MosaicPreview *preview = user_data;
  GstSample *sample = gst_app_sink_pull_sample (sink);
  if (!sample)
    return GST_FLOW_EOS;
  GstCaps *caps = gst_sample_get_caps (sample);
  GstBuffer *buffer = gst_sample_get_buffer (sample);
  GstVideoInfo info;
  if (buffer) {
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_appsink_samples++;
    g_mutex_unlock (&preview->mutex);
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_appsink_logged, 0, 1))
      g_print ("FULLSCREEN_STAGE stage=appsink status=sample "
          "source_id=%d pts_valid=%s pts_ns=%" G_GUINT64_FORMAT "\n",
          g_atomic_int_get (&preview->fullscreen_selected_source),
          GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false",
          GST_BUFFER_PTS_IS_VALID (buffer) ? GST_BUFFER_PTS (buffer) : 0);
  }
  gboolean valid_caps = caps && buffer && gst_video_info_from_caps (&info, caps) &&
      GST_VIDEO_INFO_FORMAT (&info) == GST_VIDEO_FORMAT_BGRx &&
      GST_VIDEO_INFO_WIDTH (&info) == FULLSCREEN_WIDTH &&
      GST_VIDEO_INFO_HEIGHT (&info) == FULLSCREEN_HEIGHT &&
      gst_buffer_get_size (buffer) >= GST_VIDEO_INFO_SIZE (&info);
  const GstCapsFeatures *features = caps ? gst_caps_get_features (caps, 0) : NULL;
  if (!valid_caps || (features &&
      gst_caps_features_contains (features, "memory:NVMM"))) {
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_correlation_failures++;
    g_mutex_unlock (&preview->mutex);
    g_print ("FULLSCREEN_STAGE_FAIL stage=appsink code=CAPS_OR_MEMORY_INVALID "
        "caps=%s format=%s width=%d height=%d\n",
        caps ? "present" : "missing",
        caps && gst_video_info_from_caps (&info, caps) ?
            gst_video_format_to_string (GST_VIDEO_INFO_FORMAT (&info)) : "unknown",
        caps && gst_video_info_from_caps (&info, caps) ?
            GST_VIDEO_INFO_WIDTH (&info) : 0,
        caps && gst_video_info_from_caps (&info, caps) ?
            GST_VIDEO_INFO_HEIGHT (&info) : 0);
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }

  gint selected = g_atomic_int_get (&preview->fullscreen_selected_source);
  if (selected < 0 || selected >= (gint) MOSAIC_LABEL_COUNT) {
    g_print ("FULLSCREEN_STAGE stage=appsink status=discarded_no_selection\n");
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }

  GstVideoFrame frame;
  memset (&frame, 0, sizeof frame);
  if (!gst_video_frame_map (&frame, &info, buffer, GST_MAP_READ)) {
    g_print ("FULLSCREEN_STAGE_FAIL stage=appsink code=FRAME_MAP_FAILED "
        "source_id=%d\n", selected);
    gst_sample_unref (sample);
    return GST_FLOW_OK;
  }
  gint stride = GST_VIDEO_FRAME_PLANE_STRIDE (&frame, 0);
  gsize offset = GST_VIDEO_INFO_PLANE_OFFSET (&info, 0);
  GstVideoMeta *meta = gst_buffer_get_video_meta (buffer);
  gboolean layout_ok = stride >= (gint) FULLSCREEN_STRIDE;
  if (meta) {
    layout_ok = layout_ok && meta->format == GST_VIDEO_FORMAT_BGRx &&
        meta->width == FULLSCREEN_WIDTH && meta->height == FULLSCREEN_HEIGHT &&
        meta->n_planes >= 1 && meta->stride[0] == stride;
    offset = meta->offset[0];
  }
  FullscreenFrameIdentity *identity = layout_ok ? take_fullscreen_identity (
      preview, GST_BUFFER_PTS (buffer), (guint) selected) : NULL;
  if (!layout_ok || !identity) {
    g_mutex_lock (&preview->mutex);
    preview->fullscreen_correlation_failures++;
    g_mutex_unlock (&preview->mutex);
    g_print ("FULLSCREEN_STAGE_FAIL stage=%s code=%s source_id=%d "
        "camera_id=CAM-%02d output_pts_valid=%s output_pts_ns=%"
        G_GUINT64_FORMAT " stride=%d\n",
        layout_ok ? "buffer_correlation" : "appsink_layout",
        layout_ok ? "NO_UNIQUE_MATCHING_PRETILER_FRAME" : "BGRX_LAYOUT_INVALID",
        selected, selected + 1,
        GST_BUFFER_PTS_IS_VALID (buffer) ? "true" : "false",
        GST_BUFFER_PTS_IS_VALID (buffer) ? GST_BUFFER_PTS (buffer) : 0,
        stride);
  }
  gboolean published = identity && publish_fullscreen_frame (preview,
      GST_VIDEO_FRAME_PLANE_DATA (&frame, 0), stride, offset,
      gst_buffer_get_size (buffer), identity);
  if (identity && published && g_atomic_int_compare_and_exchange (
          &preview->fullscreen_publication_logged, 0, 1))
    g_print ("FULLSCREEN_STAGE stage=ipc_publish status=published "
        "source_id=%u camera_id=CAM-%02u sequence=%" G_GUINT64_FORMAT
        " source_frame_sequence=%" G_GUINT64_FORMAT " frame_num=%u "
        "source_pts_ns=%" G_GUINT64_FORMAT "\n", identity->source_id,
        identity->source_id + 1, preview->fullscreen_sequence,
        identity->source_frame_sequence, identity->source_frame_num,
        identity->source_pts);
  if (identity && !published)
    g_print ("FULLSCREEN_STAGE_FAIL stage=ipc_publish code=PUBLISH_REJECTED "
        "source_id=%u generation=%u\n", identity->source_id,
        identity->selection_generation);
  fullscreen_identity_free (identity);
  gst_video_frame_unmap (&frame);
  gst_sample_unref (sample);
  return GST_FLOW_OK;
}

static gboolean
read_fullscreen_selection (gint *selected)
{
  char text[32] = { 0 };
  int fd = open (FULLSCREEN_REQUEST_PATH, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    if (errno == ENOENT) {
      *selected = -1;
      return TRUE;
    }
    return FALSE;
  }
  struct stat info;
  gboolean valid = fstat (fd, &info) == 0 && S_ISREG (info.st_mode) &&
      info.st_uid == getuid () && (info.st_mode & 077) == 0 &&
      info.st_size >= 0 && info.st_size < (off_t) sizeof text;
  ssize_t count = valid ? read (fd, text, sizeof text - 1) : -1;
  close (fd);
  if (!valid || count < 0)
    return FALSE;
  text[count] = '\0';
  g_strstrip (text);
  if (!text[0]) {
    *selected = -1;
    return TRUE;
  }
  guint camera_id = 0;
  gchar tail = '\0';
  if (sscanf (text, "CAM-%02u%c", &camera_id, &tail) != 1 ||
      camera_id < 1 || camera_id > MOSAIC_LABEL_COUNT)
    return FALSE;
  *selected = (gint) camera_id - 1;
  return TRUE;
}

static gboolean
fullscreen_request_tick (gpointer user_data)
{
  MosaicPreview *preview = user_data;
  gint selected = -1;
  if (!read_fullscreen_selection (&selected)) {
    if (g_atomic_int_compare_and_exchange (
            &preview->fullscreen_active_missing_logged, 0, 1))
      g_printerr ("FULLSCREEN_STAGE_FAIL stage=request_reader "
          "code=REQUEST_FILE_INVALID\n");
    return G_SOURCE_CONTINUE;
  }
  gint previous = g_atomic_int_get (&preview->fullscreen_selected_source);
  gint pending = g_atomic_int_get (&preview->fullscreen_pending_source);
  if ((selected >= 0 && (selected == previous || selected == pending)) ||
      (selected < 0 && previous < 0 && pending < 0))
    return G_SOURCE_CONTINUE;
  g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
  g_atomic_int_set (&preview->fullscreen_selected_source, -1);
  g_atomic_int_set (&preview->fullscreen_pending_source, -1);
  gint generation = g_atomic_int_add (
      &preview->fullscreen_selection_generation, 1) + 1;
  g_mutex_lock (&preview->mutex);
  clear_fullscreen_identities_locked (preview);
  clear_fullscreen_batch_copies_locked (preview);
  preview->fullscreen_source_frame_sequence = 0;
  preview->fullscreen_have_source_identity = FALSE;
  preview->fullscreen_last_source_id = selected >= 0 ? (guint) selected : 0;
  preview->fullscreen_last_source_frame_num = -1;
  preview->fullscreen_last_source_pts = 0;
  g_mutex_unlock (&preview->mutex);
  g_atomic_int_set (&preview->fullscreen_tiler_primed_source, -1);
  g_atomic_int_set (&preview->fullscreen_prime_requested_source, -1);
  g_atomic_int_set (&preview->fullscreen_pending_misses, 0);
  g_atomic_int_set (&preview->fullscreen_active_missing_logged, 0);
  g_atomic_int_set (&preview->fullscreen_prime_mismatch_logged, 0);
  g_atomic_int_set (&preview->fullscreen_tiler_input_logged, 0);
  g_atomic_int_set (&preview->fullscreen_selection_input_logged, 0);
  g_atomic_int_set (&preview->fullscreen_identity_fail_logged, 0);
  g_atomic_int_set (&preview->fullscreen_tiler_output_logged, 0);
  g_atomic_int_set (&preview->fullscreen_metadata_logged, 0);
  g_atomic_int_set (&preview->fullscreen_metadata_fail_logged, 0);
  g_atomic_int_set (&preview->fullscreen_label_logged, 0);
  g_atomic_int_set (&preview->fullscreen_osd_output_logged, 0);
  g_atomic_int_set (&preview->fullscreen_appsink_logged, 0);
  g_atomic_int_set (&preview->fullscreen_publication_logged, 0);
  for (guint i = 0; i < FULLSCREEN_COPY_FAILURE_COUNT; i++)
    g_atomic_int_set (&preview->fullscreen_copy_failure_logged[i], 0);
  g_atomic_int_set (&preview->fullscreen_copy_miss_logged, 0);
  g_atomic_int_set (&preview->fullscreen_copy_queue_drop_logged, 0);
  g_atomic_int_set (&preview->fullscreen_copy_selection_race_logged, 0);
  g_mutex_lock (&preview->mutex);
  preview->fullscreen_prime_input_pts_valid = FALSE;
  g_mutex_unlock (&preview->mutex);

  if (selected < 0) {
    g_object_set (preview->fullscreen_tiler, "show-source", -1, NULL);
    g_print ("FULLSCREEN_STAGE stage=request_reader status=acknowledged "
        "selection=cleared generation=%d\n", generation);
  } else {
    g_atomic_int_set (&preview->fullscreen_pending_source, selected);
    g_print ("FULLSCREEN_STAGE stage=request_reader status=acknowledged "
        "source_id=%d camera_id=CAM-%02d generation=%d\n", selected,
        selected + 1, generation);
  }
  return G_SOURCE_CONTINUE;
}

static gboolean
add_fullscreen_branch (AppCtx *app_ctx, MosaicPreview *preview)
{
  GstElement *pipeline = app_ctx->pipeline.pipeline;
  GstElement *tee = app_ctx->pipeline.common_elements.tee;
  if (!pipeline || !tee || app_ctx->config.num_source_sub_bins != 6 ||
      !open_fullscreen_slot (preview))
    return FALSE;

  preview->fullscreen_queue = gst_element_factory_make ("queue", "fullscreen-preview-queue");
  preview->fullscreen_valve = gst_element_factory_make ("valve", "fullscreen-preview-valve");
  preview->fullscreen_tiler = gst_element_factory_make ("nvmultistreamtiler",
      "fullscreen-preview-tiler");
  preview->fullscreen_convert = gst_element_factory_make ("nvvideoconvert",
      "fullscreen-preview-system-convert");
  preview->fullscreen_caps_filter = gst_element_factory_make ("capsfilter",
      "fullscreen-preview-bgrx-caps");
  preview->fullscreen_app_sink = gst_element_factory_make ("appsink",
      "fullscreen-preview-appsink");
  if (!preview->fullscreen_queue || !preview->fullscreen_valve ||
      !preview->fullscreen_tiler || !preview->fullscreen_convert ||
      !preview->fullscreen_caps_filter || !preview->fullscreen_app_sink ||
      !create_osd_bin (&app_ctx->config.osd_config, &preview->fullscreen_osd_bin))
    return FALSE;

  if (!g_object_class_find_property (G_OBJECT_GET_CLASS (preview->fullscreen_tiler),
          "show-source"))
    return FALSE;
  g_object_set (preview->fullscreen_queue, "max-size-buffers", 1u,
      "max-size-bytes", 0u, "max-size-time", (guint64) 0,
      "leaky", 2, NULL);
  g_object_set (preview->fullscreen_valve, "drop", TRUE, NULL);
  g_object_set (preview->fullscreen_tiler, "width", FULLSCREEN_WIDTH,
      "height", FULLSCREEN_HEIGHT, "rows", 1u, "columns", 1u,
      "gpu-id", app_ctx->config.tiled_display_config.gpu_id,
      "nvbuf-memory-type", app_ctx->config.tiled_display_config.nvbuf_memory_type,
      "show-source", -1, NULL);
  g_object_set (preview->fullscreen_convert, "gpu-id",
      app_ctx->config.osd_config.gpu_id, NULL);
  g_object_set (preview->fullscreen_app_sink, "max-buffers", 1u,
      "drop", TRUE, "sync", FALSE, "async", FALSE,
      "enable-last-sample", FALSE, "wait-on-eos", FALSE, NULL);

  GstCaps *caps = gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, "BGRx",
      "width", G_TYPE_INT, (gint) FULLSCREEN_WIDTH,
      "height", G_TYPE_INT, (gint) FULLSCREEN_HEIGHT, NULL);
  gst_caps_set_features (caps, 0,
      gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY, NULL));
  g_object_set (preview->fullscreen_caps_filter, "caps", caps, NULL);
  gst_caps_unref (caps);

  gst_bin_add_many (GST_BIN (pipeline), preview->fullscreen_queue,
      preview->fullscreen_valve, preview->fullscreen_tiler,
      preview->fullscreen_osd_bin.bin, preview->fullscreen_convert,
      preview->fullscreen_caps_filter, preview->fullscreen_app_sink, NULL);
  GstAppSinkCallbacks callbacks = { 0 };
  callbacks.new_sample = fullscreen_new_sample;
  gst_app_sink_set_callbacks (GST_APP_SINK (preview->fullscreen_app_sink),
      &callbacks, preview, NULL);

  if (!gst_element_link_many (preview->fullscreen_queue,
          preview->fullscreen_valve, preview->fullscreen_tiler,
          preview->fullscreen_osd_bin.bin, preview->fullscreen_convert,
          preview->fullscreen_caps_filter, preview->fullscreen_app_sink, NULL))
    return FALSE;
  preview->fullscreen_valve_sink_pad = gst_element_get_static_pad (
      preview->fullscreen_valve, "sink");
  preview->fullscreen_tiler_sink_pad = gst_element_get_static_pad (
      preview->fullscreen_tiler, "sink");
  if (!preview->fullscreen_valve_sink_pad || !preview->fullscreen_tiler_sink_pad)
    return FALSE;
  preview->fullscreen_tee_pad = gst_element_request_pad_simple (tee, "src_%u");
  preview->fullscreen_queue_sink_pad = gst_element_get_static_pad (
      preview->fullscreen_queue, "sink");
  if (!preview->fullscreen_tee_pad || !preview->fullscreen_queue_sink_pad ||
      gst_pad_link (preview->fullscreen_tee_pad,
          preview->fullscreen_queue_sink_pad) != GST_PAD_LINK_OK)
    return FALSE;

  preview->fullscreen_input_probe_id = gst_pad_add_probe (
      preview->fullscreen_valve_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      fullscreen_input_probe, preview, NULL);
  if (!preview->fullscreen_input_probe_id)
    return FALSE;

  preview->fullscreen_tiler_input_probe_id = gst_pad_add_probe (
      preview->fullscreen_tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      fullscreen_tiler_input_probe, preview, NULL);
  if (!preview->fullscreen_tiler_input_probe_id)
    return FALSE;

  GstPad *osd_sink_pad = gst_element_get_static_pad (
      preview->fullscreen_osd_bin.nvosd, "sink");
  if (!osd_sink_pad)
    return FALSE;
  preview->fullscreen_label_probe_id = gst_pad_add_probe (osd_sink_pad,
      GST_PAD_PROBE_TYPE_BUFFER, fullscreen_label_probe, preview, NULL);
  gst_object_unref (osd_sink_pad);
  if (!preview->fullscreen_label_probe_id)
    return FALSE;

  preview->fullscreen_osd_src_pad = gst_element_get_static_pad (
      preview->fullscreen_osd_bin.bin, "src");
  if (!preview->fullscreen_osd_src_pad)
    return FALSE;
  preview->fullscreen_osd_src_probe_id = gst_pad_add_probe (
      preview->fullscreen_osd_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      fullscreen_osd_output_probe, preview, NULL);
  if (!preview->fullscreen_osd_src_probe_id)
    return FALSE;
  preview->fullscreen_analytics_sink_pad = gst_element_get_static_pad (tee, "sink");
  if (!preview->fullscreen_analytics_sink_pad)
    return FALSE;
  preview->fullscreen_analytics_probe_id = gst_pad_add_probe (
      preview->fullscreen_analytics_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      fullscreen_analytics_probe, preview, NULL);
  if (!preview->fullscreen_analytics_probe_id)
    return FALSE;
  preview->fullscreen_request_timer_id = g_timeout_add (25,
      fullscreen_request_tick, preview);
  return preview->fullscreen_request_timer_id != 0;
}

static gboolean
add_display_branch (AppCtx *app_ctx, MosaicPreview *preview)
{
  NvDsInstanceBin *instance = &app_ctx->pipeline.instance_bins[0];
  if (!instance->bin || !instance->osd_bin.bin || !instance->osd_bin.nvosd ||
      !instance->sink_bin.bin || app_ctx->config.num_source_sub_bins != 6 ||
      app_ctx->config.num_sink_sub_bins != 1 ||
      app_ctx->config.sink_bin_sub_bin_config[0].type != NV_DS_SINK_FAKE ||
      app_ctx->config.tiled_display_config.rows != 2 ||
      app_ctx->config.tiled_display_config.columns != 3 ||
      app_ctx->config.tiled_display_config.width != MOSAIC_WIDTH ||
      app_ctx->config.tiled_display_config.height != MOSAIC_HEIGHT)
    return FALSE;
  /* DeepStream source_id is zero-based; camera-id in [sourceN] is the
   * stable, one-based logical camera number CAM-01 through CAM-06. */
  for (guint source_id = 0; source_id < MOSAIC_LABEL_COUNT; source_id++) {
    if (app_ctx->config.multi_source_config[source_id].camera_id != source_id + 1u)
      return FALSE;
  }

  GstPad *old_peer = NULL;
  preview->osd_src_pad = gst_element_get_static_pad (instance->osd_bin.bin, "src");
  preview->sink_bin_pad = gst_element_get_static_pad (instance->sink_bin.bin, "sink");
  preview->osd_sink_pad = gst_element_get_static_pad (instance->osd_bin.nvosd, "sink");
  if (!preview->osd_src_pad || !preview->sink_bin_pad || !preview->osd_sink_pad)
    return FALSE;
  old_peer = gst_pad_get_peer (preview->osd_src_pad);
  gboolean original_link = old_peer == preview->sink_bin_pad;
  if (old_peer)
    gst_object_unref (old_peer);
  if (!original_link)
    return FALSE;

  preview->tee = gst_element_factory_make ("tee", "mosaic-preview-tee");
  preview->queue = gst_element_factory_make ("queue", "mosaic-preview-queue");
  preview->converter = gst_element_factory_make ("nvvideoconvert", "mosaic-preview-convert");
  preview->caps_filter = gst_element_factory_make ("capsfilter", "mosaic-preview-system-bgrx");
  preview->app_sink = gst_element_factory_make ("appsink", "mosaic-preview-appsink");
  if (!preview->tee || !preview->queue || !preview->converter ||
      !preview->caps_filter || !preview->app_sink)
    return FALSE;

  gst_bin_add_many (GST_BIN (instance->bin), preview->tee, preview->queue,
      preview->converter, preview->caps_filter, preview->app_sink, NULL);
  g_object_set (preview->queue, "max-size-buffers", 1u, "max-size-bytes", 0u,
      "max-size-time", (guint64) 0, "leaky", 2, NULL);
  g_signal_connect (preview->queue, "overrun", G_CALLBACK (queue_overrun), preview);
  g_object_set (preview->app_sink, "max-buffers", 1u, "drop", TRUE,
      "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE,
      "wait-on-eos", FALSE, NULL);

  GstCaps *caps = gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, "BGRx",
      "width", G_TYPE_INT, (gint) MOSAIC_WIDTH,
      "height", G_TYPE_INT, (gint) MOSAIC_HEIGHT, NULL);
  gst_caps_set_features (caps, 0,
      gst_caps_features_new (GST_CAPS_FEATURE_MEMORY_SYSTEM_MEMORY, NULL));
  g_object_set (preview->caps_filter, "caps", caps, NULL);
  gst_caps_unref (caps);

  GstAppSinkCallbacks callbacks = { 0 };
  callbacks.new_sample = new_sample;
  gst_app_sink_set_callbacks (GST_APP_SINK (preview->app_sink), &callbacks, preview, NULL);

  if (!gst_pad_unlink (preview->osd_src_pad, preview->sink_bin_pad))
    return FALSE;
  if (!gst_element_link (instance->osd_bin.bin, preview->tee))
    return FALSE;

  preview->tee_main_pad = gst_element_request_pad_simple (preview->tee, "src_%u");
  preview->tee_preview_pad = gst_element_request_pad_simple (preview->tee, "src_%u");
  preview->queue_sink_pad = gst_element_get_static_pad (preview->queue, "sink");
  if (!preview->tee_main_pad || !preview->tee_preview_pad || !preview->queue_sink_pad ||
      gst_pad_link (preview->tee_main_pad, preview->sink_bin_pad) != GST_PAD_LINK_OK ||
      gst_pad_link (preview->tee_preview_pad, preview->queue_sink_pad) != GST_PAD_LINK_OK ||
      !gst_element_link_many (preview->queue, preview->converter,
          preview->caps_filter, preview->app_sink, NULL))
    return FALSE;

  preview->label_probe_id = gst_pad_add_probe (preview->osd_sink_pad,
      GST_PAD_PROBE_TYPE_BUFFER, camera_labels_probe, preview, NULL);
  return preview->label_probe_id != 0;
}

gboolean
mosaic_preview_attach (AppCtx *app_ctx)
{
  const gchar *enabled = g_getenv ("V11_UI_MOSAIC_PREVIEW_ENABLE");
  if (!enabled || g_strcmp0 (enabled, "1") != 0)
    return TRUE;
  if (active_preview || !app_ctx)
    return FALSE;

  MosaicPreview *preview = g_new0 (MosaicPreview, 1);
  preview->app_ctx = app_ctx;
  preview->fd = -1;
  preview->producer_lock_fd = -1;
  preview->fullscreen_fd = -1;
  preview->fullscreen_lock_fd = -1;
  preview->fullscreen_selected_source = -1;
  preview->fullscreen_pending_source = -1;
  preview->fullscreen_tiler_primed_source = -1;
  preview->fullscreen_prime_requested_source = -1;
  g_queue_init (&preview->fullscreen_identities);
  g_queue_init (&preview->fullscreen_batch_copies);
  g_mutex_init (&preview->mutex);
  active_preview = preview;
  if (!valid_rate (&preview->rate_fps) ||
      g_strcmp0 (g_getenv ("V11_UI_MOSAIC_PREVIEW_PATH"), IPC_PATH) != 0 ||
      !open_slot (preview)) {
    g_printerr ("MOSAIC_PREVIEW_ERROR code=CONFIG_OR_BRANCH_INIT_FAILED\n");
    mosaic_preview_stop ();
    return FALSE;
  }
  if (!add_display_branch (app_ctx, preview)) {
    g_printerr ("MOSAIC_PREVIEW_ERROR code=CONFIG_OR_BRANCH_INIT_FAILED\n");
    mosaic_preview_stop ();
    return FALSE;
  }
  const gchar *fullscreen_enabled = g_getenv ("V11_UI_FULLSCREEN_PREVIEW_ENABLE");
  if (fullscreen_enabled && g_strcmp0 (fullscreen_enabled, "1") == 0) {
    if (!add_fullscreen_branch (app_ctx, preview)) {
      g_printerr ("FULLSCREEN_PREVIEW_ERROR code=BRANCH_INIT_FAILED\n");
      mosaic_preview_stop ();
      return FALSE;
    }
    g_print ("FULLSCREEN_PREVIEW_READY {\"width\":1920,\"height\":1080,\"format\":\"BGRx\",\"ipc_version\":5,\"selection\":\"nvmultistreamtiler.show-source\"}\n");
  }
  preview->stats_timer_id = g_timeout_add_seconds (1, stats_tick, preview);
  g_print ("MOSAIC_PREVIEW_READY {\"width\":1920,\"height\":720,\"format\":\"BGRx\",\"rate_fps\":%u,\"path\":\"%s\",\"ipc_version\":4}\n",
      preview->rate_fps, IPC_PATH);
  return TRUE;
}

void
mosaic_preview_stop (void)
{
  MosaicPreview *preview = active_preview;
  if (!preview)
    return;
  g_print ("SIX_SHUTDOWN event=mosaic_preview_resources_release_started\n");
  active_preview = NULL;
  const gboolean fullscreen_was_enabled = preview->fullscreen_mapping != NULL &&
      preview->fullscreen_mapping != MAP_FAILED;
  if (preview->stats_timer_id)
    g_source_remove (preview->stats_timer_id);
  if (preview->fullscreen_request_timer_id)
    g_source_remove (preview->fullscreen_request_timer_id);
  if (preview->fullscreen_analytics_probe_id && preview->fullscreen_analytics_sink_pad)
    gst_pad_remove_probe (preview->fullscreen_analytics_sink_pad,
        preview->fullscreen_analytics_probe_id);
  if (preview->label_probe_id && preview->osd_sink_pad)
    gst_pad_remove_probe (preview->osd_sink_pad, preview->label_probe_id);
  if (preview->fullscreen_label_probe_id && preview->fullscreen_osd_bin.nvosd) {
    GstPad *pad = gst_element_get_static_pad (preview->fullscreen_osd_bin.nvosd, "sink");
    if (pad) {
      gst_pad_remove_probe (pad, preview->fullscreen_label_probe_id);
      gst_object_unref (pad);
    }
  }
  if (preview->fullscreen_osd_src_probe_id && preview->fullscreen_osd_src_pad)
    gst_pad_remove_probe (preview->fullscreen_osd_src_pad,
        preview->fullscreen_osd_src_probe_id);
  if (preview->fullscreen_input_probe_id && preview->fullscreen_valve_sink_pad)
    gst_pad_remove_probe (preview->fullscreen_valve_sink_pad,
        preview->fullscreen_input_probe_id);
  if (preview->fullscreen_tiler_input_probe_id && preview->fullscreen_tiler_sink_pad)
    gst_pad_remove_probe (preview->fullscreen_tiler_sink_pad,
        preview->fullscreen_tiler_input_probe_id);
  if (preview->tee && preview->tee_main_pad)
    gst_element_release_request_pad (preview->tee, preview->tee_main_pad);
  if (preview->tee && preview->tee_preview_pad)
    gst_element_release_request_pad (preview->tee, preview->tee_preview_pad);
  if (preview->app_ctx && preview->app_ctx->pipeline.common_elements.tee &&
      preview->fullscreen_tee_pad)
    gst_element_release_request_pad (
        preview->app_ctx->pipeline.common_elements.tee, preview->fullscreen_tee_pad);
  if (preview->tee_main_pad)
    gst_object_unref (preview->tee_main_pad);
  if (preview->tee_preview_pad)
    gst_object_unref (preview->tee_preview_pad);
  if (preview->queue_sink_pad)
    gst_object_unref (preview->queue_sink_pad);
  if (preview->fullscreen_tee_pad)
    gst_object_unref (preview->fullscreen_tee_pad);
  if (preview->fullscreen_queue_sink_pad)
    gst_object_unref (preview->fullscreen_queue_sink_pad);
  if (preview->fullscreen_valve_sink_pad)
    gst_object_unref (preview->fullscreen_valve_sink_pad);
  if (preview->fullscreen_tiler_sink_pad)
    gst_object_unref (preview->fullscreen_tiler_sink_pad);
  if (preview->fullscreen_analytics_sink_pad)
    gst_object_unref (preview->fullscreen_analytics_sink_pad);
  if (preview->fullscreen_osd_src_pad)
    gst_object_unref (preview->fullscreen_osd_src_pad);
  if (preview->sink_bin_pad)
    gst_object_unref (preview->sink_bin_pad);
  if (preview->osd_src_pad)
    gst_object_unref (preview->osd_src_pad);
  if (preview->osd_sink_pad)
    gst_object_unref (preview->osd_sink_pad);
  if (preview->mapping && preview->mapping != MAP_FAILED)
    munmap (preview->mapping, preview->mapping_size);
  if (preview->fd >= 0)
    close (preview->fd);
  if (preview->producer_lock_fd >= 0 && preview->owns_producer_lock) {
    (void) flock (preview->producer_lock_fd, LOCK_UN);
  }
  if (preview->producer_lock_fd >= 0)
    close (preview->producer_lock_fd);
  if (preview->fullscreen_mapping && preview->fullscreen_mapping != MAP_FAILED)
    munmap (preview->fullscreen_mapping, preview->fullscreen_mapping_size);
  if (preview->fullscreen_fd >= 0)
    close (preview->fullscreen_fd);
  if (preview->fullscreen_lock_fd >= 0 && preview->owns_fullscreen_lock)
    (void) flock (preview->fullscreen_lock_fd, LOCK_UN);
  if (preview->fullscreen_lock_fd >= 0)
    close (preview->fullscreen_lock_fd);
  g_print ("MOSAIC_PREVIEW_STOP {\"published\":%" G_GUINT64_FORMAT
      ",\"rate_drops\":%" G_GUINT64_FORMAT ",\"queue_overruns\":%" G_GUINT64_FORMAT
      ",\"ipc_lock_drops\":%" G_GUINT64_FORMAT ",\"ipc_failures\":%" G_GUINT64_FORMAT "}\n",
      preview->published, preview->rate_drops, preview->queue_overruns,
      preview->lock_drops, preview->ipc_failures);
  g_mutex_lock (&preview->mutex);
  clear_fullscreen_identities_locked (preview);
  clear_fullscreen_batch_copies_locked (preview);
  if (fullscreen_was_enabled)
    g_print ("FULLSCREEN_PREVIEW_STOP {\"published\":%" G_GUINT64_FORMAT
        ",\"selected_source_id\":%d,\"pending_source_id\":%d,"
      "\"deferred_batches\":%d,\"input_matches\":%" G_GUINT64_FORMAT
      ",\"tiler_inputs\":%" G_GUINT64_FORMAT
      ",\"tiler_outputs\":%" G_GUINT64_FORMAT
      ",\"osd_outputs\":%" G_GUINT64_FORMAT
      ",\"appsink_samples\":%" G_GUINT64_FORMAT
      ",\"correlation_failures\":%" G_GUINT64_FORMAT "}\n",
      preview->fullscreen_sequence,
      g_atomic_int_get (&preview->fullscreen_selected_source),
      g_atomic_int_get (&preview->fullscreen_pending_source),
      g_atomic_int_get (&preview->fullscreen_pending_misses),
      preview->fullscreen_input_matches, preview->fullscreen_tiler_inputs,
      preview->fullscreen_tiler_outputs,
      preview->fullscreen_osd_outputs, preview->fullscreen_appsink_samples,
      preview->fullscreen_correlation_failures);
  g_mutex_unlock (&preview->mutex);
  g_mutex_clear (&preview->mutex);
  g_free (preview);
  g_print ("SIX_SHUTDOWN event=mosaic_preview_resources_release_completed\n");
}
