/* In-pipeline, event-driven person crop delivery for the Dev Room identity path. */

#include "deepstream_crop_encoder.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "nvbufsurface.h"
#include "nvds_obj_encode.h"

#define CROP_SOCKET_PACKET_VERSION 1u
#define CROP_BIRTH_RETRIES 3u
#define CROP_BORDER_RETRIES 8u
#define CROP_POST_BORDER_RETRIES 3u
#define CROP_RETRY_INTERVAL_FRAMES 2u
#define CROP_TRACK_GAP_FRAMES 10u
#define CROP_SOCKET_BACKLOG 1

typedef struct {
  guint source_id;
  guint pad_index;
  guint64 object_id;
  guint64 generation;
  guint64 last_seen_frame;
  guint64 last_request_frame;
  guint requests;
  guint post_border_requests;
  gboolean birth_requests_all_border;
} CropTrackState;

typedef struct {
  guint source_id;
  guint pad_index;
  guint64 object_id;
  guint64 generation;
  guint frame_num;
  guint64 pts;
  guint64 ntp_timestamp;
  gdouble bbox[4];
  gdouble confidence;
  gchar camera_id[64];
  const gchar *reason;
} CropRequest;

typedef struct {
  NvDsFrameMeta *frame_meta;
  NvDsObjectMeta *object;
  CropRequest request;
  gint64 encode_start_ns;
} PendingCrop;

static NvDsObjEncCtxHandle encoder_ctx;
static GHashTable *track_states;
static gint crop_server_fd = -1;
static gint crop_client_fd = -1;
static gchar *crop_socket_path;
static FILE *crop_log;
static gboolean crop_enabled;

static gchar *
track_key (guint source_id, guint pad_index, guint64 object_id)
{
  return g_strdup_printf ("%u:%u:%" G_GUINT64_FORMAT,
      source_id, pad_index, object_id);
}

static void
crop_track_state_free (gpointer data)
{
  g_free (data);
}

static void
crop_log_line (const gchar *event, const CropRequest *request,
    gdouble encode_ms, gdouble delivery_ms, const gchar *detail)
{
  if (!crop_log || !request)
    return;
  fprintf (crop_log,
      "{\"event\":\"%s\",\"camera_id\":\"%s\","
      "\"source_id\":%u,\"pad_index\":%u,"
      "\"native_track_id\":\"%" G_GUINT64_FORMAT "\","
      "\"track_generation\":%" G_GUINT64_FORMAT ","
      "\"frame_num\":%u,\"pts\":\"%" G_GUINT64_FORMAT "\","
      "\"ntp_timestamp\":\"%" G_GUINT64_FORMAT "\","
      "\"bbox\":[%.3f,%.3f,%.3f,%.3f],\"reason\":\"%s\"",
      event, request->camera_id[0] ? request->camera_id : "UNMAPPED",
      request->source_id, request->pad_index, request->object_id,
      request->generation, request->frame_num, request->pts,
      request->ntp_timestamp, request->bbox[0], request->bbox[1],
      request->bbox[2], request->bbox[3], request->reason ? request->reason : "unknown");
  if (encode_ms >= 0.0)
    fprintf (crop_log, ",\"encode_latency_ms\":%.3f", encode_ms);
  if (delivery_ms >= 0.0)
    fprintf (crop_log, ",\"delivery_latency_ms\":%.3f", delivery_ms);
  if (detail)
    fprintf (crop_log, ",\"detail\":\"%s\"", detail);
  fputs ("}\n", crop_log);
  fflush (crop_log);
}

static gboolean
set_nonblocking (gint fd)
{
  gint flags = fcntl (fd, F_GETFL, 0);
  if (flags < 0 || fcntl (fd, F_SETFL, flags | O_NONBLOCK) < 0)
    return FALSE;
  return TRUE;
}

static void
crop_close_client (void)
{
  if (crop_client_fd >= 0) {
    close (crop_client_fd);
    crop_client_fd = -1;
  }
}

static gboolean
crop_socket_open (void)
{
  const gchar *path = g_getenv ("MV3DT_CROP_SOCKET");
  struct sockaddr_un address;
  gchar *directory;
  const gchar *log_dir;
  gchar *log_path;

  if (!path || !*path)
    return FALSE;
  if (strlen (path) >= sizeof (address.sun_path)) {
    g_printerr ("crop encoder: socket path is too long: %s\n", path);
    return FALSE;
  }

  crop_socket_path = g_strdup (path);
  directory = g_path_get_dirname (crop_socket_path);
  g_mkdir_with_parents (directory, 0755);
  g_free (directory);

  crop_server_fd = socket (AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (crop_server_fd < 0) {
    g_printerr ("crop encoder: socket() failed: %s\n", g_strerror (errno));
    return FALSE;
  }
  unlink (crop_socket_path);
  memset (&address, 0, sizeof (address));
  address.sun_family = AF_UNIX;
  g_strlcpy (address.sun_path, crop_socket_path, sizeof (address.sun_path));
  if (bind (crop_server_fd, (struct sockaddr *) &address, sizeof (address)) < 0 ||
      listen (crop_server_fd, CROP_SOCKET_BACKLOG) < 0 ||
      !set_nonblocking (crop_server_fd)) {
    g_printerr ("crop encoder: bind/listen failed for %s: %s\n",
        crop_socket_path, g_strerror (errno));
    close (crop_server_fd);
    crop_server_fd = -1;
    unlink (crop_socket_path);
    return FALSE;
  }
  /* The DeepStream process runs as root in the production container while the
   * host identity worker runs as the workspace user. Keep this IPC endpoint
   * local, but accessible to that worker. */
  if (chmod (crop_socket_path, 0666) < 0) {
    g_printerr ("crop encoder: chmod failed for %s: %s\n",
        crop_socket_path, g_strerror (errno));
    close (crop_server_fd);
    crop_server_fd = -1;
    unlink (crop_socket_path);
    return FALSE;
  }

  log_dir = g_getenv ("MV3DT_CROP_LOG_DIR");
  if (log_dir && *log_dir) {
    g_mkdir_with_parents (log_dir, 0755);
    log_path = g_build_filename (log_dir, "crop_encoder.jsonl", NULL);
    crop_log = fopen (log_path, "w");
    g_free (log_path);
    if (crop_log)
      setvbuf (crop_log, NULL, _IOLBF, 0);
  }
  g_print ("crop encoder: listening on %s\n", crop_socket_path);
  return TRUE;
}

static void
crop_accept_client (void)
{
  struct sockaddr_un address;
  socklen_t address_len = sizeof (address);
  gint fd;

  if (crop_server_fd < 0 || crop_client_fd >= 0)
    return;
  fd = accept (crop_server_fd, (struct sockaddr *) &address, &address_len);
  if (fd >= 0) {
    if (!set_nonblocking (fd)) {
      close (fd);
    } else {
      crop_client_fd = fd;
      g_print ("crop encoder: identity worker connected\n");
    }
  }
}

static gboolean
is_person (const NvDsObjectMeta *object)
{
  return object && (object->class_id == 0 ||
      !g_ascii_strcasecmp (object->obj_label, "person"));
}

static gboolean
crop_request_for_object (NvDsFrameMeta *frame_meta, NvDsObjectMeta *object,
    CropRequest *request)
{
  CropTrackState *state;
  gchar *key;
  gdouble right, bottom;
  gboolean border;
  gboolean due = FALSE;
  const gchar *reason = NULL;

  if (!is_person (object) || object->object_id == G_MAXUINT64)
    return FALSE;
  key = track_key (frame_meta->source_id, frame_meta->pad_index,
      object->object_id);
  state = g_hash_table_lookup (track_states, key);
  if (!state) {
    state = g_new0 (CropTrackState, 1);
    state->source_id = frame_meta->source_id;
    state->pad_index = frame_meta->pad_index;
    state->object_id = object->object_id;
    state->generation = 1;
    state->last_seen_frame = frame_meta->frame_num;
    state->last_request_frame = G_MAXUINT64;
    state->birth_requests_all_border = TRUE;
    g_hash_table_insert (track_states, key, state);
  } else {
    g_free (key);
    if (frame_meta->frame_num > state->last_seen_frame &&
        frame_meta->frame_num - state->last_seen_frame > CROP_TRACK_GAP_FRAMES) {
      state->generation++;
      state->requests = 0;
      state->post_border_requests = 0;
      state->birth_requests_all_border = TRUE;
      state->last_request_frame = G_MAXUINT64;
    }
    state->last_seen_frame = frame_meta->frame_num;
  }

  right = object->rect_params.left + object->rect_params.width;
  bottom = object->rect_params.top + object->rect_params.height;
  border = object->rect_params.left <= 2.0 || object->rect_params.top <= 2.0 ||
      right >= (gdouble) frame_meta->source_frame_width - 2.0 ||
      bottom >= (gdouble) frame_meta->source_frame_height - 2.0;

  if (state->requests == 0) {
    due = TRUE;
    reason = "new_native_track";
  } else if (frame_meta->frame_num >=
      state->last_request_frame + CROP_RETRY_INTERVAL_FRAMES) {
    if (state->requests < CROP_BIRTH_RETRIES) {
      due = TRUE;
      reason = "new_track_retry";
    } else if (border && state->requests < CROP_BORDER_RETRIES) {
      due = TRUE;
      reason = "border_quality_retry";
    } else if (!border && state->birth_requests_all_border &&
        state->post_border_requests < CROP_POST_BORDER_RETRIES) {
      /* Birth evidence was entirely border-truncated. Request a small,
       * bounded set of now-complete crops instead of leaving this visible
       * person PENDING forever; never encode stable tracks each frame. */
      due = TRUE;
      reason = "post_border_quality_retry";
    }
  }
  if (!due)
    return FALSE;

  memset (request, 0, sizeof (*request));
  request->source_id = frame_meta->source_id;
  request->pad_index = frame_meta->pad_index;
  request->object_id = object->object_id;
  request->generation = state->generation;
  request->frame_num = frame_meta->frame_num;
  request->pts = frame_meta->buf_pts;
  request->ntp_timestamp = frame_meta->ntp_timestamp;
  request->bbox[0] = object->rect_params.left;
  request->bbox[1] = object->rect_params.top;
  request->bbox[2] = right;
  request->bbox[3] = bottom;
  request->confidence = object->confidence;
  request->reason = reason ? reason : "event_driven_reid";
  if (state->requests < CROP_BIRTH_RETRIES && !border)
    state->birth_requests_all_border = FALSE;
  if (!g_strcmp0 (reason, "post_border_quality_retry"))
    state->post_border_requests++;
  state->requests++;
  state->last_request_frame = frame_meta->frame_num;
  return TRUE;
}

static gboolean
crop_send_packet (const CropRequest *request, const gchar *header,
    gsize header_len, const NvDsObjEncOutParams *encoded,
    gint64 encoded_monotonic_ns)
{
  guint32 header_size = (guint32) header_len;
  guint64 image_size = encoded->outLen;
  gsize packet_size = sizeof (header_size) + sizeof (image_size) +
      header_len + encoded->outLen;
  guint8 *packet;
  guint8 *cursor;
  ssize_t sent;
  gint64 delivery_ns;

  if (crop_client_fd < 0)
    return FALSE;
  packet = g_malloc (packet_size);
  cursor = packet;
  memcpy (cursor, &header_size, sizeof (header_size));
  cursor += sizeof (header_size);
  memcpy (cursor, &image_size, sizeof (image_size));
  cursor += sizeof (image_size);
  memcpy (cursor, header, header_len);
  cursor += header_len;
  memcpy (cursor, encoded->outBuffer, encoded->outLen);
  sent = send (crop_client_fd, packet, packet_size, MSG_DONTWAIT);
  g_free (packet);
  if (sent != (ssize_t) packet_size) {
    crop_close_client ();
    return FALSE;
  }
  delivery_ns = g_get_monotonic_time () * 1000 - encoded_monotonic_ns;
  crop_log_line ("crop_delivered", request, -1.0,
      (gdouble) delivery_ns / 1000000.0, NULL);
  return TRUE;
}

static void
crop_camera_id (guint source_id, guint pad_index, gchar *mapped, gsize mapped_size)
{
  const gchar *mapping = g_getenv ("MV3DT_CROP_CAMERA_MAP");
  gchar pair_key[32];
  gchar source_key[32];
  gchar **entries;
  guint i;

  g_snprintf (pair_key, sizeof (pair_key), "%u/%u", source_id, pad_index);
  g_snprintf (source_key, sizeof (source_key), "source:%u", source_id);
  g_strlcpy (mapped, "UNMAPPED", mapped_size);
  if (!mapping || !*mapping)
    return;
  entries = g_strsplit_set (mapping, ";,", -1);
  for (i = 0; entries && entries[i]; i++) {
    gchar **parts = g_strsplit (entries[i], "=", 2);
    if (parts[0] && parts[1] &&
        (!g_strcmp0 (parts[0], pair_key) || !g_strcmp0 (parts[0], source_key))) {
      g_strlcpy (mapped, parts[1], mapped_size);
      g_strfreev (parts);
      break;
    }
    g_strfreev (parts);
  }
  g_strfreev (entries);
}

static void
crop_send_encoded (AppCtx *app_ctx, NvDsFrameMeta *frame_meta,
    NvDsObjectMeta *object, const CropRequest *request,
    const NvDsObjEncOutParams *encoded, gint64 encode_start_ns,
    gint64 encode_end_ns)
{
  gdouble scale_w = 1.0, scale_h = 1.0;
  gchar camera_id[64];
  gchar *header;

  crop_camera_id (request->source_id, request->pad_index, camera_id, sizeof (camera_id));
  gboolean delivered;

  if (app_ctx->config.streammux_config.pipeline_width > 0 &&
      frame_meta->source_frame_width > 0)
    scale_w = (gdouble) frame_meta->source_frame_width /
        app_ctx->config.streammux_config.pipeline_width;
  if (app_ctx->config.streammux_config.pipeline_height > 0 &&
      frame_meta->source_frame_height > 0)
    scale_h = (gdouble) frame_meta->source_frame_height /
        app_ctx->config.streammux_config.pipeline_height;

  header = g_strdup_printf (
      "{\"version\":%u,\"camera_id\":\"%s\",\"source_id\":%u,"
      "\"pad_index\":%u,\"batch_id\":%u,"
      "\"native_track_id\":\"%" G_GUINT64_FORMAT "\","
      "\"frame_num\":%u,\"pts\":\"%" G_GUINT64_FORMAT "\","
      "\"ntp_timestamp\":\"%" G_GUINT64_FORMAT "\","
      "\"source_frame_width\":%u,\"source_frame_height\":%u,"
      "\"bbox\":[%.6f,%.6f,%.6f,%.6f],\"confidence\":%.6f,"
      "\"track_generation\":%" G_GUINT64_FORMAT ","
      "\"canonical_identity\":null,\"reason\":\"%s\","
      "\"encoded_monotonic_ns\":%" G_GINT64_FORMAT ","
      "\"jpeg_bytes\":%" G_GUINT64_FORMAT "}",
      CROP_SOCKET_PACKET_VERSION, camera_id, request->source_id,
      request->pad_index, frame_meta->batch_id, request->object_id,
      request->frame_num, request->pts, request->ntp_timestamp,
      frame_meta->source_frame_width, frame_meta->source_frame_height,
      request->bbox[0] * scale_w, request->bbox[1] * scale_h,
      request->bbox[2] * scale_w, request->bbox[3] * scale_h,
      request->confidence, request->generation, request->reason,
      encode_end_ns, encoded->outLen);
  delivered = crop_send_packet (request, header, strlen (header), encoded,
      encode_end_ns);
  crop_log_line (delivered ? "crop_success" : "crop_delivery_failure",
      request, (gdouble) (encode_end_ns - encode_start_ns) / 1000000.0,
      -1.0, delivered ? NULL : "identity_worker_not_connected_or_backpressure");
  g_free (header);
  (void) object;
}

gboolean
mv3dt_crop_encoder_init (gint gpu_id)
{
  if (!g_getenv ("MV3DT_CROP_SOCKET"))
    return TRUE;
  track_states = g_hash_table_new_full (g_str_hash, g_str_equal, g_free,
      crop_track_state_free);
  crop_enabled = crop_socket_open ();
  if (!crop_enabled)
    return FALSE;
  encoder_ctx = nvds_obj_enc_create_context (gpu_id >= 0 ? gpu_id : 0);
  if (!encoder_ctx) {
    g_printerr ("crop encoder: nvds_obj_enc_create_context failed\n");
    return FALSE;
  }
  return TRUE;
}

void
mv3dt_crop_encoder_process (AppCtx *app_ctx, GstBuffer *buffer,
    NvDsBatchMeta *batch_meta)
{
  GstMapInfo map = GST_MAP_INFO_INIT;
  NvBufSurface *surface;
  NvDsMetaList *frame_item;
  GPtrArray *pending;

  if (!crop_enabled || !encoder_ctx || !app_ctx || !buffer || !batch_meta)
    return;
  crop_accept_client ();
  if (!gst_buffer_map (buffer, &map, GST_MAP_READ))
    return;
  surface = (NvBufSurface *) map.data;
  pending = g_ptr_array_new_with_free_func (g_free);

  for (frame_item = batch_meta->frame_meta_list; frame_item;
      frame_item = frame_item->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) frame_item->data;
    NvDsMetaList *object_item;
    if (!frame_meta)
      continue;
    for (object_item = frame_meta->obj_meta_list; object_item;
        object_item = object_item->next) {
      NvDsObjectMeta *object = (NvDsObjectMeta *) object_item->data;
      CropRequest request;
      NvDsObjEncUsrArgs args = { 0 };
      PendingCrop *item;
      if (!crop_request_for_object (frame_meta, object, &request))
        continue;
      crop_camera_id (request.source_id, request.pad_index, request.camera_id,
          sizeof (request.camera_id));
      args.saveImg = FALSE;
      args.attachUsrMeta = TRUE;
      args.scaleImg = FALSE;
      args.objNum = 1;
      args.quality = 85;
      args.isFrame = FALSE;
      args.calcEncodeTime = FALSE;
      item = g_new0 (PendingCrop, 1);
      item->frame_meta = frame_meta;
      item->object = object;
      item->request = request;
      item->encode_start_ns = g_get_monotonic_time () * 1000;
      g_ptr_array_add (pending, item);
      crop_log_line ("crop_request", &request, -1.0, -1.0, NULL);
      if (!nvds_obj_enc_process (encoder_ctx, &args, surface, object,
              frame_meta)) {
        crop_log_line ("crop_encode_failure", &request, -1.0, -1.0,
            "nvds_obj_enc_process_failed");
      }
    }
  }
  nvds_obj_enc_finish (encoder_ctx);

  for (guint i = 0; i < pending->len; i++) {
    PendingCrop *item = g_ptr_array_index (pending, i);
    NvDsMetaList *user_item;
    gboolean found = FALSE;
    for (user_item = item->object->obj_user_meta_list; user_item;
        user_item = user_item->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) user_item->data;
      NvDsObjEncOutParams *encoded;
      gint64 encode_end_ns;
      if (!user_meta ||
          user_meta->base_meta.meta_type != NVDS_CROP_IMAGE_META ||
          !user_meta->user_meta_data)
        continue;
      encoded = (NvDsObjEncOutParams *) user_meta->user_meta_data;
      if (!encoded->outBuffer || !encoded->outLen)
        continue;
      encode_end_ns = g_get_monotonic_time () * 1000;
      crop_send_encoded (app_ctx, item->frame_meta, item->object,
          &item->request, encoded, item->encode_start_ns, encode_end_ns);
      found = TRUE;
      break;
    }
    if (!found)
      crop_log_line ("crop_encode_failure", &item->request, -1.0, -1.0,
          "encoder_did_not_attach_metadata");
  }
  g_ptr_array_free (pending, TRUE);
  gst_buffer_unmap (buffer, &map);
}

void
mv3dt_crop_encoder_shutdown (void)
{
  if (encoder_ctx) {
    nvds_obj_enc_finish (encoder_ctx);
    nvds_obj_enc_destroy_context (encoder_ctx);
    encoder_ctx = NULL;
  }
  crop_close_client ();
  if (crop_server_fd >= 0) {
    close (crop_server_fd);
    crop_server_fd = -1;
  }
  if (crop_socket_path) {
    unlink (crop_socket_path);
    g_clear_pointer (&crop_socket_path, g_free);
  }
  if (crop_log) {
    fclose (crop_log);
    crop_log = NULL;
  }
  if (track_states) {
    g_hash_table_destroy (track_states);
    track_states = NULL;
  }
  crop_enabled = FALSE;
}
