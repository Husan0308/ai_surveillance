/* SPDX-License-Identifier: Apache-2.0
 *
 * Synthetic DeepStream 9.1 plugin integration test for Fix 5E.
 *
 * This test intentionally includes the isolated candidate implementation so
 * the pad probes, token copy/transform/release callbacks, ownership validator,
 * and accounting probe are the real code. It must only be run after explicit
 * operator authorization because nvstreammux, nvinfer, and NvDCF require GPU
 * access. The input frames are generated in memory; no camera or network is
 * used.
 */
#include "../../.runtime/deepstream-fix5-generation-20261031/sources/apps/sample_apps/deepstream-app/six_live_monitor.c"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>

#define SOURCE_COUNT 6
#define INPUT_WIDTH 960
#define INPUT_HEIGHT 544
#define RUN_LIMIT_SECONDS 60
#define STAGE_PRE_MUX 0
#define STAGE_MUX 1
#define STAGE_PGIE 2
#define STAGE_TRACKER 3

typedef enum {
  MUTATE_NONE = 0,
  MUTATE_MISSING,
  MUTATE_MALFORMED,
  MUTATE_DUPLICATE,
  MUTATE_WRONG_SOURCE
} Mutation;

typedef struct {
  guint64 accepted[4][SOURCE_COUNT][3];
  guint64 rejected[4][SOURCE_COUNT];
  guint64 rejected_reason[4][SOURCE_COUNT][SOURCE_OWNER_UNAVAILABLE + 1];
  guint64 stale[4][SOURCE_COUNT][3];
  guint64 mutation_seen[SOURCE_COUNT][5];
  gboolean fatal;
  gchar fatal_reason[256];
} Audit;

typedef struct {
  GstElement *pipeline;
  GstElement *appsrc[SOURCE_COUNT];
  GstPad *output_pad[SOURCE_COUNT];
  GstPad *mux_sink_pad[SOURCE_COUNT];
  GstElement *mux;
  GstElement *pgie;
  GstElement *tracker;
  GstElement *sink;
  GMutex lock;
  GCond condition;
  GstBuffer *held_source0;
  gint hold_source0;
  gint mutation[SOURCE_COUNT];
  Audit audit;
  gint64 deadline_us;
} PluginTest;

static PluginTest test_state;
static volatile sig_atomic_t stop_signal;

static void on_stop_signal (int signal_number)
{
  stop_signal = signal_number;
}

static void fail_test (const gchar *reason)
{
  g_mutex_lock (&test_state.lock);
  test_state.audit.fatal = TRUE;
  g_strlcpy (test_state.audit.fatal_reason, reason,
      sizeof test_state.audit.fatal_reason);
  g_cond_broadcast (&test_state.condition);
  g_mutex_unlock (&test_state.lock);
}

static NvDsMeta *find_owner_meta (GstBuffer *buffer, guint *count)
{
  gpointer iter = NULL;
  GstMeta *meta;
  NvDsMeta *found = NULL;
  *count = 0;
  while ((meta = gst_buffer_iterate_meta (buffer, &iter))) {
    if (meta->info->api != NVDS_META_API_TYPE)
      continue;
    NvDsMeta *candidate = (NvDsMeta *) meta;
    if (candidate->meta_type == source_owner_meta_type) {
      found = candidate;
      (*count)++;
    }
  }
  return found;
}

static GstPadProbeReturn mutate_owner_probe (GstPad *pad,
    GstPadProbeInfo *info, gpointer data)
{
  (void) pad;
  guint id = GPOINTER_TO_UINT (data);
  if (id >= SOURCE_COUNT)
    return GST_PAD_PROBE_OK;
  gint mode = g_atomic_int_get (&test_state.mutation[id]);
  if (mode == MUTATE_NONE ||
      !g_atomic_int_compare_and_exchange (&test_state.mutation[id], mode,
          MUTATE_NONE))
    return GST_PAD_PROBE_OK;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  guint count = 0;
  NvDsMeta *meta = buffer ? find_owner_meta (buffer, &count) : NULL;
  if (!meta || count != 1 || !meta->meta_data) {
    fail_test ("candidate source probe did not attach exactly one token before mutation");
    return GST_PAD_PROBE_OK;
  }
  SourceOwnerToken *token = meta->meta_data;
  switch ((Mutation) mode) {
    case MUTATE_MISSING:
      if (!gst_buffer_remove_meta (buffer, &meta->meta))
        fail_test ("could not remove token for missing-token negative case");
      break;
    case MUTATE_MALFORMED:
      token->magic = 0;
      break;
    case MUTATE_DUPLICATE: {
      SourceOwnerToken *copy = source_owner_copy (token, NULL);
      NvDsMeta *duplicate = gst_buffer_add_nvds_meta (buffer, copy, NULL,
          source_owner_copy, source_owner_release);
      if (!duplicate) {
        source_owner_release (copy, NULL);
        fail_test ("could not attach duplicate token for negative case");
      } else {
        duplicate->meta_type = source_owner_meta_type;
        duplicate->gst_to_nvds_meta_transform_func = source_owner_to_frame_meta;
        duplicate->gst_to_nvds_meta_release_func = source_owner_frame_meta_release;
      }
      break;
    }
    case MUTATE_WRONG_SOURCE:
      token->source_id = (id + 1) % SOURCE_COUNT;
      break;
    default:
      fail_test ("unknown token mutation requested");
      break;
  }
  g_mutex_lock (&test_state.lock);
  test_state.audit.mutation_seen[id][mode]++;
  g_cond_broadcast (&test_state.condition);
  g_mutex_unlock (&test_state.lock);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn hold_source0_probe (GstPad *pad,
    GstPadProbeInfo *info, gpointer data)
{
  (void) pad;
  (void) data;
  if (!g_atomic_int_compare_and_exchange (&test_state.hold_source0, TRUE, FALSE))
    return GST_PAD_PROBE_OK;
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  if (!buffer) {
    fail_test ("hold probe received no buffer");
    return GST_PAD_PROBE_DROP;
  }
  g_mutex_lock (&test_state.lock);
  if (test_state.held_source0) {
    test_state.audit.fatal = TRUE;
    g_strlcpy (test_state.audit.fatal_reason,
        "more than one source-0 frame was held", sizeof test_state.audit.fatal_reason);
  } else {
    test_state.held_source0 = gst_buffer_ref (buffer);
  }
  g_cond_broadcast (&test_state.condition);
  g_mutex_unlock (&test_state.lock);
  return GST_PAD_PROBE_DROP;
}

static gboolean raw_token (NvDsFrameMeta *frame, SourceOwnerToken *token)
{
  if (!frame)
    return FALSE;
  guint count = 0;
  NvDsUserMeta *found = NULL;
  for (NvDsMetaList *node = frame->frame_user_meta_list; node; node = node->next) {
    NvDsUserMeta *user = node->data;
    if (user && user->base_meta.meta_type == source_owner_meta_type) {
      found = user;
      count++;
    }
  }
  if (count != 1 || !found || !found->user_meta_data)
    return FALSE;
  memcpy (token, found->user_meta_data, sizeof *token);
  return TRUE;
}

static GstPadProbeReturn audit_probe (GstPad *pad, GstPadProbeInfo *info,
    gpointer data)
{
  (void) pad;
  guint stage = GPOINTER_TO_UINT (data);
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  NvDsBatchMeta *batch = buffer ? gst_buffer_get_nvds_batch_meta (buffer) : NULL;
  if (stage < STAGE_MUX || stage > STAGE_TRACKER || !batch) {
    fail_test ("DeepStream plugin stage did not expose NvDsBatchMeta");
    return GST_PAD_PROBE_OK;
  }
  for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
    NvDsFrameMeta *frame = node->data;
    if (!frame || frame->source_id >= SOURCE_COUNT ||
        frame->pad_index != frame->source_id) {
      fail_test ("mux source_id/pad_index mapping is invalid");
      continue;
    }
    SourceOwnerToken token = {0};
    gboolean token_present = raw_token (frame, &token);
    SourceOwnerResult result = source_frame_owner (frame, NULL);
    g_mutex_lock (&test_state.lock);
    if (result == SOURCE_OWNER_OK && token_present && token.generation < 3)
      test_state.audit.accepted[stage][frame->source_id][token.generation]++;
    else {
      test_state.audit.rejected[stage][frame->source_id]++;
      if (result <= SOURCE_OWNER_UNAVAILABLE)
        test_state.audit.rejected_reason[stage][frame->source_id][result]++;
      if (result == SOURCE_OWNER_STALE && token_present && token.generation < 3)
        test_state.audit.stale[stage][frame->source_id][token.generation]++;
    }
    g_cond_broadcast (&test_state.condition);
    g_mutex_unlock (&test_state.lock);
  }
  return GST_PAD_PROBE_OK;
}

/* Observe the exact buffer entering each nvstreammux request pad. This is the
 * last point where the test can check candidate GstMeta before NVIDIA
 * transforms it into NvDsFrameMeta. */
static GstPadProbeReturn pre_mux_audit_probe (GstPad *pad,
    GstPadProbeInfo *info, gpointer data)
{
  (void) pad;
  guint id = GPOINTER_TO_UINT (data);
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER (info);
  if (id >= SOURCE_COUNT || !buffer) {
    fail_test ("pre-mux probe received invalid source or buffer");
    return GST_PAD_PROBE_OK;
  }

  guint matches = 0;
  NvDsMeta *meta = find_owner_meta (buffer, &matches);
  SourceOwnerResult result = SOURCE_OWNER_OK;
  SourceOwnerToken *token = meta ? meta->meta_data : NULL;
  if (!matches)
    result = SOURCE_OWNER_MISSING;
  else if (matches != 1)
    result = SOURCE_OWNER_DUPLICATE;
  else if (!token || token->magic != SOURCE_OWNER_MAGIC ||
      token->version != SOURCE_OWNER_VERSION || token->source_id >= SOURCE_COUNT ||
      token->source_sequence == 0 || token->generation >= 3)
    result = SOURCE_OWNER_MALFORMED;
  else if (token->source_id != id)
    result = SOURCE_OWNER_WRONG_SOURCE;
  else {
    /* Match production lock order; a token is current only in an open epoch. */
    g_mutex_lock (&monitor_lock);
    g_mutex_lock (&telemetry_lock);
    gboolean current = source_epoch_ready[id] &&
        source_generation[id] == token->generation &&
        g_strcmp0 (source_lifecycle[id].state, "reconnecting") != 0 &&
        g_strcmp0 (source_lifecycle[id].state, "offline") != 0;
    g_mutex_unlock (&telemetry_lock);
    g_mutex_unlock (&monitor_lock);
    if (!current)
      result = SOURCE_OWNER_STALE;
  }

  g_mutex_lock (&test_state.lock);
  if (result == SOURCE_OWNER_OK)
    test_state.audit.accepted[STAGE_PRE_MUX][id][token->generation]++;
  else {
    test_state.audit.rejected[STAGE_PRE_MUX][id]++;
    if (result <= SOURCE_OWNER_UNAVAILABLE)
      test_state.audit.rejected_reason[STAGE_PRE_MUX][id][result]++;
    if (result == SOURCE_OWNER_STALE && token && token->generation < 3)
      test_state.audit.stale[STAGE_PRE_MUX][id][token->generation]++;
  }
  g_cond_broadcast (&test_state.condition);
  g_mutex_unlock (&test_state.lock);
  return GST_PAD_PROBE_OK;
}

static gboolean add_probe (GstElement *element, const gchar *pad_name,
    GstPadProbeCallback callback, gpointer data)
{
  GstPad *pad = gst_element_get_static_pad (element, pad_name);
  if (!pad)
    return FALSE;
  gulong id = gst_pad_add_probe (pad, GST_PAD_PROBE_TYPE_BUFFER, callback,
      data, NULL);
  gst_object_unref (pad);
  return id != 0;
}

static GstElement *make_element (const gchar *factory, const gchar *name)
{
  GstElement *element = gst_element_factory_make (factory, name);
  if (!element)
    g_printerr ("FIX5F_MISSING_PLUGIN factory=%s\n", factory);
  return element;
}

static gboolean build_pipeline (const gchar *infer_config)
{
  GstCaps *input_caps = gst_caps_from_string (
      "video/x-raw,format=(string)NV12,width=(int)960,height=(int)544,framerate=(fraction)10/1");
  GstCaps *output_caps = gst_caps_from_string (
      "video/x-raw(memory:NVMM),format=(string)NV12,width=(int)960,height=(int)544,framerate=(fraction)10/1");
  if (!input_caps || !output_caps)
    return FALSE;

  test_state.pipeline = gst_pipeline_new ("fix5f-six-source-synthetic");
  test_state.mux = make_element ("nvstreammux", "mux");
  test_state.pgie = make_element ("nvinfer", "pgie");
  test_state.tracker = make_element ("nvtracker", "tracker");
  test_state.sink = make_element ("fakesink", "sink");
  if (!test_state.pipeline || !test_state.mux || !test_state.pgie ||
      !test_state.tracker || !test_state.sink)
    goto fail;

  g_object_set (test_state.mux, "batch-size", SOURCE_COUNT,
      "width", INPUT_WIDTH, "height", INPUT_HEIGHT,
      "live-source", TRUE, "batched-push-timeout", 40000,
      "gpu-id", 0, NULL);
  g_object_set (test_state.pgie, "config-file-path", infer_config, NULL);
  g_object_set (test_state.tracker,
      "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
      "ll-config-file", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
      "tracker-width", INPUT_WIDTH, "tracker-height", INPUT_HEIGHT,
      "gpu-id", 0, NULL);
  g_object_set (test_state.sink, "sync", FALSE, "async", FALSE, NULL);
  gst_bin_add_many (GST_BIN (test_state.pipeline), test_state.mux,
      test_state.pgie, test_state.tracker, test_state.sink, NULL);
  if (!gst_element_link_many (test_state.mux, test_state.pgie,
          test_state.tracker, test_state.sink, NULL))
    goto fail;

  for (guint id = 0; id < SOURCE_COUNT; ++id) {
    gchar name[32];
    g_snprintf (name, sizeof name, "appsrc_%u", id);
    test_state.appsrc[id] = make_element ("appsrc", name);
    g_snprintf (name, sizeof name, "queue_%u", id);
    GstElement *queue = make_element ("queue", name);
    g_snprintf (name, sizeof name, "convert_%u", id);
    GstElement *convert = make_element ("nvvideoconvert", name);
    g_snprintf (name, sizeof name, "caps_%u", id);
    GstElement *filter = make_element ("capsfilter", name);
    if (!test_state.appsrc[id] || !queue || !convert || !filter)
      goto fail;
    g_object_set (test_state.appsrc[id], "caps", input_caps,
        "format", GST_FORMAT_TIME, "is-live", TRUE, "do-timestamp", FALSE,
        "block", FALSE, NULL);
    g_object_set (filter, "caps", output_caps, NULL);
    gst_bin_add_many (GST_BIN (test_state.pipeline), test_state.appsrc[id],
        queue, convert, filter, NULL);
    if (!gst_element_link_many (test_state.appsrc[id], queue, convert, filter, NULL))
      goto fail;
    GstPad *decoded_output = gst_element_get_static_pad (convert, "src");
    if (!decoded_output || !gst_pad_add_probe (decoded_output,
            GST_PAD_PROBE_TYPE_BUFFER, decode_probe, GUINT_TO_POINTER (id), NULL)) {
      if (decoded_output) gst_object_unref (decoded_output);
      goto fail;
    }
    gst_object_unref (decoded_output);
    GstPad *src = gst_element_get_static_pad (filter, "src");
    GstPad *sink = gst_element_request_pad_simple (test_state.mux, "sink_%u");
    if (!src || !sink || gst_pad_link (src, sink) != GST_PAD_LINK_OK) {
      if (src) gst_object_unref (src);
      if (sink) gst_object_unref (sink);
      goto fail;
    }
    test_state.output_pad[id] = src;
    test_state.mux_sink_pad[id] = sink;
    /* Same production callback, on the source-output boundary immediately
     * before its corresponding nvstreammux request pad. */
    if (!gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER,
            source_lineage_probe, GUINT_TO_POINTER (id), NULL) ||
        !gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER,
            mutate_owner_probe, GUINT_TO_POINTER (id), NULL) ||
        !gst_pad_add_probe (src, GST_PAD_PROBE_TYPE_BUFFER,
            hold_source0_probe, GUINT_TO_POINTER (id), NULL) ||
        !gst_pad_add_probe (sink, GST_PAD_PROBE_TYPE_BUFFER,
            pre_mux_audit_probe, GUINT_TO_POINTER (id), NULL))
      goto fail;
  }

  if (!add_probe (test_state.mux, "src", metadata_probe,
          GUINT_TO_POINTER (STAGE_MUX)) ||
      !add_probe (test_state.mux, "src", audit_probe,
          GUINT_TO_POINTER (STAGE_MUX)) ||
      !add_probe (test_state.pgie, "src", metadata_probe,
          GUINT_TO_POINTER (STAGE_PGIE)) ||
      !add_probe (test_state.pgie, "src", audit_probe,
          GUINT_TO_POINTER (STAGE_PGIE)) ||
      !add_probe (test_state.tracker, "src", metadata_probe,
          GUINT_TO_POINTER (STAGE_TRACKER)) ||
      !add_probe (test_state.tracker, "src", audit_probe,
          GUINT_TO_POINTER (STAGE_TRACKER)))
    goto fail;
  gst_caps_unref (input_caps);
  gst_caps_unref (output_caps);
  return TRUE;

fail:
  if (input_caps) gst_caps_unref (input_caps);
  if (output_caps) gst_caps_unref (output_caps);
  return FALSE;
}

static gboolean push_frame (guint id, guint64 frame_number)
{
  const gsize size = (gsize) INPUT_WIDTH * INPUT_HEIGHT * 3 / 2;
  GstBuffer *buffer = gst_buffer_new_allocate (NULL, size, NULL);
  if (!buffer)
    return FALSE;
  GstMapInfo map;
  if (!gst_buffer_map (buffer, &map, GST_MAP_WRITE)) {
    gst_buffer_unref (buffer);
    return FALSE;
  }
  memset (map.data, 16, (gsize) INPUT_WIDTH * INPUT_HEIGHT);
  memset (map.data + (gsize) INPUT_WIDTH * INPUT_HEIGHT, 128,
      size - (gsize) INPUT_WIDTH * INPUT_HEIGHT);
  gst_buffer_unmap (buffer, &map);
  GST_BUFFER_PTS (buffer) = frame_number * (GST_SECOND / 10);
  GST_BUFFER_DTS (buffer) = GST_CLOCK_TIME_NONE;
  GST_BUFFER_DURATION (buffer) = GST_SECOND / 10;
  GST_BUFFER_OFFSET (buffer) = frame_number;
  return gst_app_src_push_buffer (GST_APP_SRC (test_state.appsrc[id]), buffer) ==
      GST_FLOW_OK;
}

static guint64 read_stage_count (guint stage, guint id)
{
  g_mutex_lock (&monitor_lock);
  guint64 value = stage < G_N_ELEMENTS (stages) && id < SOURCE_COUNT ?
      stages[stage][id].count : 0;
  g_mutex_unlock (&monitor_lock);
  return value;
}

static guint64 read_generation (guint id)
{
  if (id >= SOURCE_COUNT)
    return G_MAXUINT64;
  /* Keep the fixture's observer consistent with the production lock order. */
  g_mutex_lock (&monitor_lock);
  g_mutex_lock (&telemetry_lock);
  guint64 value = source_generation[id];
  g_mutex_unlock (&telemetry_lock);
  g_mutex_unlock (&monitor_lock);
  return value;
}

static gboolean wait_for (gboolean (*predicate)(void *), void *arg,
    gint64 timeout_us)
{
  gint64 until = MIN (test_state.deadline_us,
      g_get_monotonic_time () + timeout_us);
  g_mutex_lock (&test_state.lock);
  while (!predicate (arg) && !test_state.audit.fatal && !stop_signal &&
      g_get_monotonic_time () < until) {
    gint64 wake = MIN (until, g_get_monotonic_time () + 100 * 1000);
    g_cond_wait_until (&test_state.condition, &test_state.lock, wake);
  }
  gboolean ok = predicate (arg) && !test_state.audit.fatal && !stop_signal;
  g_mutex_unlock (&test_state.lock);
  return ok;
}

typedef struct { guint stage, source; guint64 generation, minimum; gboolean rejected; } WaitCount;
static gboolean count_reached (void *data)
{
  WaitCount *w = data;
  if (w->rejected)
    return test_state.audit.rejected[w->stage][w->source] >= w->minimum;
  return test_state.audit.accepted[w->stage][w->source][w->generation] >= w->minimum;
}

static guint64 read_accepted_total (guint stage, guint source)
{
  if (stage > STAGE_TRACKER || source >= SOURCE_COUNT)
    return 0;
  g_mutex_lock (&test_state.lock);
  guint64 value = 0;
  for (guint generation = 0; generation < 3; ++generation)
    value += test_state.audit.accepted[stage][source][generation];
  g_mutex_unlock (&test_state.lock);
  return value;
}

static gboolean held_ready (void *unused)
{
  (void) unused;
  return test_state.held_source0 != NULL;
}

static gboolean mutation_seen (void *data)
{
  guint *ids = data;
  return test_state.audit.mutation_seen[ids[0]][ids[1]] > ids[2];
}

static gboolean release_held_source0 (void)
{
  g_mutex_lock (&test_state.lock);
  GstBuffer *buffer = test_state.held_source0;
  test_state.held_source0 = NULL;
  g_mutex_unlock (&test_state.lock);
  if (!buffer)
    return FALSE;
  GstFlowReturn flow = gst_pad_push (test_state.output_pad[0], buffer);
  return flow == GST_FLOW_OK;
}

static gboolean wait_bus_until (GstMessageType wanted, guint timeout_seconds)
{
  GstBus *bus = gst_element_get_bus (test_state.pipeline);
  gint64 stop = MIN (test_state.deadline_us,
      g_get_monotonic_time () + timeout_seconds * G_TIME_SPAN_SECOND);
  gboolean ok = FALSE;
  while (!stop_signal && g_get_monotonic_time () < stop) {
    GstMessage *message = gst_bus_timed_pop_filtered (bus, 50 * GST_MSECOND,
        wanted | GST_MESSAGE_ERROR);
    if (!message)
      continue;
    if (GST_MESSAGE_TYPE (message) == GST_MESSAGE_ERROR) {
      GError *error = NULL;
      gchar *debug = NULL;
      gst_message_parse_error (message, &error, &debug);
      g_printerr ("FIX5F_BUS_ERROR element=%s error=%s\n",
          GST_OBJECT_NAME (message->src), error ? error->message : "unknown");
      if (error) g_error_free (error);
      g_free (debug);
      gst_message_unref (message);
      break;
    }
    ok = TRUE;
    gst_message_unref (message);
    break;
  }
  gst_object_unref (bus);
  return ok;
}

static void print_results (void)
{
  g_mutex_lock (&test_state.lock);
  for (guint id = 0; id < SOURCE_COUNT; ++id) {
    for (guint stage = STAGE_PRE_MUX; stage <= STAGE_TRACKER; ++stage) {
      for (guint gen = 0; gen < 3; ++gen) {
        guint64 accepted = test_state.audit.accepted[stage][id][gen];
        guint64 stale = test_state.audit.stale[stage][id][gen];
        if (accepted || stale)
          g_print ("FIX5F_STAGE source_id=%u stage=%u generation=%u accepted=%" G_GUINT64_FORMAT " stale_rejected=%" G_GUINT64_FORMAT "\n",
              id, stage, gen, accepted, stale);
      }
      g_print ("FIX5F_REJECTED source_id=%u stage=%u total=%" G_GUINT64_FORMAT "\n",
          id, stage, test_state.audit.rejected[stage][id]);
      for (guint reason = SOURCE_OWNER_MISSING;
          reason <= SOURCE_OWNER_UNAVAILABLE; ++reason) {
        guint64 count = test_state.audit.rejected_reason[stage][id][reason];
        if (count)
          g_print ("FIX5F_REJECT_REASON source_id=%u stage=%u reason=%u count=%" G_GUINT64_FORMAT "\n",
              id, stage, reason, count);
      }
    }
  }
  g_mutex_unlock (&test_state.lock);
}

static gboolean expect_accept (guint id, guint gen)
{
  for (guint stage = STAGE_PRE_MUX; stage <= STAGE_TRACKER; ++stage) {
    WaitCount w = {stage, id, gen, 1, FALSE};
    if (!wait_for (count_reached, &w, 8 * G_TIME_SPAN_SECOND))
      return FALSE;
    if (read_stage_count (stage, id) == 0)
      return FALSE;
  }
  return TRUE;
}

static gboolean hold_old_generation_and_advance (guint64 from, guint64 to,
    guint64 frame_number)
{
  (void) from;
  g_atomic_int_set (&test_state.hold_source0, TRUE);
  if (!push_frame (0, frame_number) ||
      !wait_for (held_ready, NULL, 5 * G_TIME_SPAN_SECOND))
    return FALSE;
  six_live_source_reconnecting (0);
  six_live_source_generation_boundary (0);
  if (read_generation (0) != to || !release_held_source0 ())
    return FALSE;
  for (guint stage = STAGE_PRE_MUX; stage <= STAGE_TRACKER; ++stage) {
    gint64 until = MIN (test_state.deadline_us,
        g_get_monotonic_time () + 8 * G_TIME_SPAN_SECOND);
    g_mutex_lock (&test_state.lock);
    while (test_state.audit.stale[stage][0][from] == 0 &&
        !test_state.audit.fatal && !stop_signal &&
        g_get_monotonic_time () < until) {
      gint64 wake = MIN (until, g_get_monotonic_time () + 100 * 1000);
      g_cond_wait_until (&test_state.condition, &test_state.lock, wake);
    }
    gboolean rejected = test_state.audit.stale[stage][0][from] > 0;
    g_mutex_unlock (&test_state.lock);
    if (!rejected || read_stage_count (stage, 0) != 0)
      return FALSE;
  }
  return TRUE;
}

static void teardown (void)
{
  g_atomic_int_set (&ownership_stopping, 1);
  if (test_state.pipeline) {
    gst_element_set_state (test_state.pipeline, GST_STATE_NULL);
    GstState state = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
    gst_element_get_state (test_state.pipeline, &state, &pending,
        5 * GST_SECOND);
    if (state != GST_STATE_NULL || pending != GST_STATE_VOID_PENDING)
      g_printerr ("FIX5F_TEARDOWN_INCOMPLETE state=%s pending=%s\n",
          gst_element_state_get_name (state), gst_element_state_get_name (pending));
  }
  if (test_state.held_source0)
    gst_buffer_unref (test_state.held_source0);
  for (guint id = 0; id < SOURCE_COUNT; ++id) {
    if (test_state.output_pad[id])
      gst_object_unref (test_state.output_pad[id]);
    if (test_state.mux_sink_pad[id] && test_state.mux) {
      gst_element_release_request_pad (test_state.mux, test_state.mux_sink_pad[id]);
      gst_object_unref (test_state.mux_sink_pad[id]);
    }
  }
  if (test_state.pipeline)
    gst_object_unref (test_state.pipeline);
  g_cond_clear (&test_state.condition);
  g_mutex_clear (&test_state.lock);
  g_mutex_clear (&monitor_lock);
  g_mutex_clear (&telemetry_lock);
}

int main (int argc, char **argv)
{
  if (argc != 2) {
    g_printerr ("usage: fix5_plugin_pipeline_test /absolute/path/infer_config.txt\n");
    return 2;
  }
  memset (&test_state, 0, sizeof test_state);
  g_mutex_init (&test_state.lock);
  g_cond_init (&test_state.condition);
  test_state.deadline_us = g_get_monotonic_time () +
      RUN_LIMIT_SECONDS * G_TIME_SPAN_SECOND;
  gst_init (&argc, &argv);
  signal (SIGINT, on_stop_signal);
  signal (SIGTERM, on_stop_signal);
  g_mutex_init (&monitor_lock);
  g_mutex_init (&telemetry_lock);
  enabled = TRUE;
  telemetry_enabled = TRUE;
  g_atomic_int_set (&ownership_stopping, 0);
  gchar owner_name[] = SOURCE_OWNER_META_NAME;
  source_owner_meta_type = nvds_get_user_meta_type (owner_name);
  if (source_owner_meta_type < NVDS_START_USER_META) {
    g_printerr ("FIX5F_FAIL cannot allocate candidate token meta type\n");
    teardown ();
    return 1;
  }
  for (guint id = 0; id < SOURCE_COUNT; ++id) {
    source_generation[id] = 0;
    source_output_sequence[id] = 0;
    source_epoch_ready[id] = TRUE;
    source_lifecycle[id].state = "online";
    source_lifecycle[id].generation = 0;
    source_lifecycle[id].revision = 1;
  }
  g_unsetenv ("USE_NEW_NVSTREAMMUX");
  if (!build_pipeline (argv[1])) {
    g_printerr ("FIX5F_FAIL pipeline construction\n");
    teardown ();
    return 1;
  }
  live_pipeline = test_state.pipeline;
  if (gst_element_set_state (test_state.pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    g_printerr ("FIX5F_FAIL pipeline did not enter PLAYING\n");
    teardown ();
    return 1;
  }
  GstState current = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
  if (gst_element_get_state (test_state.pipeline, &current, &pending,
          5 * GST_SECOND) == GST_STATE_CHANGE_FAILURE || current != GST_STATE_PLAYING) {
    g_printerr ("FIX5F_FAIL PLAYING state not confirmed\n");
    teardown ();
    return 1;
  }
  g_print ("FIX5F_PLAYING monotonic_ns=%" G_GINT64_FORMAT " mux=legacy\n",
      g_get_monotonic_time () * 1000);

  gboolean ok = stop_signal == 0;
  for (guint id = 0; id < SOURCE_COUNT && ok; ++id) {
    ok = push_frame (id, 1) && expect_accept (id, 0);
  }
  ok = ok && hold_old_generation_and_advance (0, 1, 10);
  if (ok) {
    for (guint id = 1; id < SOURCE_COUNT && ok; ++id)
      ok = push_frame (id, 11);
    ok = ok && push_frame (0, 11) && expect_accept (0, 1);
  }
  ok = ok && hold_old_generation_and_advance (1, 2, 20);
  if (ok) {
    for (guint id = 1; id < SOURCE_COUNT && ok; ++id)
      ok = push_frame (id, 21);
    ok = ok && push_frame (0, 21) && expect_accept (0, 2);
  }

  const Mutation negative_cases[] = {MUTATE_MISSING, MUTATE_MALFORMED,
      MUTATE_DUPLICATE, MUTATE_WRONG_SOURCE};
  for (guint i = 0; i < G_N_ELEMENTS (negative_cases) && ok; ++i) {
    guint id = i + 1;
    guint64 before[4] = {0};
    guint64 progress_before[4] = {0};
    for (guint stage = STAGE_PRE_MUX; stage <= STAGE_TRACKER; ++stage)
      before[stage] = read_accepted_total (stage, id);
    for (guint stage = STAGE_MUX; stage <= STAGE_TRACKER; ++stage)
      progress_before[stage] = read_stage_count (stage, id);
    g_atomic_int_set (&test_state.mutation[id], negative_cases[i]);
    guint seen[] = {id, negative_cases[i], 0};
    ok = push_frame (id, 30 + i) && wait_for (mutation_seen, seen,
        5 * G_TIME_SPAN_SECOND);
    for (guint stage = STAGE_PRE_MUX; stage <= STAGE_TRACKER && ok; ++stage) {
      WaitCount w = {stage, id, 0, 1, TRUE};
      ok = wait_for (count_reached, &w, 8 * G_TIME_SPAN_SECOND);
      if (ok && read_accepted_total (stage, id) != before[stage])
        ok = FALSE;
      if (ok && stage >= STAGE_MUX &&
          read_stage_count (stage, id) != progress_before[stage])
        ok = FALSE;
    }
    if (ok)
      g_print ("FIX5F_NEGATIVE_REJECTED source_id=%u kind=%u all_stages=1 progress_unchanged=1\n",
          id, negative_cases[i]);
  }

  if (ok) {
    /* CAM-01's synthetic reconnect must not change the other five generations. */
    for (guint id = 1; id < SOURCE_COUNT; ++id)
      ok = ok && read_generation (id) == 0;
  }
  for (guint id = 0; id < SOURCE_COUNT; ++id)
    if (test_state.appsrc[id])
      gst_app_src_end_of_stream (GST_APP_SRC (test_state.appsrc[id]));
  gboolean eos = test_state.pipeline && wait_bus_until (GST_MESSAGE_EOS, 8);
  print_results ();
  gboolean null_ok = FALSE;
  if (test_state.pipeline) {
    gst_element_set_state (test_state.pipeline, GST_STATE_NULL);
    GstState state = GST_STATE_VOID_PENDING, transition = GST_STATE_VOID_PENDING;
    GstStateChangeReturn r = gst_element_get_state (test_state.pipeline,
        &state, &transition, 5 * GST_SECOND);
    null_ok = r == GST_STATE_CHANGE_SUCCESS && state == GST_STATE_NULL &&
        transition == GST_STATE_VOID_PENDING;
  }
  gboolean fatal = test_state.audit.fatal;
  if (!ok || !eos || !null_ok || fatal) {
    g_printerr ("FIX5F_FAIL assertions=%d eos=%d null=%d reason=%s\n",
        ok, eos, null_ok, test_state.audit.fatal_reason);
    teardown ();
    return 1;
  }
  g_print ("FIX5F_PASS valid_owner_survived_mux_pgie_nvdcf=1 generations=0,1,2 stale_rejected=1 invalid_tokens=4 eos=1 null=1\n");
  teardown ();
  return 0;
}
