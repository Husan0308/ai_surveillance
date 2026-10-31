/* Opt-in, payload-free upstream progress probes for the isolated Fix 6 build. */
#include "six_live_upstream_progress.h"
#include <string.h>

typedef struct {
  guint64 buffers;
  guint64 last_progress_ns;
  guint64 generation;
  gint attached;
} UpstreamCounter;

typedef struct {
  GstPad *pad;
  gulong probe_id;
} ProbeSlot;

typedef struct {
  guint source_id;
  SixUpstreamStage stage;
} ProbeContext;

static UpstreamCounter counters[6][2];
static guint64 source_generation[6];
static gint source_active[6];
static ProbeSlot slots[6][2];
static GMutex slots_lock;
static gsize initialized;
static gboolean diagnostics_enabled;

static void ensure_initialized (void)
{
  if (g_once_init_enter (&initialized)) {
    g_mutex_init (&slots_lock);
    diagnostics_enabled = g_strcmp0 (
        g_getenv ("SIX_LIVE_UPSTREAM_DIAGNOSTICS"), "1") == 0;
    for (guint source = 0; source < 6; source++) {
      __atomic_store_n (&source_generation[source], 0, __ATOMIC_RELAXED);
      __atomic_store_n (&source_active[source], diagnostics_enabled ? 1 : 0,
          __ATOMIC_RELAXED);
      for (guint stage = 0; stage < 2; stage++) {
        __atomic_store_n (&counters[source][stage].buffers, 0, __ATOMIC_RELAXED);
        __atomic_store_n (&counters[source][stage].last_progress_ns, 0,
            __ATOMIC_RELAXED);
        __atomic_store_n (&counters[source][stage].generation, 0,
            __ATOMIC_RELAXED);
        __atomic_store_n (&counters[source][stage].attached, 0,
            __ATOMIC_RELAXED);
      }
    }
    g_once_init_leave (&initialized, 1);
  }
}

static gboolean stage_valid (SixUpstreamStage stage)
{
  return stage == SIX_UPSTREAM_RTP_PAD ||
      stage == SIX_UPSTREAM_COMPRESSED_PRE_DECODE;
}

void six_upstream_progress_init (gboolean enabled)
{
  ensure_initialized ();
  g_mutex_lock (&slots_lock);
  gboolean has_attached_probe = FALSE;
  for (guint source = 0; source < 6; source++)
    for (guint stage = 0; stage < 2; stage++)
      has_attached_probe |= slots[source][stage].pad != NULL;
  if (has_attached_probe) {
    /* The source-bin queue exists before the app's authenticated prepare
     * hook. Preserve probes installed from the explicit environment opt-in. */
    g_mutex_unlock (&slots_lock);
    return;
  }
  diagnostics_enabled = enabled;
  for (guint source = 0; source < 6; source++) {
    __atomic_store_n (&source_generation[source], 0, __ATOMIC_RELEASE);
    __atomic_store_n (&source_active[source], enabled ? 1 : 0, __ATOMIC_RELEASE);
    for (guint stage = 0; stage < 2; stage++) {
      __atomic_store_n (&counters[source][stage].buffers, 0, __ATOMIC_RELAXED);
      __atomic_store_n (&counters[source][stage].last_progress_ns, 0,
          __ATOMIC_RELAXED);
      __atomic_store_n (&counters[source][stage].generation, 0,
          __ATOMIC_RELEASE);
      __atomic_store_n (&counters[source][stage].attached, 0, __ATOMIC_RELEASE);
    }
  }
  g_mutex_unlock (&slots_lock);
}

static GstPadProbeReturn upstream_progress_probe (GstPad *pad,
    GstPadProbeInfo *info, gpointer user_data)
{
  (void) pad;
  const ProbeContext *context = user_data;
  if (!diagnostics_enabled || !context || context->source_id >= 6 ||
      !stage_valid (context->stage))
    return GST_PAD_PROBE_OK;

  guint64 count = 0;
  GstPadProbeType type = GST_PAD_PROBE_INFO_TYPE (info);
  if (type & GST_PAD_PROBE_TYPE_BUFFER) {
    if (GST_PAD_PROBE_INFO_BUFFER (info))
      count = 1;
  } else if (type & GST_PAD_PROBE_TYPE_BUFFER_LIST) {
    GstBufferList *list = GST_PAD_PROBE_INFO_BUFFER_LIST (info);
    if (list)
      count = gst_buffer_list_length (list);
  }
  if (!count)
    return GST_PAD_PROBE_OK;

  guint source = context->source_id;
  guint stage = (guint) context->stage;
  UpstreamCounter *counter = &counters[source][stage];
  if (!__atomic_load_n (&source_active[source], __ATOMIC_ACQUIRE) ||
      !__atomic_load_n (&counter->attached, __ATOMIC_ACQUIRE))
    return GST_PAD_PROBE_OK;
  guint64 generation = __atomic_load_n (&source_generation[source],
      __ATOMIC_ACQUIRE);
  if (generation != __atomic_load_n (&counter->generation, __ATOMIC_ACQUIRE))
    return GST_PAD_PROBE_OK;

  /* Probes do not take locks, map payloads, or perform I/O. The source-bin
   * NULL transition is the existing quiescence boundary before generation
   * reset, so these relaxed increments cannot cross into a later epoch. */
  __atomic_fetch_add (&counter->buffers, count, __ATOMIC_RELAXED);
  guint64 now_ns = (guint64) g_get_monotonic_time () * 1000;
  guint64 previous = __atomic_load_n (&counter->last_progress_ns,
      __ATOMIC_RELAXED);
  while (previous < now_ns &&
      !__atomic_compare_exchange_n (&counter->last_progress_ns, &previous,
          now_ns, FALSE, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
    ;
  return GST_PAD_PROBE_OK;
}

static void detach_slot_locked (guint source, guint stage)
{
  ProbeSlot *slot = &slots[source][stage];
  if (!slot->pad)
    return;
  __atomic_store_n (&counters[source][stage].attached, 0, __ATOMIC_RELEASE);
  if (slot->probe_id)
    gst_pad_remove_probe (slot->pad, slot->probe_id);
  gst_object_unref (slot->pad);
  slot->pad = NULL;
  slot->probe_id = 0;
}

gboolean six_upstream_progress_attach (GstPad *pad, guint source_id,
    SixUpstreamStage stage)
{
  ensure_initialized ();
  if (!pad || source_id >= 6 || !stage_valid (stage))
    return FALSE;
  if (!diagnostics_enabled)
    return TRUE; /* Default-off: intentional no-op, not an attach failure. */
  guint stage_id = (guint) stage;
  g_mutex_lock (&slots_lock);
  ProbeSlot *slot = &slots[source_id][stage_id];
  if (slot->pad == pad && slot->probe_id) {
    g_mutex_unlock (&slots_lock);
    return TRUE;
  }
  detach_slot_locked (source_id, stage_id);

  ProbeContext *context = g_new0 (ProbeContext, 1);
  context->source_id = source_id;
  context->stage = stage;
  gulong probe_id = gst_pad_add_probe (pad,
      (GstPadProbeType) (GST_PAD_PROBE_TYPE_BUFFER |
          GST_PAD_PROBE_TYPE_BUFFER_LIST), upstream_progress_probe, context,
      g_free);
  if (!probe_id) {
    g_free (context);
    g_mutex_unlock (&slots_lock);
    return FALSE;
  }
  slot->pad = GST_PAD (gst_object_ref (pad));
  slot->probe_id = probe_id;
  __atomic_store_n (&counters[source_id][stage_id].attached, 1,
      __ATOMIC_RELEASE);
  g_mutex_unlock (&slots_lock);
  return TRUE;
}

void six_upstream_progress_detach (GstPad *pad, guint source_id,
    SixUpstreamStage stage)
{
  ensure_initialized ();
  if (!pad || source_id >= 6 || !stage_valid (stage))
    return;
  guint stage_id = (guint) stage;
  g_mutex_lock (&slots_lock);
  if (slots[source_id][stage_id].pad == pad)
    detach_slot_locked (source_id, stage_id);
  g_mutex_unlock (&slots_lock);
}

void six_upstream_progress_generation_begin (guint source_id)
{
  ensure_initialized ();
  if (!diagnostics_enabled || source_id >= 6)
    return;
  __atomic_store_n (&source_active[source_id], 0, __ATOMIC_RELEASE);
}

void six_upstream_progress_generation_boundary (guint source_id,
    guint64 generation)
{
  ensure_initialized ();
  if (!diagnostics_enabled || source_id >= 6)
    return;
  __atomic_store_n (&source_active[source_id], 0, __ATOMIC_RELEASE);
  for (guint stage = 0; stage < 2; stage++) {
    __atomic_store_n (&counters[source_id][stage].buffers, 0, __ATOMIC_RELAXED);
    __atomic_store_n (&counters[source_id][stage].last_progress_ns, 0,
        __ATOMIC_RELAXED);
    __atomic_store_n (&counters[source_id][stage].generation, generation,
        __ATOMIC_RELEASE);
  }
  __atomic_store_n (&source_generation[source_id], generation, __ATOMIC_RELEASE);
  __atomic_store_n (&source_active[source_id], 1, __ATOMIC_RELEASE);
}

gboolean six_upstream_progress_snapshot (guint source_id,
    guint64 expected_generation, SixUpstreamStage stage,
    SixUpstreamSnapshot *snapshot)
{
  ensure_initialized ();
  if (!snapshot || source_id >= 6 || !stage_valid (stage))
    return FALSE;
  memset (snapshot, 0, sizeof *snapshot);
  if (!diagnostics_enabled)
    return FALSE;
  guint stage_id = (guint) stage;
  UpstreamCounter *counter = &counters[source_id][stage_id];
  gboolean active_before = __atomic_load_n (&source_active[source_id],
      __ATOMIC_ACQUIRE) != 0;
  guint64 generation_before = __atomic_load_n (&source_generation[source_id],
      __ATOMIC_ACQUIRE);
  guint64 counter_generation = __atomic_load_n (&counter->generation,
      __ATOMIC_ACQUIRE);
  guint64 buffers = __atomic_load_n (&counter->buffers, __ATOMIC_RELAXED);
  guint64 last_ns = __atomic_load_n (&counter->last_progress_ns,
      __ATOMIC_ACQUIRE);
  gboolean attached = __atomic_load_n (&counter->attached, __ATOMIC_ACQUIRE) != 0;
  guint64 generation_after = __atomic_load_n (&source_generation[source_id],
      __ATOMIC_ACQUIRE);
  gboolean active_after = __atomic_load_n (&source_active[source_id],
      __ATOMIC_ACQUIRE) != 0;

  snapshot->probe_attached = attached;
  snapshot->source_generation = generation_after;
  snapshot->valid = attached && active_before && active_after &&
      generation_before == generation_after && counter_generation == generation_after &&
      generation_after == expected_generation;
  if (snapshot->valid) {
    snapshot->buffers = buffers;
    snapshot->last_progress_monotonic_ns = last_ns;
  }
  return TRUE;
}

void six_upstream_progress_shutdown (void)
{
  ensure_initialized ();
  if (!diagnostics_enabled)
    return;
  for (guint source = 0; source < 6; source++)
    __atomic_store_n (&source_active[source], 0, __ATOMIC_RELEASE);
  g_mutex_lock (&slots_lock);
  for (guint source = 0; source < 6; source++)
    for (guint stage = 0; stage < 2; stage++)
      detach_slot_locked (source, stage);
  g_mutex_unlock (&slots_lock);
}
