#include "six_live_upstream_progress.h"
#include <stdio.h>
#include <string.h>

typedef struct {
  GstPad *pad;
  gint *start;
} PushArgs;

static GstFlowReturn consume_buffer (GstPad *pad, GstObject *parent,
    GstBuffer *buffer)
{
  (void) pad; (void) parent;
  gst_buffer_unref (buffer);
  return GST_FLOW_OK;
}

static GstFlowReturn consume_list (GstPad *pad, GstObject *parent,
    GstBufferList *list)
{
  (void) pad; (void) parent;
  gst_buffer_list_unref (list);
  return GST_FLOW_OK;
}

static GstPad *make_pad (const gchar *name)
{
  GstPad *pad = gst_pad_new (name, GST_PAD_SINK);
  gst_pad_set_chain_function (pad, GST_DEBUG_FUNCPTR (consume_buffer));
  gst_pad_set_chain_list_function (pad, GST_DEBUG_FUNCPTR (consume_list));
  if (!gst_pad_set_active (pad, TRUE)) {
    gst_object_unref (pad);
    return NULL;
  }
  return pad;
}

static gboolean push_buffer (GstPad *pad)
{
  GstFlowReturn flow = gst_pad_chain (pad, gst_buffer_new ());
  return flow == GST_FLOW_OK;
}

static gboolean push_list (GstPad *pad, guint size)
{
  GstBufferList *list = gst_buffer_list_new_sized (size);
  for (guint i = 0; i < size; i++)
    gst_buffer_list_add (list, gst_buffer_new ());
  return gst_pad_chain_list (pad, list) == GST_FLOW_OK;
}

static gboolean snapshot (guint source, guint64 generation,
    SixUpstreamStage stage, SixUpstreamSnapshot *out)
{
  return six_upstream_progress_snapshot (source, generation, stage, out);
}

static gpointer concurrent_push (gpointer value)
{
  PushArgs *args = value;
  while (!g_atomic_int_get (args->start))
    g_thread_yield ();
  for (guint i = 0; i < 5000; i++)
    if (!push_buffer (args->pad))
      break;
  return NULL;
}

#define CHECK(expr, message) do { \
  if (!(expr)) { fprintf (stderr, "FAIL: %s\n", message); return 1; } \
  passed++; \
} while (0)

int main (int argc, char **argv)
{
  gst_init (&argc, &argv);
  guint passed = 0;

  six_upstream_progress_init (FALSE);
  GstPad *disabled_pad = make_pad ("disabled");
  CHECK (disabled_pad != NULL, "disabled fixture pad created");
  CHECK (six_upstream_progress_attach (disabled_pad, 0, SIX_UPSTREAM_RTP_PAD),
      "diagnostics disabled is a safe no-op, not a pipeline error");
  SixUpstreamSnapshot sample;
  CHECK (!snapshot (0, 0, SIX_UPSTREAM_RTP_PAD, &sample),
      "disabled snapshot is unavailable, not zero progress");
  gst_pad_set_active (disabled_pad, FALSE);
  gst_object_unref (disabled_pad);

  six_upstream_progress_init (TRUE);
  GstPad *pads[6][2] = {{0}};
  for (guint source = 0; source < 6; source++) {
    for (guint stage = 0; stage < 2; stage++) {
      gchar name[32];
      g_snprintf (name, sizeof name, "source%u-stage%u", source, stage);
      pads[source][stage] = make_pad (name);
      CHECK (pads[source][stage] != NULL, "enabled fixture pad created");
      CHECK (six_upstream_progress_attach (pads[source][stage], source,
          (SixUpstreamStage) stage), "stage probe attached");
      CHECK (snapshot (source, 0, (SixUpstreamStage) stage, &sample) &&
          sample.probe_attached && sample.valid && sample.buffers == 0 &&
          sample.last_progress_monotonic_ns == 0,
          "attached no-progress sample is valid zero, distinct from missing");
    }
  }

  CHECK (push_buffer (pads[0][SIX_UPSTREAM_RTP_PAD]), "single RTP buffer accepted");
  CHECK (push_list (pads[0][SIX_UPSTREAM_RTP_PAD], 3), "RTP buffer list accepted");
  CHECK (snapshot (0, 0, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 4 && sample.last_progress_monotonic_ns > 0,
      "RTP counter counts Buffer and every BufferList member exactly once");
  CHECK (push_list (pads[0][SIX_UPSTREAM_COMPRESSED_PRE_DECODE], 2),
      "compressed pre-decode list accepted");
  CHECK (snapshot (0, 0, SIX_UPSTREAM_COMPRESSED_PRE_DECODE, &sample) &&
      sample.valid && sample.buffers == 2,
      "compressed stage is counted independently from RTP");

  CHECK (snapshot (1, 0, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 0, "one source's progress does not affect another");
  six_upstream_progress_generation_begin (0);
  CHECK (snapshot (0, 0, SIX_UPSTREAM_RTP_PAD, &sample) && !sample.valid,
      "reconnect quarantines old-generation diagnostic sample");
  six_upstream_progress_generation_boundary (0, 1);
  CHECK (!snapshot (0, 0, SIX_UPSTREAM_RTP_PAD, &sample) || !sample.valid,
      "generation-0 snapshot is rejected after transition");
  CHECK (snapshot (0, 1, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 0 && sample.last_progress_monotonic_ns == 0,
      "generation 1 starts with independent zero baseline");
  CHECK (push_buffer (pads[0][SIX_UPSTREAM_RTP_PAD]), "generation 1 buffer accepted");
  CHECK (snapshot (0, 1, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 1, "generation 1 progresses");
  six_upstream_progress_generation_begin (0);
  six_upstream_progress_generation_boundary (0, 2);
  CHECK (snapshot (0, 2, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 0, "generation 2 resets independently");
  CHECK (snapshot (1, 0, SIX_UPSTREAM_COMPRESSED_PRE_DECODE, &sample) &&
      sample.valid && sample.source_generation == 0 && sample.buffers == 0,
      "unaffected source retains its generation and baseline");

  CHECK (push_buffer (pads[5][SIX_UPSTREAM_RTP_PAD]),
      "old dynamic RTP pad contributes before replacement");
  six_upstream_progress_detach (pads[5][SIX_UPSTREAM_RTP_PAD], 5,
      SIX_UPSTREAM_RTP_PAD);
  CHECK (push_buffer (pads[5][SIX_UPSTREAM_RTP_PAD]),
      "detached old pad remains harmlessly chainable");
  GstPad *replacement = make_pad ("source5-rtp-replacement");
  CHECK (replacement && six_upstream_progress_attach (replacement, 5,
      SIX_UPSTREAM_RTP_PAD), "replacement dynamic pad receives one probe");
  CHECK (push_list (replacement, 2), "replacement RTP buffer list accepted");
  CHECK (snapshot (5, 0, SIX_UPSTREAM_RTP_PAD, &sample) && sample.valid &&
      sample.buffers == 3,
      "pad replacement excludes detached-pad traffic and counts the new pad");
  six_upstream_progress_detach (replacement, 5, SIX_UPSTREAM_RTP_PAD);
  gst_pad_set_active (replacement, FALSE);
  gst_object_unref (replacement);

  gint start = 0;
  PushArgs args[3] = {
    {pads[2][SIX_UPSTREAM_RTP_PAD], &start},
    {pads[3][SIX_UPSTREAM_COMPRESSED_PRE_DECODE], &start},
    {pads[4][SIX_UPSTREAM_RTP_PAD], &start},
  };
  GThread *threads[3];
  for (guint i = 0; i < G_N_ELEMENTS (threads); i++)
    threads[i] = g_thread_new ("fix6-push", concurrent_push, &args[i]);
  g_atomic_int_set (&start, 1);
  g_usleep (1000);
  six_upstream_progress_shutdown ();
  for (guint i = 0; i < G_N_ELEMENTS (threads); i++)
    g_thread_join (threads[i]);
  CHECK (snapshot (2, 0, SIX_UPSTREAM_RTP_PAD, &sample) &&
      !sample.valid && !sample.probe_attached,
      "concurrent probe removal leaves no attached probe or valid sample");

  for (guint source = 0; source < 6; source++)
    for (guint stage = 0; stage < 2; stage++) {
      gst_pad_set_active (pads[source][stage], FALSE);
      gst_object_unref (pads[source][stage]);
    }
  printf ("PASS fix6_upstream_probe assertions=%u sources=6 stages=2\n", passed);
  return 0;
}
