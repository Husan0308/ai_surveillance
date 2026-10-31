#ifndef SIX_LIVE_UPSTREAM_PROGRESS_H
#define SIX_LIVE_UPSTREAM_PROGRESS_H
#include <gst/gst.h>

typedef enum {
  SIX_UPSTREAM_RTP_PAD = 0,
  SIX_UPSTREAM_COMPRESSED_PRE_DECODE = 1
} SixUpstreamStage;

typedef struct {
  gboolean probe_attached;
  gboolean valid;
  guint64 source_generation;
  guint64 buffers;
  guint64 last_progress_monotonic_ns;
} SixUpstreamSnapshot;

void six_upstream_progress_init (gboolean enabled);
gboolean six_upstream_progress_attach (GstPad *pad, guint source_id,
    SixUpstreamStage stage);
void six_upstream_progress_detach (GstPad *pad, guint source_id,
    SixUpstreamStage stage);
void six_upstream_progress_generation_begin (guint source_id);
void six_upstream_progress_generation_boundary (guint source_id,
    guint64 generation);
gboolean six_upstream_progress_snapshot (guint source_id,
    guint64 expected_generation, SixUpstreamStage stage,
    SixUpstreamSnapshot *snapshot);
void six_upstream_progress_shutdown (void);

#endif
