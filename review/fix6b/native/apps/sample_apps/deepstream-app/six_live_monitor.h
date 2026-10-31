#ifndef SIX_LIVE_MONITOR_H
#define SIX_LIVE_MONITOR_H
#include "deepstream_app.h"
#include "six_live_upstream_progress.h"
gboolean six_live_prepare (AppCtx *app);
gboolean six_live_attach (AppCtx *app);
gboolean six_live_source_auth (GstElement *source, guint id, const gchar *uri);
void six_live_decode_pad (GstPad *pad, guint id);
void six_live_pipeline_state (const gchar *state);
void six_live_source_reconnecting (guint source_id);
void six_live_source_generation_boundary (guint source_id);
gboolean six_live_preview_frame_generation (NvDsFrameMeta *frame_meta,
    guint64 *generation);
void six_live_preview_frame (guint source_id, guint frame_num, guint64 source_pts,
    guint64 source_generation);
void six_live_stop (void);
#endif
