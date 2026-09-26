#ifndef MV3DT_DEEPSTREAM_CROP_ENCODER_H
#define MV3DT_DEEPSTREAM_CROP_ENCODER_H

#include <gst/gst.h>
#include "deepstream_app.h"
#include "gstnvdsmeta.h"

G_BEGIN_DECLS

gboolean mv3dt_crop_encoder_init (gint gpu_id);
void mv3dt_crop_encoder_process (AppCtx *app_ctx, GstBuffer *buffer,
    NvDsBatchMeta *batch_meta);
void mv3dt_crop_encoder_shutdown (void);

G_END_DECLS

#endif
