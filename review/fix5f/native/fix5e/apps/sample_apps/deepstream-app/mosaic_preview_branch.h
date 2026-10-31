#ifndef V11_MOSAIC_PREVIEW_BRANCH_H
#define V11_MOSAIC_PREVIEW_BRANCH_H

#include "deepstream_app.h"

/* Disabled unless V11_UI_MOSAIC_PREVIEW_ENABLE=1 is set by the opt-in runner. */
gboolean mosaic_preview_attach (AppCtx *app_ctx);
void mosaic_preview_stop (void);

#endif
