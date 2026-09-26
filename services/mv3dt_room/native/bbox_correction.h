#ifndef PN263_BBOX_CORRECTION_H
#define PN263_BBOX_CORRECTION_H

#include <gst/gst.h>

gboolean pn263_bbox_correction_attach (GstElement *primary_gie,
    GstElement *tracker);

#endif
