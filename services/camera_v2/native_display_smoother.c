#include <stdint.h>

/*
 * Intentionally a no-op.
 *
 * NvDCF already produces a current-frame tracked bounding box. Earlier versions
 * added a second presentation predictor/hold after nvtracker. That could create
 * rectangles over empty floor, keep an obsolete oversized box alive, and make
 * people counts/labels disagree with the tracker.
 *
 * Keep the exported symbol for ABI compatibility with the native bridge, but do
 * not create, move, resize, or hold any metadata here. The OSD now receives only
 * current NvDCF object metadata.
 */
int camera_v2_smooth_display_boxes(uintptr_t buffer_ptr) {
    return buffer_ptr ? 0 : -1;
}
