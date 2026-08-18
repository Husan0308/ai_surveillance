#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/*
 * Remove only native heatmap circles that belong to disabled camera tiles.
 * Heat accumulation remains untouched, so re-enabling a camera immediately
 * shows its real recent history instead of starting from zero.
 */
int camera_v2_heatmap_filter(uintptr_t buffer_ptr,
                             unsigned int wall_width,
                             unsigned int wall_height,
                             unsigned int rows,
                             unsigned int columns,
                             uint32_t enabled_mask) {
    if (!buffer_ptr || wall_width < 2 || wall_height < 2 || rows == 0 || columns == 0) {
        return -1;
    }

    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    const float tile_w = (float)wall_width / (float)columns;
    const float tile_h = (float)wall_height / (float)rows;
    const unsigned int tile_count = rows * columns;
    int removed = 0;

    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;

        for (NvDsMetaList *dnode = frame_meta->display_meta_list; dnode != NULL; dnode = dnode->next) {
            NvDsDisplayMeta *display = (NvDsDisplayMeta *)dnode->data;
            if (!display || display->num_circles == 0) continue;

            unsigned int write_index = 0;
            for (unsigned int read_index = 0; read_index < display->num_circles; ++read_index) {
                NvOSD_CircleParams circle = display->circle_params[read_index];

                /* Native heatmap circles use a filled background, width=1 and a
                 * deliberately low alpha. Leave unrelated circles untouched. */
                int is_heat_circle = (
                    circle.has_bg_color &&
                    circle.circle_width == 1 &&
                    circle.bg_color.alpha <= 0.22f
                );

                int keep = 1;
                if (is_heat_circle) {
                    unsigned int col = (unsigned int)((float)circle.xc / tile_w);
                    unsigned int row = (unsigned int)((float)circle.yc / tile_h);
                    if (col >= columns) col = columns - 1;
                    if (row >= rows) row = rows - 1;
                    unsigned int source_id = row * columns + col;
                    if (source_id < tile_count && source_id < 32U) {
                        keep = ((enabled_mask >> source_id) & 1U) != 0U;
                    }
                }

                if (keep) {
                    if (write_index != read_index) {
                        display->circle_params[write_index] = circle;
                    }
                    ++write_index;
                } else {
                    ++removed;
                }
            }
            display->num_circles = write_index;
        }
    }

    return removed;
}
