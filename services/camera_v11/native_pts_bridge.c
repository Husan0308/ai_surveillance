#include <gst/gst.h>
#include <stdint.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

typedef struct {
    uint32_t source_id;
    uint64_t frame_num;
    uint64_t buf_pts;
} V11FramePtsRow;

int camera_v11_copy_frame_pts(uint64_t gst_buffer_ptr, V11FramePtsRow *rows, int max_rows) {
    if (gst_buffer_ptr == 0 || rows == NULL || max_rows <= 0) {
        return 0;
    }

    GstBuffer *buffer = (GstBuffer *)(uintptr_t)gst_buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta == NULL) {
        return 0;
    }

    int count = 0;
    for (NvDsMetaList *node = batch_meta->frame_meta_list;
         node != NULL && count < max_rows;
         node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)node->data;
        if (frame_meta == NULL) {
            continue;
        }
        rows[count].source_id = (uint32_t)frame_meta->source_id;
        rows[count].frame_num = (uint64_t)frame_meta->frame_num;
        rows[count].buf_pts = (uint64_t)frame_meta->buf_pts;
        ++count;
    }
    return count;
}
