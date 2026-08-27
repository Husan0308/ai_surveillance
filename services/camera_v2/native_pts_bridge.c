#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

typedef struct {
    uint32_t source_id;
    uint32_t pad_index;
    uint64_t frame_num;
    uint64_t buf_pts;
} CameraV95FramePtsRow;

int camera_v95_copy_frame_pts(uintptr_t buffer_ptr,
                              CameraV95FramePtsRow *rows,
                              int max_rows) {
    if (!buffer_ptr || !rows || max_rows <= 0) return 0;

    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int count = 0;
    for (NvDsMetaList *node = batch_meta->frame_meta_list;
         node != NULL && count < max_rows;
         node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)node->data;
        if (!frame_meta) continue;
        CameraV95FramePtsRow *dst = &rows[count++];
        dst->source_id = (uint32_t)frame_meta->source_id;
        dst->pad_index = (uint32_t)frame_meta->pad_index;
        dst->frame_num = (uint64_t)frame_meta->frame_num;
        dst->buf_pts = (uint64_t)frame_meta->buf_pts;
    }
    return count;
}
