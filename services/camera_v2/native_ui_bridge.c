#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

typedef struct {
    uint32_t source_id;
    uint32_t reserved;
    uint64_t object_id;
    float left;
    float top;
    float width;
    float height;
    float confidence;
    float tracker_confidence;
} CameraV2TrackRow;

int camera_v2_snapshot_tracks(uintptr_t buffer_ptr, CameraV2TrackRow *rows, int max_rows) {
    if (!buffer_ptr || !rows || max_rows <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int count = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;
        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            if (count >= max_rows) return count;
            CameraV2TrackRow *row = &rows[count++];
            row->source_id = frame_meta->source_id;
            row->reserved = 0;
            row->object_id = (uint64_t)obj->object_id;
            row->left = obj->rect_params.left;
            row->top = obj->rect_params.top;
            row->width = obj->rect_params.width;
            row->height = obj->rect_params.height;
            row->confidence = obj->confidence;
            row->tracker_confidence = obj->tracker_confidence;
        }
    }
    return count;
}
