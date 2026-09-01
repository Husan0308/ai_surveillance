#include <stdint.h>
#include <gst/gst.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

int camera_v11_count_person_detections(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int count = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;
        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            if (!obj || obj->class_id != 0) continue;
            if (obj->rect_params.width <= 1.0f || obj->rect_params.height <= 1.0f) continue;
            ++count;
        }
    }
    return count;
}
