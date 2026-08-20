#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

static NvDsFrameMeta *find_frame(NvDsBatchMeta *batch_meta, unsigned int source_id) {
    if (!batch_meta) return NULL;
    for (NvDsMetaList *node = batch_meta->frame_meta_list; node != NULL; node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
        if (!frame_meta) continue;
        if (frame_meta->source_id == source_id || frame_meta->pad_index == source_id) {
            return frame_meta;
        }
    }
    return NULL;
}

int vision_v3_add_person_boxes(uintptr_t buffer_ptr,
                               unsigned int source_id,
                               const float *boxes,
                               int count) {
    if (!buffer_ptr || !boxes || count <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;
    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_id);
    if (!frame_meta) return 0;

    int added = 0;
    for (int i = 0; i < count; ++i) {
        const float *b = boxes + (i * 5);
        const float x1 = b[0];
        const float y1 = b[1];
        const float x2 = b[2];
        const float y2 = b[3];
        const float conf = b[4];
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsObjectMeta *obj = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj) continue;
        const float width = x2 - x1;
        const float height = y2 - y1;

        obj->unique_component_id = 301;
        obj->class_id = 0;
        obj->object_id = UNTRACKED_OBJECT_ID;
        obj->confidence = conf;
        obj->tracker_confidence = -0.1f;
        strncpy(obj->obj_label, "Person", MAX_LABEL_SIZE - 1);
        obj->obj_label[MAX_LABEL_SIZE - 1] = '\0';

        obj->detector_bbox_info.org_bbox_coords.left = x1;
        obj->detector_bbox_info.org_bbox_coords.top = y1;
        obj->detector_bbox_info.org_bbox_coords.width = width;
        obj->detector_bbox_info.org_bbox_coords.height = height;

        obj->rect_params.left = x1;
        obj->rect_params.top = y1;
        obj->rect_params.width = width;
        obj->rect_params.height = height;
        obj->rect_params.border_width = 3;
        obj->rect_params.border_color.red = 0.12;
        obj->rect_params.border_color.green = 1.0;
        obj->rect_params.border_color.blue = 0.20;
        obj->rect_params.border_color.alpha = 1.0;
        obj->rect_params.has_bg_color = 0;

        nvds_add_obj_meta_to_frame(frame_meta, obj, NULL);
        ++added;
    }
    return added;
}
