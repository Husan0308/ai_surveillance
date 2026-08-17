#include <stdint.h>
#include <string.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvds_tracker_meta.h"

#define CAMERA_V2_MAX_REID_FEATURE 512

typedef struct {
    uint32_t source_id;
    uint32_t feature_size;
    uint64_t object_id;
    float left;
    float top;
    float width;
    float height;
    float confidence;
    float tracker_confidence;
    float feature[CAMERA_V2_MAX_REID_FEATURE];
} CameraV2ReidRow;

typedef struct {
    uint32_t source_id;
    uint32_t reserved;
    uint64_t object_id;
    char label[MAX_LABEL_SIZE];
} CameraV2TrackLabel;

static NvDsObjReid *find_object_reid(NvDsObjectMeta *obj) {
    if (!obj) return NULL;
    for (NvDsUserMetaList *node = obj->obj_user_meta_list; node != NULL; node = node->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *)node->data;
        if (!user_meta || !user_meta->user_meta_data) continue;
        if (user_meta->base_meta.meta_type != NVDS_TRACKER_OBJ_REID_META) continue;
        NvDsObjReid *reid = (NvDsObjReid *)user_meta->user_meta_data;
        if (!reid || !reid->ptr_host || reid->featureSize == 0) continue;
        return reid;
    }
    return NULL;
}

int camera_v2_snapshot_reid(uintptr_t buffer_ptr,
                            CameraV2ReidRow *rows,
                            int max_rows) {
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
            if (obj->unique_component_id == 191) continue; /* display-only hold box */

            NvDsObjReid *reid = find_object_reid(obj);
            if (!reid) continue;
            if (count >= max_rows) return count;

            CameraV2ReidRow *row = &rows[count++];
            memset(row, 0, sizeof(*row));
            row->source_id = frame_meta->source_id;
            row->object_id = (uint64_t)obj->object_id;
            row->left = obj->rect_params.left;
            row->top = obj->rect_params.top;
            row->width = obj->rect_params.width;
            row->height = obj->rect_params.height;
            row->confidence = obj->confidence;
            row->tracker_confidence = obj->tracker_confidence;

            uint32_t size = reid->featureSize;
            if (size > CAMERA_V2_MAX_REID_FEATURE) size = CAMERA_V2_MAX_REID_FEATURE;
            row->feature_size = size;
            memcpy(row->feature, reid->ptr_host, sizeof(float) * size);
        }
    }
    return count;
}

static const CameraV2TrackLabel *find_track_label(const CameraV2TrackLabel *labels,
                                                   int count,
                                                   uint32_t source_id,
                                                   uint64_t object_id) {
    if (!labels || count <= 0) return NULL;
    for (int i = 0; i < count; ++i) {
        const CameraV2TrackLabel *label = &labels[i];
        if (label->source_id == source_id && label->object_id == object_id) {
            return label;
        }
    }
    return NULL;
}

int camera_v2_apply_track_labels(uintptr_t buffer_ptr,
                                 const CameraV2TrackLabel *labels,
                                 int count) {
    if (!buffer_ptr || !labels || count <= 0) return 0;
    GstBuffer *buffer = (GstBuffer *)buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int applied = 0;
    for (NvDsMetaList *fnode = batch_meta->frame_meta_list; fnode != NULL; fnode = fnode->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)fnode->data;
        if (!frame_meta) continue;

        for (NvDsMetaList *onode = frame_meta->obj_meta_list; onode != NULL; onode = onode->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)onode->data;
            if (!obj || obj->class_id != 0 || obj->object_id == UNTRACKED_OBJECT_ID) continue;
            const CameraV2TrackLabel *label = find_track_label(
                labels,
                count,
                frame_meta->source_id,
                (uint64_t)obj->object_id
            );
            if (!label || !label->label[0]) continue;
            g_strlcpy(obj->obj_label, label->label, MAX_LABEL_SIZE);
            ++applied;
        }
    }
    return applied;
}
