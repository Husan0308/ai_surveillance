#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvds_tracker_meta.h"

typedef struct {
    uint64_t object_id;
    uint32_t frame_num;
    uint32_t source_id;
    float left;
    float top;
    float width;
    float height;
    float confidence;
    uint32_t age;
    uint32_t tracker_state;
    float visibility;
} CameraV2ShadowTrackRow;

/*
 * Copy the latest row for each target from NVDS_TRACKER_SHADOW_LIST_META.
 *
 * This is diagnostic-only. It never mutates NvDsObjectMeta, never injects a
 * box, and never feeds anything back into NvDCF. The caller can therefore
 * compare current active NvDsObjectMeta IDs with NvDCF's own shadow/inactive
 * target list and tell whether an apparent bbox gap is only an output-state
 * transition or a real tracker termination.
 */
int camera_v2_copy_shadow_tracks(uintptr_t buffer_ptr,
                                 CameraV2ShadowTrackRow *rows,
                                 int max_rows) {
    if (!buffer_ptr || !rows || max_rows <= 0) return 0;

    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int count = 0;
    for (NvDsUserMetaList *node = batch_meta->batch_user_meta_list;
         node != NULL;
         node = node->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) node->data;
        if (!user_meta ||
            user_meta->base_meta.meta_type != NVDS_TRACKER_SHADOW_LIST_META ||
            !user_meta->user_meta_data) {
            continue;
        }

        NvDsTargetMiscDataBatch *shadow_batch =
            (NvDsTargetMiscDataBatch *) user_meta->user_meta_data;
        if (!shadow_batch->list) continue;

        for (uint32_t si = 0; si < shadow_batch->numFilled; ++si) {
            NvDsTargetMiscDataStream *stream = shadow_batch->list + si;
            if (!stream->list) continue;

            for (uint32_t oi = 0; oi < stream->numFilled; ++oi) {
                NvDsTargetMiscDataObject *object = stream->list + oi;
                if (!object->list || object->numObj == 0) continue;
                if (count >= max_rows) return count;

                /* Shadow/trajectory metadata may contain multiple frames for a
                 * target. Use the newest frame rather than assuming list order. */
                NvDsTargetMiscDataFrame *latest = object->list;
                for (uint32_t fi = 1; fi < object->numObj; ++fi) {
                    NvDsTargetMiscDataFrame *candidate = object->list + fi;
                    if (candidate->frameNum >= latest->frameNum) {
                        latest = candidate;
                    }
                }

                CameraV2ShadowTrackRow *dst = &rows[count++];
                dst->object_id = object->uniqueId;
                dst->frame_num = latest->frameNum;
                dst->source_id = stream->streamID;
                dst->left = latest->tBbox.left;
                dst->top = latest->tBbox.top;
                dst->width = latest->tBbox.width;
                dst->height = latest->tBbox.height;
                dst->confidence = latest->confidence;
                dst->age = latest->age;
                dst->tracker_state = (uint32_t) latest->trackerState;
                dst->visibility = latest->visibility;
            }
        }
    }

    return count;
}
