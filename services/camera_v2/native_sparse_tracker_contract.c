#include <stdint.h>
#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/*
 * RF-DETR runs sparsely while NvDCF consumes every muxed video frame.
 *
 * DeepStream's tracker contract expects bInferDone to describe whether detector
 * inference for a frame is complete. With an external sparse detector there are
 * many perfectly valid frames where inference is complete but the detection list
 * is empty. Leaving bInferDone false on those frames can stall/mis-schedule the
 * tracker path even though nvstreammux continues producing video.
 *
 * This helper marks every NvDsFrameMeta already present in the current mux batch
 * as inference-complete. Fresh RF-DETR object metadata, when available, is added
 * by the existing detector injection probe immediately afterwards.
 */
int camera_v2_mark_batch_infer_done(uintptr_t buffer_ptr) {
    if (!buffer_ptr) return -1;

    GstBuffer *buffer = (GstBuffer *) buffer_ptr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    int marked = 0;
    for (NvDsMetaList *node = batch_meta->frame_meta_list;
         node != NULL;
         node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
        if (!frame_meta) continue;
        frame_meta->bInferDone = TRUE;
        ++marked;
    }
    return marked;
}
