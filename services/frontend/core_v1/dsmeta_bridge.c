#include <stdint.h>
#include <string.h>

#include <gst/gst.h>
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/*
 * Minimal C bridge used by the Python camera+detection runtime.
 *
 * DeepStream 7.1 Python metadata bindings are tied to a narrower Python/OS
 * compatibility matrix than the GStreamer runtime itself.  Keep the hot path
 * native and expose only two tiny functions through ctypes:
 *   - add person object metadata for one source in an nvstreammux batch
 *   - count object metadata for diagnostics
 */

static NvDsFrameMeta *find_frame(NvDsBatchMeta *batch_meta, int source_index) {
    if (!batch_meta) return NULL;
    for (GList *node = batch_meta->frame_meta_list; node != NULL; node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
        if (!frame_meta) continue;
        if ((int) frame_meta->pad_index == source_index ||
            (int) frame_meta->source_id == source_index) {
            return frame_meta;
        }
    }
    return NULL;
}

int dsmeta_add_person_boxes(
    uintptr_t gst_buffer_addr,
    int source_index,
    const float *xyxy,
    const float *confidences,
    int count,
    int source_width,
    int source_height,
    int detection_width,
    int detection_height
) {
    if (!gst_buffer_addr || !xyxy || count <= 0 ||
        source_width <= 0 || source_height <= 0 ||
        detection_width <= 0 || detection_height <= 0) {
        return 0;
    }

    GstBuffer *buffer = (GstBuffer *) gst_buffer_addr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -1;

    NvDsFrameMeta *frame_meta = find_frame(batch_meta, source_index);
    if (!frame_meta) return -2;

    const float sx = (float) source_width / (float) detection_width;
    const float sy = (float) source_height / (float) detection_height;
    int added = 0;

    for (int i = 0; i < count; ++i) {
        const float x1 = xyxy[i * 4 + 0] * sx;
        const float y1 = xyxy[i * 4 + 1] * sy;
        const float x2 = xyxy[i * 4 + 2] * sx;
        const float y2 = xyxy[i * 4 + 3] * sy;

        float left = x1;
        float top = y1;
        float right = x2;
        float bottom = y2;
        if (left < 0.0f) left = 0.0f;
        if (top < 0.0f) top = 0.0f;
        if (right > (float) source_width) right = (float) source_width;
        if (bottom > (float) source_height) bottom = (float) source_height;
        if (right <= left + 1.0f || bottom <= top + 1.0f) continue;

        NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
        if (!obj_meta) continue;

        obj_meta->unique_component_id = 1;
        obj_meta->class_id = 0;
        obj_meta->object_id = UNTRACKED_OBJECT_ID;
        obj_meta->confidence = confidences ? confidences[i] : 1.0f;
        g_strlcpy(obj_meta->obj_label, "Person", MAX_LABEL_SIZE);

        obj_meta->rect_params.left = left;
        obj_meta->rect_params.top = top;
        obj_meta->rect_params.width = right - left;
        obj_meta->rect_params.height = bottom - top;
        obj_meta->rect_params.border_width = 5;
        obj_meta->rect_params.border_color.red = 0.0;
        obj_meta->rect_params.border_color.green = 1.0;
        obj_meta->rect_params.border_color.blue = 0.10;
        obj_meta->rect_params.border_color.alpha = 1.0;
        obj_meta->rect_params.has_bg_color = 0;

        obj_meta->detector_bbox_info.org_bbox_coords.left = left;
        obj_meta->detector_bbox_info.org_bbox_coords.top = top;
        obj_meta->detector_bbox_info.org_bbox_coords.width = right - left;
        obj_meta->detector_bbox_info.org_bbox_coords.height = bottom - top;

        nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
        ++added;
    }

    return added;
}

int dsmeta_count_objects(uintptr_t gst_buffer_addr) {
    if (!gst_buffer_addr) return -1;
    GstBuffer *buffer = (GstBuffer *) gst_buffer_addr;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) return -2;

    int count = 0;
    for (GList *node = batch_meta->frame_meta_list; node != NULL; node = node->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) node->data;
        if (!frame_meta) continue;
        count += (int) frame_meta->num_obj_meta;
    }
    return count;
}
