// Isolated DeepStream 9.1 appsrc receiver for Entrance Room live staging.
// Input is packed BGR over stdin; no URI, credential, broker, or map state is
// present in this process.  This app deliberately runs local NvDCF only.
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <nvdsmeta.h>
#include <gstnvdsmeta.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t source_id;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint64_t frame_index;
    int64_t source_pts_ns;
    int64_t source_dts_ns;
    int64_t arrival_mono_ns;
    uint64_t payload_bytes;
};
#pragma pack(pop)
static_assert(sizeof(FrameHeader) == 60, "host frame protocol layout changed");

constexpr uint32_t kMagic = 0x454E5446;  // ENTF
constexpr uint16_t kVersion = 1;
constexpr uint64_t kUntracked = 0xFFFFFFFFFFFFFFFFULL;

struct InputRecord {
    uint64_t frame_index;
    int64_t source_pts_ns;
    int64_t source_dts_ns;
    int64_t arrival_mono_ns;
};

struct Key {
    uint32_t source_id;
    uint64_t pts;
    bool operator==(const Key& other) const {
        return source_id == other.source_id && pts == other.pts;
    }
};
struct KeyHash {
    size_t operator()(const Key& key) const {
        return std::hash<uint64_t>{}((static_cast<uint64_t>(key.source_id) << 56) ^ key.pts);
    }
};

std::unordered_map<Key, InputRecord, KeyHash> g_input_records;
std::unordered_map<Key, uint64_t, KeyHash> g_pgie_input_ns;
std::unordered_map<Key, uint64_t, KeyHash> g_tracker_input_ns;
std::mutex g_record_mutex;
std::mutex g_stage_mutex;
std::mutex g_stdout_mutex;
std::atomic<bool> g_failed{false};
std::atomic<bool> g_eos{false};
std::array<GstElement*, 2> g_appsrc{{nullptr, nullptr}};
GstElement* g_pipeline = nullptr;
bool g_mux_only = false;
bool g_lossless_input = false;

static std::string json_quote(const char* text) {
    std::ostringstream out;
    out << '"';
    if (text) {
        for (const unsigned char c : std::string(text)) {
            switch (c) {
                case '"': out << "\\\""; break;
                case '\\': out << "\\\\"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default:
                    if (c < 0x20) out << ' ';
                    else out << static_cast<char>(c);
            }
        }
    }
    out << '"';
    return out.str();
}

static void emit(const std::string& line) {
    std::lock_guard<std::mutex> lock(g_stdout_mutex);
    std::cout << line << std::endl;
}

static bool read_exact(std::istream& input, char* data, size_t bytes) {
    input.read(data, static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(input.gcount()) == bytes;
}

static uint64_t current_running_time() {
    GstClock* clock = gst_element_get_clock(g_pipeline);
    if (!clock) return GST_CLOCK_TIME_NONE;
    const GstClockTime now = gst_clock_get_time(clock);
    const GstClockTime base = gst_element_get_base_time(g_pipeline);
    gst_object_unref(clock);
    return (GST_CLOCK_TIME_IS_VALID(now) && GST_CLOCK_TIME_IS_VALID(base) && now >= base)
        ? now - base : GST_CLOCK_TIME_NONE;
}

static bool lookup_input(uint32_t source_id, uint64_t pts, InputRecord& record) {
    std::lock_guard<std::mutex> lock(g_record_mutex);
    const auto found = g_input_records.find(Key{source_id, pts});
    if (found == g_input_records.end()) return false;
    record = found->second;
    return true;
}

static uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

static bool take_stage_start(const std::string& stage, const Key& key, uint64_t& start_ns) {
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    auto& starts = stage == "PGIE" ? g_pgie_input_ns : g_tracker_input_ns;
    const auto found = starts.find(key);
    if (found == starts.end()) return false;
    start_ns = found->second;
    starts.erase(found);
    return true;
}

static GstPadProbeReturn stage_input_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) return GST_PAD_PROBE_OK;
    const std::string stage = static_cast<const char*>(user_data);
    auto& starts = stage == "PGIE" ? g_pgie_input_ns : g_tracker_input_ns;
    const uint64_t now_ns = monotonic_ns();
    std::lock_guard<std::mutex> lock(g_stage_mutex);
    for (NvDsMetaList* node = batch->frame_meta_list; node; node = node->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(node->data);
        if (frame) starts[Key{frame->source_id, frame->buf_pts}] = now_ns;
    }
    return GST_PAD_PROBE_OK;
}

static void append_record_fields(std::ostringstream& out, uint32_t source_id, uint64_t pts) {
    InputRecord record{};
    if (!lookup_input(source_id, pts, record)) {
        out << ",\"input_frame_index\":null,\"source_pts_ns\":null"
            << ",\"source_dts_ns\":null,\"arrival_mono_ns\":null";
        return;
    }
    out << ",\"input_frame_index\":" << record.frame_index
        << ",\"source_pts_ns\":" << record.source_pts_ns
        << ",\"source_dts_ns\":" << record.source_dts_ns
        << ",\"arrival_mono_ns\":" << record.arrival_mono_ns;
}

static GstPadProbeReturn metadata_probe(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
    const char* stage = static_cast<const char*>(user_data);
    const uint64_t stage_ns = monotonic_ns();
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    NvDsBatchMeta* batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch) {
        emit(std::string("ERROR {\"stage\":") + json_quote(stage) +
             ",\"message\":\"NvDsBatchMeta missing\"}");
        g_failed = true;
        return GST_PAD_PROBE_OK;
    }

    for (NvDsMetaList* node = batch->frame_meta_list; node; node = node->next) {
        auto* frame = static_cast<NvDsFrameMeta*>(node->data);
        if (!frame) continue;
        const uint32_t sid = frame->source_id;
        const uint64_t pts = frame->buf_pts;
        std::ostringstream out;
        out << std::setprecision(7)
            << stage << " {\"source_id\":" << sid
            << ",\"frame_num\":" << frame->frame_num
            << ",\"buf_pts\":" << pts
            << ",\"batch_id\":" << frame->batch_id
            << ",\"num_obj_meta\":" << frame->num_obj_meta;
        append_record_fields(out, sid, pts);
        out << ",\"stage_mono_ns\":" << stage_ns;
        InputRecord timing_record{};
        if (lookup_input(sid, pts, timing_record) && timing_record.arrival_mono_ns >= 0 &&
            stage_ns >= static_cast<uint64_t>(timing_record.arrival_mono_ns)) {
            out << ",\"host_input_to_stage_ns\":"
                << stage_ns - static_cast<uint64_t>(timing_record.arrival_mono_ns);
        }
        const std::string stage_name(stage);
        if (stage_name == "PGIE" || stage_name == "TRACK") {
            uint64_t start_ns = 0;
            if (take_stage_start(stage_name, Key{sid, pts}, start_ns) && stage_ns >= start_ns) {
                out << (stage_name == "PGIE" ? ",\"pgie_pad_latency_ns\":" : ",\"tracker_pad_latency_ns\":")
                    << stage_ns - start_ns;
            }
        }
        if (std::string(stage) == "MUX") {
            out << '}';
            emit(out.str());
            continue;
        }

        out << ",\"objects\":[";
        bool first = true;
        for (NvDsMetaList* obj_node = frame->obj_meta_list; obj_node; obj_node = obj_node->next) {
            auto* object = static_cast<NvDsObjectMeta*>(obj_node->data);
            if (!object) continue;
            if (!first) out << ',';
            first = false;
            const auto& rect = object->rect_params;
            out << "{\"class_id\":" << object->class_id
                << ",\"confidence\":" << object->confidence
                << ",\"tracker_id\":";
            if (object->object_id == kUntracked) out << "null";
            else out << object->object_id;
            out << ",\"left\":" << rect.left << ",\"top\":" << rect.top
                << ",\"width\":" << rect.width << ",\"height\":" << rect.height
                << ",\"label\":" << json_quote(object->obj_label) << '}';
        }
        out << "]}";
        emit(out.str());
    }
    return GST_PAD_PROBE_OK;
}

static bool add_branch(uint32_t source_id, GstElement* mux, uint32_t width, uint32_t height) {
    const std::string n = std::to_string(source_id);
    GstElement* queue = gst_element_factory_make("queue", ("q" + n).c_str());
    GstElement* convert = gst_element_factory_make("videoconvert", ("vc" + n).c_str());
    GstElement* caps = gst_element_factory_make("capsfilter", ("cpu_caps" + n).c_str());
    GstElement* nvconvert = gst_element_factory_make("nvvideoconvert", ("nvc" + n).c_str());
    GstElement* nvmm_caps = gst_element_factory_make("capsfilter", ("nvmm_caps" + n).c_str());
    if (!g_appsrc[source_id] || !queue || !convert || !caps || !nvconvert || !nvmm_caps) {
        std::cerr << "required appsrc conversion element unavailable for source " << source_id << std::endl;
        return false;
    }

    GstCaps* input_caps = gst_caps_new_simple("video/x-raw",
        "format", G_TYPE_STRING, "BGR", "width", G_TYPE_INT, static_cast<int>(width),
        "height", G_TYPE_INT, static_cast<int>(height), "framerate", GST_TYPE_FRACTION, 20, 1, nullptr);
    g_object_set(g_appsrc[source_id], "caps", input_caps, "is-live", TRUE,
        "format", GST_FORMAT_TIME, "do-timestamp", FALSE, nullptr);
    if (g_lossless_input) {
        g_object_set(g_appsrc[source_id], "block", TRUE,
            "max-buffers", static_cast<guint64>(2), "max-bytes", static_cast<guint64>(0),
            "leaky-type", 0, nullptr);
    } else {
        // Keep the established live-preview bounded/drop-stale behavior unchanged.
        g_object_set(g_appsrc[source_id], "block", FALSE,
            "max-buffers", static_cast<guint64>(2), "leaky-type", 2, nullptr);
    }
    gst_caps_unref(input_caps);
    g_object_set(queue, "max-size-buffers", 2u, "max-size-bytes", 0u,
        "max-size-time", static_cast<guint64>(0),
        "leaky", g_lossless_input ? 0 : 2, nullptr);
    GstCaps* cpu_caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
        "NV12", "width", G_TYPE_INT, static_cast<int>(width), "height", G_TYPE_INT,
        static_cast<int>(height), "framerate", GST_TYPE_FRACTION, 20, 1, nullptr);
    g_object_set(caps, "caps", cpu_caps, nullptr);
    gst_caps_unref(cpu_caps);
    GstCaps* gpu_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12,width=1920,height=1080,framerate=20/1");
    g_object_set(nvmm_caps, "caps", gpu_caps, nullptr);
    gst_caps_unref(gpu_caps);
    gst_bin_add_many(GST_BIN(g_pipeline), g_appsrc[source_id], queue, convert, caps,
        nvconvert, nvmm_caps, nullptr);
    if (!gst_element_link_many(g_appsrc[source_id], queue, convert, caps, nvconvert,
            nvmm_caps, nullptr)) {
        std::cerr << "failed linking CPU BGR -> NV12 -> NVMM branch for source " << source_id << std::endl;
        return false;
    }
    GstPad* srcpad = gst_element_get_static_pad(nvmm_caps, "src");
    const std::string templ = "sink_%u";
    const std::string requested_name = "sink_" + std::to_string(source_id);
    GstPadTemplate* pad_template = gst_element_class_get_pad_template(
        GST_ELEMENT_GET_CLASS(mux), templ.c_str());
    GstPad* sinkpad = pad_template
        ? gst_element_request_pad(mux, pad_template, requested_name.c_str(), nullptr) : nullptr;
    if (!srcpad || !sinkpad) {
        std::cerr << "could not request nvstreammux sink_%u pad; element="
                  << G_OBJECT_TYPE_NAME(mux) << ", template="
                  << (pad_template ? "present" : "missing")
                  << ", request-pad=" << (sinkpad ? "present" : "null") << std::endl;
        if (srcpad) gst_object_unref(srcpad);
        if (sinkpad) gst_object_unref(sinkpad);
        return false;
    }
    GstPadLinkReturn linked = gst_pad_link(srcpad, sinkpad);
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
    if (linked != GST_PAD_LINK_OK) {
        std::cerr << "nvstreammux request pad link failed: " << linked << std::endl;
        return false;
    }
    return true;
}

static bool build_pipeline(bool mux_only, const std::string& pgie_config) {
    g_pipeline = gst_pipeline_new("entrance-live-local-tracking");
    GstElement* mux = gst_element_factory_make("nvstreammux", "mux");
    if (!g_pipeline || !mux) {
        std::cerr << "nvstreammux unavailable" << std::endl;
        return false;
    }
    g_object_set(mux, "batch-size", 2u, "width", 1920u, "height", 1080u,
        "live-source", TRUE, "batched-push-timeout", 40000, "gpu-id", 0u, nullptr);
    // Do not ask the mux to synchronize cameras with independent source clocks.
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(mux), "sync-inputs"))
        g_object_set(mux, "sync-inputs", FALSE, nullptr);
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(mux), "align-inputs"))
        g_object_set(mux, "align-inputs", FALSE, nullptr);
    gst_bin_add(GST_BIN(g_pipeline), mux);
    for (uint32_t sid = 0; sid < 2; ++sid) {
        g_appsrc[sid] = gst_element_factory_make("appsrc", ("src" + std::to_string(sid)).c_str());
        if (!g_appsrc[sid] || !add_branch(sid, mux, 1920, 1080)) return false;
    }

    GstElement* downstream = mux;
    GstElement* pgie = nullptr;
    GstElement* tracker = nullptr;
    if (!mux_only) {
        pgie = gst_element_factory_make("nvinfer", "pgie");
        tracker = gst_element_factory_make("nvtracker", "nvtracker");
        if (!pgie || !tracker || pgie_config.empty()) {
            std::cerr << "nvinfer/nvtracker or PGIE config unavailable" << std::endl;
            return false;
        }
        const char* tracker_yaml = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml";
        g_object_set(pgie, "config-file-path", pgie_config.c_str(), nullptr);
        g_object_set(tracker, "ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file", tracker_yaml, "tracker-width", 1920, "tracker-height", 1080,
            "gpu-id", 0, "display-tracking-id", 1, nullptr);
        gst_bin_add_many(GST_BIN(g_pipeline), pgie, tracker, nullptr);
        if (!gst_element_link_many(mux, pgie, tracker, nullptr)) {
            std::cerr << "failed linking nvstreammux -> selected PGIE -> NvDCF" << std::endl;
            return false;
        }
        downstream = tracker;
        GstPad* pgie_src = gst_element_get_static_pad(pgie, "src");
        GstPad* pgie_sink = gst_element_get_static_pad(pgie, "sink");
        GstPad* tracker_sink = gst_element_get_static_pad(tracker, "sink");
        GstPad* tracker_src = gst_element_get_static_pad(tracker, "src");
        if (!pgie_src || !pgie_sink || !tracker_src || !tracker_sink) return false;
        gst_pad_add_probe(pgie_sink, GST_PAD_PROBE_TYPE_BUFFER, stage_input_probe,
            const_cast<char*>("PGIE"), nullptr);
        gst_pad_add_probe(pgie_src, GST_PAD_PROBE_TYPE_BUFFER, metadata_probe,
            const_cast<char*>("PGIE"), nullptr);
        gst_pad_add_probe(tracker_sink, GST_PAD_PROBE_TYPE_BUFFER, stage_input_probe,
            const_cast<char*>("TRACK"), nullptr);
        gst_pad_add_probe(tracker_src, GST_PAD_PROBE_TYPE_BUFFER, metadata_probe,
            const_cast<char*>("TRACK"), nullptr);
        gst_object_unref(pgie_src);
        gst_object_unref(pgie_sink);
        gst_object_unref(tracker_sink);
        gst_object_unref(tracker_src);
    }
    GstPad* mux_src = gst_element_get_static_pad(mux, "src");
    gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER, metadata_probe,
        const_cast<char*>("MUX"), nullptr);
    gst_object_unref(mux_src);
    GstElement* sink = gst_element_factory_make("fakesink", "sink");
    if (!sink) return false;
    g_object_set(sink, "sync", FALSE, "async", FALSE, nullptr);
    gst_bin_add(GST_BIN(g_pipeline), sink);
    if (!gst_element_link(downstream, sink)) {
        std::cerr << "failed linking DeepStream pipeline to bounded fakesink" << std::endl;
        return false;
    }
    return true;
}

static void feeder_thread() {
    std::array<char, 60> raw{};
    std::array<uint64_t, 2> last_pts{{0, 0}};
    while (!g_failed) {
        if (!read_exact(std::cin, raw.data(), raw.size())) break;
        FrameHeader h{};
        std::memcpy(&h, raw.data(), sizeof(h));
        if (h.magic != kMagic || h.version != kVersion) {
            std::cerr << "invalid host frame protocol header" << std::endl;
            g_failed = true;
            break;
        }
        if (h.source_id == 0xFFFF && h.payload_bytes == 0) break;
        if (h.source_id > 1 || h.width != 1920 || h.height != 1080 || h.stride != 1920 * 3 ||
            h.payload_bytes != static_cast<uint64_t>(h.stride) * h.height) {
            std::cerr << "invalid frame geometry or source ID" << std::endl;
            g_failed = true;
            break;
        }
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, static_cast<gsize>(h.payload_bytes), nullptr);
        if (!buffer) { g_failed = true; break; }
        std::vector<uint8_t> payload(static_cast<size_t>(h.payload_bytes));
        if (!read_exact(std::cin, reinterpret_cast<char*>(payload.data()), payload.size())) {
            gst_buffer_unref(buffer); g_failed = true; break;
        }
        gst_buffer_fill(buffer, 0, payload.data(), payload.size());
        uint64_t pts = current_running_time();
        if (!GST_CLOCK_TIME_IS_VALID(pts)) pts = last_pts[h.source_id] + GST_SECOND / 20;
        if (pts <= last_pts[h.source_id]) pts = last_pts[h.source_id] + 1;
        const uint64_t duration = GST_SECOND / 20;
        GST_BUFFER_PTS(buffer) = pts;
        GST_BUFFER_DTS(buffer) = pts;
        GST_BUFFER_DURATION(buffer) = duration;
        GST_BUFFER_OFFSET(buffer) = h.frame_index;
        GST_BUFFER_OFFSET_END(buffer) = h.frame_index + 1;
        last_pts[h.source_id] = pts;
        {
            std::lock_guard<std::mutex> lock(g_record_mutex);
            g_input_records[Key{h.source_id, pts}] = InputRecord{
                h.frame_index, h.source_pts_ns, h.source_dts_ns, h.arrival_mono_ns};
        }
        GstFlowReturn flow = gst_app_src_push_buffer(GST_APP_SRC(g_appsrc[h.source_id]), buffer);
        if (flow != GST_FLOW_OK) {
            std::cerr << "appsrc push failed for source " << h.source_id << ": " << flow << std::endl;
            g_failed = true;
            break;
        }
    }
    for (GstElement* src : g_appsrc) if (src) gst_app_src_end_of_stream(GST_APP_SRC(src));
    g_eos = true;
}

int main(int argc, char** argv) {
    std::string pgie_config;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mux-only") g_mux_only = true;
        else if (std::string(argv[i]) == "--pgie-config" && i + 1 < argc) pgie_config = argv[++i];
        else if (std::string(argv[i]) == "--lossless-input") g_lossless_input = true;
        else { std::cerr << "usage: deepstream_appsrc [--mux-only] [--pgie-config PATH] [--lossless-input]" << std::endl; return 2; }
    }
    gst_init(&argc, &argv);
    if (!build_pipeline(g_mux_only, pgie_config)) return 3;
    GstStateChangeReturn state = gst_element_set_state(g_pipeline, GST_STATE_PLAYING);
    if (state == GST_STATE_CHANGE_FAILURE) {
        std::cerr << "pipeline refused PLAYING" << std::endl;
        gst_element_set_state(g_pipeline, GST_STATE_NULL);
        return 4;
    }
    emit("READY {\"mux\":\"legacy\",\"batch_size\":2,\"width\":1920,\"height\":1080,\"mode\":" +
         json_quote(g_mux_only ? "mux-only" : "pgie+nvdcf") + "}");
    std::thread feeder(feeder_thread);
    GstBus* bus = gst_element_get_bus(g_pipeline);
    int result = 0;
    bool done = false;
    while (!done) {
        GstMessage* message = gst_bus_timed_pop_filtered(bus, 100 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (!message) continue;
        switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr; gchar* debug = nullptr;
                gst_message_parse_error(message, &err, &debug);
                std::cerr << "GSTREAMER_ERROR " << (GST_OBJECT_NAME(message->src) ? GST_OBJECT_NAME(message->src) : "unknown")
                    << ": " << (err ? err->message : "unknown") << " | " << (debug ? debug : "") << std::endl;
                if (err) g_error_free(err);
                g_free(debug);
                result = 5; done = true; break;
            }
            case GST_MESSAGE_EOS: done = true; break;
            default: break;
        }
        gst_message_unref(message);
        if (g_failed) { result = 6; done = true; }
    }
    if (!g_eos && !g_failed) {
        // EOS can arise after the feeder closes; keep its join deterministic.
    }
    if (feeder.joinable()) feeder.join();
    gst_element_set_state(g_pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(g_pipeline);
    g_pipeline = nullptr;
    emit(result == 0 ? "DONE {\"status\":\"eos\"}" : "DONE {\"status\":\"error\"}");
    return result;
}
