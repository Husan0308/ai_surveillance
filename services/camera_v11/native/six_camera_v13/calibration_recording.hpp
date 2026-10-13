#pragma once

// Optional, on-demand 120s/180s calibration recording from the existing
// decoded source tees. This branch is deliberately outside the analytics path.
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace calibration_recording {

struct Source {
  std::string camera_id;
  GstElement *tee = nullptr;
};

struct FrameStamp {
  guint64 frame_num = 0;
  guint64 pts_ns = GST_CLOCK_TIME_NONE;
  guint64 dts_ns = GST_CLOCK_TIME_NONE;
  guint64 monotonic_ns = 0;
};

struct MetadataRow {
  bool encoded = false;
  FrameStamp stamp;
  guint width = 0;
  guint height = 0;
};

inline bool source_pts_duration_elapsed(GstClockTime first_pts,
                                        GstClockTime current_pts,
                                        guint duration_sec) {
  if (!GST_CLOCK_TIME_IS_VALID(first_pts) ||
      !GST_CLOCK_TIME_IS_VALID(current_pts) || current_pts < first_pts)
    return false;
  const guint64 requested_ns = static_cast<guint64>(duration_sec) * GST_SECOND;
  return current_pts - first_pts >= requested_ns;
}

static constexpr size_t METADATA_QUEUE_CAPACITY = 32768;

struct Branch {
  std::string camera_id;
  std::string camera_name;
  std::string video_path;
  std::string rows_path;
  GstElement *queue = nullptr;
  GstElement *converter = nullptr;
  GstElement *capsfilter = nullptr;
  GstElement *encoder = nullptr;
  GstElement *parser = nullptr;
  GstElement *mux = nullptr;
  GstElement *sink = nullptr;
  GstPad *tee_pad = nullptr;
  GstPad *sink_pad = nullptr;
  gulong input_probe_id = 0;
  gulong queue_output_probe_id = 0;
  gulong converter_output_probe_id = 0;
  gulong encoder_input_probe_id = 0;
  gulong encoder_output_probe_id = 0;
  gulong encoded_probe_id = 0;
  gulong eos_probe_id = 0;
  std::atomic<guint64> input_frames{0};
  std::atomic<guint64> queue_output_frames{0};
  std::atomic<guint64> converted_frames{0};
  std::atomic<guint64> encoder_input_frames{0};
  std::atomic<guint64> encoder_output_frames{0};
  std::atomic<guint64> encoded_frames{0};
  std::atomic<guint64> queue_overruns{0};
  std::atomic<bool> ready_seen{false};
  std::atomic<bool> eos_seen{false};
  std::atomic<bool> eos_scheduled{false};
  std::atomic<bool> eos_sent{false};
  std::atomic<bool> source_pts_duration_boundary_reached{false};
  std::atomic<guint> width{0};
  std::atomic<guint> height{0};
  std::atomic<guint> fps_num{0};
  std::atomic<guint> fps_den{1};
  std::string pixel_format;
  std::atomic<guint64> first_pts_ns{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> last_pts_ns{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> first_encoded_pts_ns{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> last_encoded_pts_ns{GST_CLOCK_TIME_NONE};
  std::atomic<guint64> first_monotonic_ns{0};
  std::atomic<guint64> last_monotonic_ns{0};
  std::vector<FrameStamp> input_stamps;
  std::vector<FrameStamp> encoded_stamps;
  std::mutex rows_mutex;
  std::condition_variable rows_ready;
  std::array<MetadataRow, METADATA_QUEUE_CAPACITY> metadata_queue{};
  size_t metadata_queue_head = 0;
  size_t metadata_queue_size = 0;
  std::thread metadata_writer;
  std::atomic<bool> metadata_writer_stop{false};
  std::atomic<bool> metadata_writer_done{false};
  std::atomic<bool> metadata_writer_failed{false};
  std::atomic<guint64> metadata_rows_enqueued{0};
  std::atomic<guint64> metadata_rows_written{0};
  std::atomic<guint64> metadata_queue_drops{0};
  std::atomic<guint64> stamp_history_overflows{0};
  std::ofstream rows;
};

struct Session {
  std::string id;
  std::string room;
  std::string directory;
  std::string state = "ARMING";
  std::string error;
  std::string stop_reason;
  std::vector<std::unique_ptr<Branch>> branches;
  std::atomic<bool> start_scheduled{false};
  std::atomic<bool> finalize_scheduled{false};
  std::atomic<bool> stop_scheduled{false};
  std::atomic<int> stream_state{0};  // ARMING=0, ACTIVE=1, STOPPING=2, FAILED=3
  guint64 start_monotonic_ns = 0;
  guint64 end_monotonic_ns = 0;
  std::string start_utc;
  std::string end_utc;
  guint duration_sec = 120;
  bool requested_duration_completed = false;
};

class Controller {
 public:
  using LogFn = void (*)(const char *, const std::string &);

  Controller(GstElement *pipeline, std::vector<Source> sources,
             std::string command_path, std::string status_path,
             std::string output_root, LogFn log_fn)
      : pipeline_(pipeline), sources_(std::move(sources)),
        command_path_(std::move(command_path)), status_path_(std::move(status_path)),
        output_root_(std::move(output_root)), log_fn_(log_fn) {}

  bool begin() {
    std::error_code ec;
    if (command_path_.empty() || status_path_.empty() || output_root_.empty()) return false;
    if (std::filesystem::exists(command_path_, ec) || ec) return false;
    if (!std::filesystem::exists(output_root_, ec) || ec) return false;
    write_status("IDLE");
    g_timeout_add(100, poll_callback, this);
    log("CALIBRATION_RECORDER_READY", "on-demand native-resolution tee branch; no recorder branch active");
    return true;
  }

  bool owns(GstObject *object) const {
    if (!session_ || !object) return false;
    for (const auto &b : session_->branches) {
      for (GstElement *e : {b->queue, b->converter, b->capsfilter,
                            b->encoder, b->parser, b->mux, b->sink}) {
        if (e && (object == GST_OBJECT(e) || gst_object_has_as_ancestor(object, GST_OBJECT(e))))
          return true;
      }
    }
    return false;
  }

  void handle_error(const std::string &message) {
    if (!session_) return;
    session_->error = message;
    session_->state = "FAILED";
    log("CALIBRATION_RECORDER_ERROR", message);
    g_idle_add(abort_callback, this);
  }

  gboolean tick() {
    if (!session_) return G_SOURCE_CONTINUE;
    for (const auto &branch : session_->branches) {
      if (branch->metadata_writer_failed.load(std::memory_order_acquire) ||
          branch->metadata_queue_drops.load(std::memory_order_acquire) != 0 ||
          branch->queue_overruns.load(std::memory_order_acquire) != 0) {
        session_->error = branch->queue_overruns.load() != 0
            ? "recorder branch queue overrun; frames may be missing"
            : branch->metadata_queue_drops.load() != 0
            ? "bounded JSONL metadata queue overflow; recording evidence is incomplete"
            : "JSONL metadata writer failed; recording evidence is incomplete";
        log("CALIBRATION_RECORDER_ERROR", branch->camera_id + " " + session_->error);
        abort_session();
        break;
      }
    }
    if (!session_) return G_SOURCE_CONTINUE;
    if (session_->state == "FAILED") {
      finish_failed_session_if_ready();
      write_status(session_ ? session_->state : last_state_);
      return G_SOURCE_CONTINUE;
    }
    if (session_->state == "ACTIVE" && monotonic_ns() - session_->start_monotonic_ns >=
                                               guint64(session_->duration_sec) * GST_SECOND) {
      request_stop("duration elapsed");
    }
    if (session_ && session_->state == "STOPPING" && monotonic_ns() - stop_requested_ns_ > 8 * GST_SECOND) {
      session_->error = "branch EOS/finalization timed out";
      log("CALIBRATION_RECORDER_ERROR", session_->error);
      abort_session();
    }
    if (session_ && session_->state == "STOPPING" &&
        std::all_of(session_->branches.begin(), session_->branches.end(),
                    [](const auto &b) { return b->eos_seen.load(); }) &&
        !session_->finalize_scheduled.exchange(true)) {
      g_idle_add(finalize_callback, this);
    }
    write_status(session_ ? session_->state : last_state_);
    return G_SOURCE_CONTINUE;
  }

  void shutdown() {
    if (!session_) return;
    if (session_->state == "ACTIVE" || session_->state == "ARMING") {
      session_->error = "native owner stopped before the requested recording completed";
      request_stop("owner shutdown");
    }
    const gint64 deadline = g_get_monotonic_time() + 8 * G_USEC_PER_SEC;
    while (session_ && g_get_monotonic_time() < deadline) {
      while (g_main_context_iteration(nullptr, FALSE)) {}
      g_usleep(10'000);
    }
    if (session_) abort_session();
    // The parent pipeline is already shutting down here. Drain the bounded
    // evidence writer before destroying its Branch storage; this wait is never
    // performed by a streaming callback or during normal live recording.
    while (session_) {
      finish_failed_session_if_ready();
      if (session_) {
        while (g_main_context_iteration(nullptr, FALSE)) {}
        g_usleep(10'000);
      }
    }
    write_status(last_state_);
  }

 private:
  friend struct RecorderTestAccess;

  enum class StreamState { ARMING, ACTIVE, STOPPING, FAILED };
  GstElement *pipeline_ = nullptr;
  std::vector<Source> sources_;
  std::string command_path_;
  std::string status_path_;
  std::string output_root_;
  LogFn log_fn_ = nullptr;
  std::unique_ptr<Session> session_;
  std::string last_state_ = "IDLE";
  std::string last_result_path_;
  gint64 stop_requested_ns_ = 0;
  guint session_counter_ = 0;

  static guint64 monotonic_ns() {
    return static_cast<guint64>(g_get_monotonic_time()) * 1000;
  }

  static std::string utc_now() {
    GDateTime *now = g_date_time_new_now_utc();
    gchar *value = g_date_time_format_iso8601(now);
    std::string result = value ? value : "";
    g_free(value);
    g_date_time_unref(now);
    return result;
  }

  static std::string json_escape(const std::string &value) {
    std::string result;
    for (unsigned char c : value) {
      switch (c) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: if (c >= 0x20) result += static_cast<char>(c); break;
      }
    }
    return result;
  }

  static bool enqueue_metadata(Branch *branch, const MetadataRow &row) {
    if (!branch || branch->metadata_writer_stop.load(std::memory_order_acquire)) {
      if (branch) branch->metadata_queue_drops.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    std::unique_lock<std::mutex> guard(branch->rows_mutex, std::try_to_lock);
    if (!guard.owns_lock() || branch->metadata_queue_size == METADATA_QUEUE_CAPACITY) {
      branch->metadata_queue_drops.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    const size_t tail = (branch->metadata_queue_head + branch->metadata_queue_size) % METADATA_QUEUE_CAPACITY;
    branch->metadata_queue[tail] = row;
    ++branch->metadata_queue_size;
    branch->metadata_rows_enqueued.fetch_add(1, std::memory_order_relaxed);
    guard.unlock();
    branch->rows_ready.notify_one();
    return true;
  }

  static void metadata_writer_loop(Branch *branch) noexcept {
    try {
      branch->rows.open(branch->rows_path, std::ios::out | std::ios::trunc);
      if (!branch->rows) branch->metadata_writer_failed.store(true, std::memory_order_release);
      guint64 rows_since_flush = 0;
      for (;;) {
        MetadataRow row;
        bool have_row = false;
        {
          std::unique_lock<std::mutex> guard(branch->rows_mutex);
          branch->rows_ready.wait_for(guard, std::chrono::milliseconds(100), [branch] {
            return branch->metadata_queue_size != 0 ||
                branch->metadata_writer_stop.load(std::memory_order_acquire);
          });
          if (branch->metadata_queue_size != 0) {
            row = branch->metadata_queue[branch->metadata_queue_head];
            branch->metadata_queue_head = (branch->metadata_queue_head + 1) % METADATA_QUEUE_CAPACITY;
            --branch->metadata_queue_size;
            have_row = true;
          } else if (branch->metadata_writer_stop.load(std::memory_order_acquire)) {
            break;
          }
        }
        if (!have_row) continue;
        if (!branch->metadata_writer_failed.load(std::memory_order_acquire)) {
          const char *stage = row.encoded ? "encoded_access_unit" : "decoded_source";
          branch->rows << "{\"stage\":\"" << stage << "\",\"frame_num\":" << row.stamp.frame_num
                       << ",\"pts_ns\":" << row.stamp.pts_ns << ",\"dts_ns\":" << row.stamp.dts_ns
                       << ",\"monotonic_ns\":" << row.stamp.monotonic_ns;
          if (!row.encoded) {
            branch->rows << ",\"width\":" << row.width << ",\"height\":" << row.height
                         << ",\"format\":\"" << json_escape(branch->pixel_format) << "\"";
          }
          branch->rows << "}\n";
          ++rows_since_flush;
          if (!branch->rows) {
            branch->metadata_writer_failed.store(true, std::memory_order_release);
          } else {
            branch->metadata_rows_written.fetch_add(1, std::memory_order_relaxed);
            if (rows_since_flush >= 20) {
              branch->rows.flush();
              rows_since_flush = 0;
              if (!branch->rows) branch->metadata_writer_failed.store(true, std::memory_order_release);
            }
          }
        }
      }
      if (branch->rows.is_open()) {
        branch->rows.flush();
        if (!branch->rows) branch->metadata_writer_failed.store(true, std::memory_order_release);
        branch->rows.close();
      }
    } catch (...) {
      branch->metadata_writer_failed.store(true, std::memory_order_release);
      if (branch->rows.is_open()) branch->rows.close();
    }
    branch->metadata_writer_done.store(true, std::memory_order_release);
  }

  static bool start_metadata_writer(Branch *branch) {
    try {
      branch->metadata_writer = std::thread(metadata_writer_loop, branch);
      return true;
    } catch (...) {
      branch->metadata_writer_failed.store(true, std::memory_order_release);
      branch->metadata_writer_done.store(true, std::memory_order_release);
      return false;
    }
  }

  static void request_metadata_writer_stop(Branch &branch) {
    branch.metadata_writer_stop.store(true, std::memory_order_release);
    branch.rows_ready.notify_all();
  }

  static bool metadata_writer_finished(const Branch &branch) {
    return branch.metadata_writer_done.load(std::memory_order_acquire);
  }

  static void join_metadata_writer_if_finished(Branch &branch) {
    if (branch.metadata_writer.joinable() && metadata_writer_finished(branch))
      branch.metadata_writer.join();
  }

  static void detach_branch_probes(Branch &branch) {
    if (branch.input_probe_id && branch.tee_pad) {
      gst_pad_remove_probe(branch.tee_pad, branch.input_probe_id);
      branch.input_probe_id = 0;
    }
    auto detach_stage = [](GstElement *element, const char *pad_name, gulong &probe_id) {
      if (!probe_id || !element) return;
      GstPad *pad = gst_element_get_static_pad(element, pad_name);
      if (pad) { gst_pad_remove_probe(pad, probe_id); gst_object_unref(pad); }
      probe_id = 0;
    };
    detach_stage(branch.queue, "src", branch.queue_output_probe_id);
    detach_stage(branch.converter, "src", branch.converter_output_probe_id);
    detach_stage(branch.encoder, "sink", branch.encoder_input_probe_id);
    detach_stage(branch.encoder, "src", branch.encoder_output_probe_id);
    if (branch.encoded_probe_id && branch.parser) {
      GstPad *pad = gst_element_get_static_pad(branch.parser, "src");
      if (pad) { gst_pad_remove_probe(pad, branch.encoded_probe_id); gst_object_unref(pad); }
      branch.encoded_probe_id = 0;
    }
    if (branch.eos_probe_id && branch.sink_pad) {
      gst_pad_remove_probe(branch.sink_pad, branch.eos_probe_id);
      branch.eos_probe_id = 0;
    }
  }

  void log(const char *kind, const std::string &detail) {
    if (log_fn_) log_fn_(kind, detail);
  }

  static gboolean poll_callback(gpointer data) {
    return static_cast<Controller *>(data)->poll();
  }

  static gboolean tick_callback(gpointer data) {
    return static_cast<Controller *>(data)->tick();
  }

  static gboolean begin_callback(gpointer data) {
    static_cast<Controller *>(data)->begin_recording();
    return G_SOURCE_REMOVE;
  }

  static gboolean eos_callback(gpointer data) {
    auto *branch = static_cast<Branch *>(data);
    bool expected = false;
    if (!branch->tee_pad || !branch->eos_sent.compare_exchange_strong(expected, true))
      return G_SOURCE_REMOVE;
    const gboolean sent = gst_pad_push_event(branch->tee_pad, gst_event_new_eos());
    if (!sent) {
      branch->eos_seen = true;
      auto *controller = static_cast<Controller *>(g_object_get_data(G_OBJECT(branch->tee_pad), "recording-controller"));
      if (controller && !controller->session_->finalize_scheduled.exchange(true))
        g_idle_add(finalize_callback, controller);
    }
    return G_SOURCE_REMOVE;
  }

  static void schedule_eos(Branch *branch) {
    if (branch && !branch->eos_scheduled.exchange(true))
      g_idle_add(eos_callback, branch);
  }

  static gboolean finalize_callback(gpointer data) {
    static_cast<Controller *>(data)->finalize_if_ready();
    return G_SOURCE_REMOVE;
  }

  static gboolean abort_callback(gpointer data) {
    static_cast<Controller *>(data)->abort_session();
    return G_SOURCE_REMOVE;
  }

  static void queue_overrun(GstElement *, gpointer data) {
    static_cast<Branch *>(data)->queue_overruns.fetch_add(1);
  }

  static GstPadProbeReturn count_stage_buffer(GstPad *, GstPadProbeInfo *info, gpointer data) {
    if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER)
      static_cast<std::atomic<guint64> *>(data)->fetch_add(1, std::memory_order_relaxed);
    return GST_PAD_PROBE_OK;
  }

  static GstPadProbeReturn input_probe(GstPad *pad, GstPadProbeInfo *info, gpointer data) {
    auto *branch = static_cast<Branch *>(data);
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    GstCaps *caps = gst_pad_get_current_caps(pad);
    if (caps && branch->width.load(std::memory_order_relaxed) == 0) {
      GstVideoInfo video_info;
      if (gst_video_info_from_caps(&video_info, caps)) {
        branch->width.store(GST_VIDEO_INFO_WIDTH(&video_info), std::memory_order_relaxed);
        branch->height.store(GST_VIDEO_INFO_HEIGHT(&video_info), std::memory_order_relaxed);
        branch->fps_num.store(GST_VIDEO_INFO_FPS_N(&video_info), std::memory_order_relaxed);
        branch->fps_den.store(GST_VIDEO_INFO_FPS_D(&video_info), std::memory_order_relaxed);
        const GstStructure *st = gst_caps_get_structure(caps, 0);
        const gchar *format = gst_structure_get_string(st, "format");
        branch->pixel_format = format ? format : "unknown";
      }
    }
    if (caps) gst_caps_unref(caps);
    // Session state and deadline are attached to the branch's request pad.
    // Before activation, this probe drops only the recorder branch's warm-up
    // buffers; analytics continues through its independent tee pads.
    auto *session = static_cast<Session *>(g_object_get_data(G_OBJECT(branch->tee_pad), "recording-session"));
    if (!session) return GST_PAD_PROBE_DROP;
    const int stream_state = session->stream_state.load(std::memory_order_acquire);
    if (stream_state == 0) {
      branch->ready_seen = true;
      auto *controller = static_cast<Controller *>(g_object_get_data(G_OBJECT(branch->tee_pad), "recording-controller"));
      if (controller && !session->start_scheduled.exchange(true))
        g_idle_add(begin_callback, controller);
      // This is only the recorder tee pad. Drop warm-up buffers here so
      // encoder counts begin exactly at the shared session gate.
      return GST_PAD_PROBE_DROP;
    }
    if (stream_state != 1) return GST_PAD_PROBE_DROP;
    const guint64 now = monotonic_ns();
    const guint64 current_pts = GST_BUFFER_PTS(buffer);
    const guint64 first_pts = branch->first_pts_ns.load(std::memory_order_acquire);
    const bool pts_duration_elapsed = source_pts_duration_elapsed(
        first_pts, current_pts, session->duration_sec);
    const bool wall_duration_elapsed = now >= session->start_monotonic_ns +
        guint64(session->duration_sec) * GST_SECOND;
    if (pts_duration_elapsed || wall_duration_elapsed) {
      if (pts_duration_elapsed)
        branch->source_pts_duration_boundary_reached.store(true, std::memory_order_release);
      auto *controller = static_cast<Controller *>(g_object_get_data(G_OBJECT(branch->tee_pad), "recording-controller"));
      if (controller) schedule_eos(branch);
      return GST_PAD_PROBE_DROP;
    }
    const guint64 n = branch->input_frames.fetch_add(1);
    FrameStamp stamp{n, GST_BUFFER_PTS(buffer), GST_BUFFER_DTS(buffer), now};
    if (!branch->first_monotonic_ns.load()) {
      branch->first_monotonic_ns.store(now);
      branch->first_pts_ns.store(stamp.pts_ns);
    }
    branch->last_monotonic_ns.store(now);
    branch->last_pts_ns.store(stamp.pts_ns);
    if (branch->input_stamps.size() < branch->input_stamps.capacity())
      branch->input_stamps.push_back(stamp);
    else
      branch->stamp_history_overflows.fetch_add(1, std::memory_order_relaxed);
    enqueue_metadata(branch, MetadataRow{false, stamp, branch->width.load(), branch->height.load()});
    return GST_PAD_PROBE_OK;
  }

  static GstPadProbeReturn encoded_probe(GstPad *, GstPadProbeInfo *info, gpointer data) {
    auto *branch = static_cast<Branch *>(data);
    GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer) return GST_PAD_PROBE_OK;
    const guint64 n = branch->encoded_frames.fetch_add(1);
    const guint64 pts = GST_BUFFER_PTS(buffer);
    const guint64 now = monotonic_ns();
    if (!n) branch->first_encoded_pts_ns.store(pts);
    branch->last_encoded_pts_ns.store(pts);
    const FrameStamp stamp{n, pts, GST_BUFFER_DTS(buffer), now};
    if (branch->encoded_stamps.size() < branch->encoded_stamps.capacity())
      branch->encoded_stamps.push_back(stamp);
    else
      branch->stamp_history_overflows.fetch_add(1, std::memory_order_relaxed);
    enqueue_metadata(branch, MetadataRow{true, stamp, 0, 0});
    return GST_PAD_PROBE_OK;
  }

  static GstPadProbeReturn sink_event_probe(GstPad *, GstPadProbeInfo *info, gpointer data) {
    auto *branch = static_cast<Branch *>(data);
    GstEvent *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (event && GST_EVENT_TYPE(event) == GST_EVENT_EOS) {
      branch->eos_seen = true;
      auto *controller = static_cast<Controller *>(g_object_get_data(G_OBJECT(branch->tee_pad), "recording-controller"));
      if (controller && !controller->session_->finalize_scheduled.exchange(true))
        g_idle_add(finalize_callback, controller);
    }
    return GST_PAD_PROBE_OK;
  }

  static std::pair<std::vector<std::string>, std::vector<std::string>> room_cameras(const std::string &room) {
    if (room == "entrance") return {{"CAM-02", "CAM-05"}, {"Entrance 2", "Entrance 1"}};
    if (room == "main_room") return {{"CAM-03", "CAM-06"}, {"Main Room 1", "Main Room 2"}};
    return {};
  }

  static bool safe_id(const std::string &id) {
    return !id.empty() && id.size() <= 64 && std::all_of(id.begin(), id.end(), [](unsigned char c) {
      return std::isalnum(c) || c == '-' || c == '_';
    });
  }

  bool poll() {
    if (std::filesystem::exists(command_path_)) {
      std::ifstream command_file(command_path_);
      std::string line;
      std::getline(command_file, line);
      command_file.close();
      std::error_code ec;
      std::filesystem::remove(command_path_, ec);
      if (!line.empty()) handle_command(line);
    }
    tick();
    write_status(session_ ? session_->state : last_state_);
    return G_SOURCE_CONTINUE;
  }

  void handle_command(const std::string &line) {
    std::vector<std::string> fields;
    size_t begin = 0;
    while (begin <= line.size()) {
      const size_t end = line.find('\t', begin);
      fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    if (fields.size() == 2 && fields[0] == "STOP") {
      if (!session_ || fields[1] != session_->id ||
          (session_->state != "ACTIVE" && session_->state != "ARMING")) {
        log("CALIBRATION_RECORDER_REJECT", "STOP requires the current active session id");
        return;
      }
      request_stop("operator cancellation");
      return;
    }
    if (fields.size() != 5 || fields[0] != "START" ||
        (fields[4] != "120" && fields[4] != "180")) {
      log("CALIBRATION_RECORDER_REJECT", "command duration must be 120 or 180 seconds");
      return;
    }
    if (session_) {
      log("CALIBRATION_RECORDER_REJECT", "another recording session is still active");
      return;
    }
    const auto mapping = room_cameras(fields[1]);
    if (mapping.first.size() != 2 || !safe_id(fields[2])) {
      log("CALIBRATION_RECORDER_REJECT", "unsupported room or unsafe session id");
      return;
    }
    const std::filesystem::path expected = std::filesystem::path(output_root_) / fields[1] / fields[2];
    const std::filesystem::path supplied = std::filesystem::path(fields[3]).lexically_normal();
    const std::string supplied_text = supplied.string();
    if (supplied != expected.lexically_normal() ||
        supplied_text.rfind(output_root_ + "/", 0) != 0) {
      log("CALIBRATION_RECORDER_REJECT", "recording directory escaped the configured root");
      return;
    }
    std::error_code ec;
    if (std::filesystem::exists(supplied, ec)) {
      log("CALIBRATION_RECORDER_REJECT", "refusing to overwrite an existing recording session");
      return;
    }
    std::filesystem::create_directories(supplied, ec);
    if (ec) {
      log("CALIBRATION_RECORDER_REJECT", "could not create session directory: " + ec.message());
      return;
    }
    auto recording = std::make_unique<Session>();
    recording->id = fields[2];
    recording->room = fields[1];
    recording->directory = supplied.string();
    recording->duration_sec = static_cast<guint>(std::stoul(fields[4]));
    for (size_t i = 0; i < mapping.first.size(); ++i) {
      Source *source = nullptr;
      for (auto &candidate : sources_) if (candidate.camera_id == mapping.first[i]) source = &candidate;
      if (!source || !source->tee) {
        log("CALIBRATION_RECORDER_REJECT", mapping.first[i] + " decoded tee is unavailable");
        return;
      }
      auto branch = create_branch(*source, mapping.second[i], supplied.string(), i + 1,
                                  recording->duration_sec);
      if (!branch) {
        destroy_partial(recording.get());
        log("CALIBRATION_RECORDER_ERROR", "failed to create both native recording branches");
        last_state_ = "FAILED";
        return;
      }
      recording->branches.push_back(std::move(branch));
    }
    session_ = std::move(recording);
    for (auto &branch : session_->branches) {
      g_object_set_data(G_OBJECT(branch->tee_pad), "recording-session", session_.get());
      g_object_set_data(G_OBJECT(branch->tee_pad), "recording-controller", this);
    }
    log("CALIBRATION_RECORDING_ARMED", session_->room + " session=" + session_->id +
        " cameras=" + mapping.first[0] + "," + mapping.first[1] + " duration_sec=" +
        std::to_string(session_->duration_sec) + " native_caps=preserved");
    write_status(session_->state);
  }

  std::unique_ptr<Branch> create_branch(Source &source, const std::string &camera_name,
                                       const std::string &directory, size_t ordinal,
                                       guint duration_sec) {
    auto branch = std::make_unique<Branch>();
    auto fail_branch = [&]() -> std::unique_ptr<Branch> {
      cleanup_branch(*branch);
      return nullptr;
    };
    branch->camera_id = source.camera_id;
    branch->camera_name = camera_name;
    branch->video_path = directory + "/camera_" + std::to_string(ordinal) + ".mp4";
    branch->rows_path = directory + "/frames-" + branch->camera_id + ".jsonl";
    if (std::filesystem::exists(branch->video_path) || std::filesystem::exists(branch->rows_path)) return nullptr;
    {
      std::ofstream preflight(branch->rows_path, std::ios::out | std::ios::trunc);
      if (!preflight) return fail_branch();
    }
    // Bound in-memory correlation history to 120 FPS. Any higher-rate source
    // is recorded but marked REVIEW_REQUIRED instead of growing memory.
    const size_t max_history_rows = static_cast<size_t>(duration_sec) * 120;
    branch->input_stamps.reserve(max_history_rows);
    branch->encoded_stamps.reserve(max_history_rows);
    const std::string suffix = std::to_string(++session_counter_) + "-" + branch->camera_id;
    auto create = [&](const char *factory, const std::string &name) {
      return gst_element_factory_make(factory, name.c_str());
    };
    branch->queue = create("queue", "calrec-queue-" + suffix);
    branch->converter = create("nvvideoconvert", "calrec-convert-" + suffix);
    branch->capsfilter = create("capsfilter", "calrec-caps-" + suffix);
    branch->encoder = create("nvv4l2h264enc", "calrec-encoder-" + suffix);
    branch->parser = create("h264parse", "calrec-parser-" + suffix);
    branch->mux = create("qtmux", "calrec-mux-" + suffix);
    branch->sink = create("filesink", "calrec-file-" + suffix);
    for (GstElement *element : {branch->queue, branch->converter, branch->capsfilter,
                                branch->encoder, branch->parser, branch->mux, branch->sink}) {
      if (!element) return fail_branch();
      // A live GstBin can synchronize a newly-added child immediately. Keep
      // every new element locked in NULL until properties and links are ready.
      gst_element_set_locked_state(element, TRUE);
      if (!gst_bin_add(GST_BIN(pipeline_), element)) return fail_branch();
    }
    // Provide bounded headroom for short encoder scheduling bursts without
    // blocking the shared tee. Sustained lag stays leaky and is rejected by
    // frame counters during finalization.
    g_object_set(branch->queue, "max-size-buffers", 24u, "max-size-bytes", 0u,
                 "max-size-time", guint64(0), "leaky", 2, "silent", TRUE, nullptr);
    g_signal_connect(branch->queue, "overrun", G_CALLBACK(queue_overrun), branch.get());
    g_object_set(branch->converter, "gpu-id", 0u, nullptr);
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=NV12");
    g_object_set(branch->capsfilter, "caps", caps, nullptr);
    gst_caps_unref(caps);
    // Constant bitrate is selected from source dimensions after caps arrive;
    // encoder values below are a safe high-resolution default for this fleet.
    g_object_set(branch->encoder, "bitrate", 30000000u, "control-rate", 1,
                 "iframeinterval", 40u, "idrinterval", 40u, "num-B-Frames", 0u,
                 "insert-sps-pps", TRUE, "profile", 4, nullptr);
    // Do not enable qtmux faststart: it spools media to a temporary file and
    // leaves the requested MP4 path empty until EOS. The finalized normal MP4
    // remains playable and can be monitored for live file growth.
    g_object_set(branch->sink, "location", branch->video_path.c_str(),
                 "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE, nullptr);
    if (!gst_element_link_many(branch->queue, branch->converter, branch->capsfilter,
                               branch->encoder, branch->parser,
                               branch->mux, branch->sink, nullptr)) {
      return fail_branch();
    }
    for (GstElement *element : {branch->queue, branch->converter, branch->capsfilter,
                                branch->encoder, branch->parser, branch->mux, branch->sink}) {
      gst_element_set_locked_state(element, FALSE);
      if (!gst_element_sync_state_with_parent(element)) return fail_branch();
    }
    branch->tee_pad = gst_element_request_pad_simple(source.tee, "src_%u");
    GstPad *queue_sink = gst_element_get_static_pad(branch->queue, "sink");
    if (!branch->tee_pad || !queue_sink || gst_pad_link(branch->tee_pad, queue_sink) != GST_PAD_LINK_OK) {
      if (queue_sink) gst_object_unref(queue_sink);
      return fail_branch();
    }
    gst_object_unref(queue_sink);
    branch->input_probe_id = gst_pad_add_probe(branch->tee_pad, GST_PAD_PROBE_TYPE_BUFFER,
                                                input_probe, branch.get(), nullptr);
    GstPad *queue_src = gst_element_get_static_pad(branch->queue, "src");
    GstPad *converter_src = gst_element_get_static_pad(branch->converter, "src");
    GstPad *encoder_sink = gst_element_get_static_pad(branch->encoder, "sink");
    GstPad *encoder_src = gst_element_get_static_pad(branch->encoder, "src");
    if (!queue_src || !converter_src || !encoder_sink || !encoder_src) {
      if (queue_src) gst_object_unref(queue_src);
      if (converter_src) gst_object_unref(converter_src);
      if (encoder_sink) gst_object_unref(encoder_sink);
      if (encoder_src) gst_object_unref(encoder_src);
      return fail_branch();
    }
    branch->queue_output_probe_id = gst_pad_add_probe(
        queue_src, GST_PAD_PROBE_TYPE_BUFFER, count_stage_buffer,
        &branch->queue_output_frames, nullptr);
    branch->converter_output_probe_id = gst_pad_add_probe(
        converter_src, GST_PAD_PROBE_TYPE_BUFFER, count_stage_buffer,
        &branch->converted_frames, nullptr);
    branch->encoder_input_probe_id = gst_pad_add_probe(
        encoder_sink, GST_PAD_PROBE_TYPE_BUFFER, count_stage_buffer,
        &branch->encoder_input_frames, nullptr);
    branch->encoder_output_probe_id = gst_pad_add_probe(
        encoder_src, GST_PAD_PROBE_TYPE_BUFFER, count_stage_buffer,
        &branch->encoder_output_frames, nullptr);
    gst_object_unref(queue_src);
    gst_object_unref(converter_src);
    gst_object_unref(encoder_sink);
    gst_object_unref(encoder_src);
    GstPad *parser_src = gst_element_get_static_pad(branch->parser, "src");
    branch->encoded_probe_id = gst_pad_add_probe(parser_src, GST_PAD_PROBE_TYPE_BUFFER,
                                                  encoded_probe, branch.get(), nullptr);
    gst_object_unref(parser_src);
    branch->sink_pad = gst_element_get_static_pad(branch->sink, "sink");
    branch->eos_probe_id = gst_pad_add_probe(branch->sink_pad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
                                              sink_event_probe, branch.get(), nullptr);
    if (!start_metadata_writer(branch.get())) return fail_branch();
    return branch;
  }

  void begin_recording() {
    if (!session_ || session_->state != "ARMING") return;
    if (!std::all_of(session_->branches.begin(), session_->branches.end(),
                     [](const auto &b) { return b->ready_seen.load(); })) {
      session_->start_scheduled = false;
      return;
    }
    session_->start_monotonic_ns = monotonic_ns();
    session_->start_utc = utc_now();
    session_->state = "ACTIVE";
    session_->stream_state.store(1, std::memory_order_release);
    log("CALIBRATION_RECORDING_STARTED", "both decoded tee branches opened on one monotonic session clock; duration_sec=" +
        std::to_string(session_->duration_sec));
    write_status(session_->state);
  }

  void request_stop(const std::string &reason) {
    if (!session_ || session_->state == "STOPPING" || session_->state == "FAILED") return;
    session_->state = "STOPPING";
    session_->stream_state.store(2, std::memory_order_release);
    stop_requested_ns_ = monotonic_ns();
    session_->end_monotonic_ns = stop_requested_ns_;
    session_->end_utc = utc_now();
    session_->stop_reason = reason;
    session_->requested_duration_completed = reason == "duration elapsed";
    log("CALIBRATION_RECORDING_STOPPING", reason + "; sending per-branch EOS so qtmux finalizes MP4 files");
    for (auto &branch : session_->branches) schedule_eos(branch.get());
  }

  void finalize_if_ready() {
    if (!session_) return;
    if (!std::all_of(session_->branches.begin(), session_->branches.end(),
                     [](const auto &b) { return b->eos_seen.load(); })) {
      session_->finalize_scheduled = false;
      return;
    }
    if (!session_->end_monotonic_ns) {
      session_->end_monotonic_ns = monotonic_ns();
      session_->end_utc = utc_now();
    }
    for (auto &branch : session_->branches) {
      detach_branch_probes(*branch);
      request_metadata_writer_stop(*branch);
    }
    if (!std::all_of(session_->branches.begin(), session_->branches.end(),
                     [](const auto &b) { return metadata_writer_finished(*b); })) {
      session_->finalize_scheduled = false;
      return;
    }
    for (auto &branch : session_->branches) join_metadata_writer_if_finished(*branch);
    bool ok = session_->error.empty() && session_->requested_duration_completed &&
        std::all_of(session_->branches.begin(), session_->branches.end(), [](const auto &b) {
          return b->input_frames.load() > 0 && b->encoded_frames.load() > 0 &&
                 b->input_frames.load() == b->queue_output_frames.load() &&
                 b->queue_output_frames.load() == b->converted_frames.load() &&
                 b->converted_frames.load() == b->encoder_input_frames.load() &&
                 b->encoder_input_frames.load() == b->encoder_output_frames.load() &&
                 b->encoder_output_frames.load() == b->encoded_frames.load() &&
                 b->queue_overruns.load() == 0 && b->input_frames.load() == b->encoded_frames.load() &&
                 b->stamp_history_overflows.load() == 0 &&
                 b->metadata_queue_drops.load() == 0 && !b->metadata_writer_failed.load() &&
                 b->metadata_rows_enqueued.load() == b->metadata_rows_written.load() &&
                 b->metadata_writer_done.load();
        });
    last_state_ = ok ? "COMPLETE" : "REVIEW_REQUIRED";
    session_->state = last_state_;
    last_session_id_ = session_->id;
    if (!write_metadata()) {
      ok = false;
      last_state_ = session_->state = "REVIEW_REQUIRED";
      log("CALIBRATION_RECORDER_ERROR", "recording metadata could not be finalized; session cannot PASS");
    }
    last_result_path_ = session_->directory + "/recording_metadata.json";
    log(ok ? "CALIBRATION_RECORDING_COMPLETE" : "CALIBRATION_RECORDING_REVIEW_REQUIRED",
        "metadata=" + last_result_path_ + " encoded_frames_match=" + std::to_string(ok));
    remove_branches();
    session_.reset();
    write_status(last_state_);
  }

  void abort_session() {
    if (!session_) return;
    if (!session_->end_monotonic_ns) {
      session_->end_monotonic_ns = monotonic_ns();
      session_->end_utc = utc_now();
    }
    session_->stream_state.store(3, std::memory_order_release);
    session_->state = "FAILED";
    last_session_id_ = session_->id;
    for (auto &branch : session_->branches) {
      detach_branch_probes(*branch);
      request_metadata_writer_stop(*branch);
    }
    finish_failed_session_if_ready();
  }

  void finish_failed_session_if_ready() {
    if (!session_ || session_->state != "FAILED" ||
        !std::all_of(session_->branches.begin(), session_->branches.end(),
                     [](const auto &b) { return metadata_writer_finished(*b); })) return;
    for (auto &branch : session_->branches) join_metadata_writer_if_finished(*branch);
    last_session_id_ = session_->id;
    if (!write_metadata()) log("CALIBRATION_RECORDER_ERROR", "failed-session metadata could not be finalized");
    last_state_ = "FAILED";
    last_result_path_ = session_->directory + "/recording_metadata.json";
    remove_branches();
    session_.reset();
    write_status(last_state_);
  }

  void remove_branches() {
    if (!session_) return;
    for (auto &branch : session_->branches) cleanup_branch(*branch);
  }

  void destroy_partial(Session *session) {
    if (!session) return;
    for (auto &branch : session->branches) cleanup_branch(*branch);
  }

  void cleanup_branch(Branch &branch) {
    detach_branch_probes(branch);
    request_metadata_writer_stop(branch);
    if (branch.metadata_writer.joinable()) branch.metadata_writer.join();
    if (branch.tee_pad && branch.queue) {
      GstPad *queue_sink = gst_element_get_static_pad(branch.queue, "sink");
      if (queue_sink) { gst_pad_unlink(branch.tee_pad, queue_sink); gst_object_unref(queue_sink); }
      GstObject *parent = gst_pad_get_parent(branch.tee_pad);
      if (parent) {
        gst_element_release_request_pad(GST_ELEMENT(parent), branch.tee_pad);
        gst_object_unref(parent);
      }
      gst_object_unref(branch.tee_pad);
      branch.tee_pad = nullptr;
    }
    if (branch.sink_pad) { gst_object_unref(branch.sink_pad); branch.sink_pad = nullptr; }
    std::vector<GstElement *> elements{branch.sink, branch.mux, branch.parser,
        branch.encoder, branch.capsfilter, branch.converter, branch.queue};
    for (GstElement *element : elements) if (element) gst_element_set_state(element, GST_STATE_NULL);
    for (GstElement *element : elements) {
      if (!element) continue;
      if (GST_OBJECT_PARENT(element) == GST_OBJECT(pipeline_))
        gst_bin_remove(GST_BIN(pipeline_), element);
      else
        gst_object_unref(element);
    }
  }

  bool write_metadata() {
    if (!session_) return false;
    std::ofstream out(session_->directory + "/recording_metadata.json.tmp", std::ios::trunc);
    out << "{\n  \"schema_version\":1,\n  \"session_id\":\"" << json_escape(session_->id)
        << "\",\n  \"room\":\"" << json_escape(session_->room)
        << "\",\n  \"status\":\"" << json_escape(session_->state)
        << "\",\n  \"requested_duration_sec\":" << session_->duration_sec
        << ",\n  \"start_utc\":\"" << json_escape(session_->start_utc)
        << "\",\n  \"end_utc\":\"" << json_escape(session_->end_utc)
        << "\",\n  \"stop_reason\":\"" << json_escape(session_->stop_reason)
        << "\",\n  \"start_monotonic_ns\":" << session_->start_monotonic_ns
        << ",\n  \"end_monotonic_ns\":" << session_->end_monotonic_ns
        << ",\n  \"actual_duration_sec\":"
        << (session_->start_monotonic_ns && session_->end_monotonic_ns
              ? double(session_->end_monotonic_ns - session_->start_monotonic_ns) / GST_SECOND : 0.0)
        << ",\n  \"capture_method\":\"single-owner decoded NVMM tee; NVENC H.264; native dimensions; source PTS preserved\",\n  \"mp4_mux\":{\"element\":\"qtmux\",\"faststart\":false,\"finalization\":\"moov index written on EOS; MP4 playable after branch finalizes\"},\n  \"cross_camera_identity\":\"not emitted by current local-NvDCF candidate; native local IDs recorded\",\n  \"cameras\":[\n";
    for (size_t i = 0; i < session_->branches.size(); ++i) {
      const auto &b = *session_->branches[i];
      if (i) out << ",\n";
      out << "    {\"camera_id\":\"" << json_escape(b.camera_id)
          << "\",\"camera_name\":\"" << json_escape(b.camera_name)
          << "\",\"video\":\"" << json_escape(b.video_path)
          << "\",\"frame_rows\":\"" << json_escape(b.rows_path)
          << "\",\"width\":" << b.width.load() << ",\"height\":" << b.height.load()
          << ",\"pixel_format\":\"" << json_escape(b.pixel_format)
          << "\",\"fps_num\":" << b.fps_num.load() << ",\"fps_den\":" << b.fps_den.load()
          << ",\"decoded_input_frames\":" << b.input_frames.load()
          << ",\"queue_output_frames\":" << b.queue_output_frames.load()
          << ",\"converted_frames\":" << b.converted_frames.load()
          << ",\"encoder_input_frames\":" << b.encoder_input_frames.load()
          << ",\"encoder_output_frames\":" << b.encoder_output_frames.load()
          << ",\"encoded_access_units\":" << b.encoded_frames.load()
          << ",\"queue_overruns\":" << b.queue_overruns.load()
          << ",\"metadata_rows_enqueued\":" << b.metadata_rows_enqueued.load()
          << ",\"metadata_rows_written\":" << b.metadata_rows_written.load()
          << ",\"metadata_queue_drops\":" << b.metadata_queue_drops.load()
          << ",\"stamp_history_overflows\":" << b.stamp_history_overflows.load()
          << ",\"metadata_writer_failed\":" << (b.metadata_writer_failed.load() ? "true" : "false")
          << ",\"source_pts_duration_boundary_reached\":"
          << (b.source_pts_duration_boundary_reached.load() ? "true" : "false")
          << ",\"first_pts_ns\":" << b.first_pts_ns.load() << ",\"last_pts_ns\":" << b.last_pts_ns.load()
          << ",\"first_encoded_pts_ns\":" << b.first_encoded_pts_ns.load()
          << ",\"last_encoded_pts_ns\":" << b.last_encoded_pts_ns.load()
          << ",\"first_frame_monotonic_ns\":" << b.first_monotonic_ns.load()
          << ",\"last_frame_monotonic_ns\":" << b.last_monotonic_ns.load() << "}";
    }
    out << "\n  ],\n  \"synchronization\":{\"method\":\"shared monotonic start; each camera stops at the earlier of host-duration or first-source-PTS duration; included source PTS retained\",\"max_pair_frame_arrival_skew_ms\":"
        << max_pair_skew_ms() << "}\n}\n";
    out.close();
    if (!out) {
      std::error_code remove_error;
      std::filesystem::remove(session_->directory + "/recording_metadata.json.tmp", remove_error);
      log("CALIBRATION_RECORDER_ERROR", "recording metadata write failed; refusing PASS");
      return false;
    }
    std::error_code ec;
    std::filesystem::rename(session_->directory + "/recording_metadata.json.tmp",
                             session_->directory + "/recording_metadata.json", ec);
    if (ec) {
      std::error_code remove_error;
      std::filesystem::remove(session_->directory + "/recording_metadata.json.tmp", remove_error);
      log("CALIBRATION_RECORDER_ERROR", "metadata atomic rename failed: " + ec.message());
      return false;
    }
    return true;
  }

  double max_pair_skew_ms() const {
    if (!session_ || session_->branches.size() != 2) return 0.0;
    const auto &a = session_->branches[0]->input_stamps;
    const auto &b = session_->branches[1]->input_stamps;
    size_t j = 0;
    guint64 max_skew = 0;
    for (const auto &left : a) {
      while (j + 1 < b.size() &&
             std::llabs(static_cast<long long>(b[j + 1].monotonic_ns) - static_cast<long long>(left.monotonic_ns)) <
             std::llabs(static_cast<long long>(b[j].monotonic_ns) - static_cast<long long>(left.monotonic_ns))) ++j;
      if (!b.empty()) {
        const guint64 skew = std::llabs(static_cast<long long>(b[j].monotonic_ns) - static_cast<long long>(left.monotonic_ns));
        max_skew = std::max(max_skew, skew);
      }
    }
    return double(max_skew) / 1e6;
  }

  void write_status(const std::string &state) {
    const std::string tmp = status_path_ + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    out << "{\"state\":\"" << json_escape(state) << "\",\"last_session_id\":\""
        << json_escape(session_ ? session_->id : last_session_id_) << "\",\"last_result\":\""
        << json_escape(session_ ? session_->directory + "/recording_metadata.json" : last_result_path_)
        << "\",\"start_monotonic_ns\":" << (session_ ? session_->start_monotonic_ns : 0)
        << ",\"elapsed_sec\":" << (session_ && session_->start_monotonic_ns
              ? double(monotonic_ns() - session_->start_monotonic_ns) / GST_SECOND : 0.0)
        << ",\"cameras\":[";
    if (session_) {
      for (size_t i = 0; i < session_->branches.size(); ++i) {
        const auto &b = *session_->branches[i];
        if (i) out << ',';
        std::error_code ec;
        const auto bytes = std::filesystem::file_size(b.video_path, ec);
        out << "{\"camera_id\":\"" << json_escape(b.camera_id) << "\",\"camera_name\":\""
            << json_escape(b.camera_name) << "\",\"video_bytes\":" << (ec ? 0 : bytes)
            << ",\"input_frames\":" << b.input_frames.load()
            << ",\"encoded_frames\":" << b.encoded_frames.load()
            << ",\"queue_overruns\":" << b.queue_overruns.load()
            << ",\"width\":" << b.width.load() << ",\"height\":" << b.height.load()
            << ",\"fps_num\":" << b.fps_num.load() << ",\"fps_den\":" << b.fps_den.load()
            << ",\"first_pts_ns\":" << b.first_pts_ns.load() << ",\"last_pts_ns\":" << b.last_pts_ns.load() << "}";
      }
    }
    out << "]}\n";
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, status_path_, ec);
  }

  std::string last_session_id_;
};

}  // namespace calibration_recording
