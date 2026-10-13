#include "services/camera_v11/native/six_camera_v13/calibration_recording.hpp"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace calibration_recording {

struct RecorderTestAccess {
  static bool start_writer(Branch *branch) { return Controller::start_metadata_writer(branch); }
  static bool enqueue(Branch *branch, const MetadataRow &row) {
    return Controller::enqueue_metadata(branch, row);
  }
  static void stop_writer(Branch *branch) {
    Controller::request_metadata_writer_stop(*branch);
    if (branch->metadata_writer.joinable()) branch->metadata_writer.join();
  }
  static void set_session(Controller *controller, std::unique_ptr<Session> session) {
    controller->session_ = std::move(session);
  }
  static void finalize(Controller *controller) { controller->finalize_if_ready(); }
  static void request_stop(Controller *controller, const std::string &reason) {
    controller->request_stop(reason);
  }
  static void fail(Controller *controller, const std::string &message) {
    controller->handle_error(message);
  }
  static const std::string &last_state(const Controller &controller) {
    return controller.last_state_;
  }
  static bool session_active(const Controller &controller) {
    return static_cast<bool>(controller.session_);
  }
};

}  // namespace calibration_recording

using namespace calibration_recording;

static MetadataRow test_row(guint64 frame) {
  MetadataRow row;
  row.stamp.frame_num = frame;
  row.stamp.pts_ns = frame * 50'000'000;
  row.stamp.dts_ns = row.stamp.pts_ns;
  row.stamp.monotonic_ns = row.stamp.pts_ns;
  return row;
}

static void test_duration_boundary_uses_source_pts_without_rewriting_it() {
  constexpr GstClockTime first_pts = 50'000'000;
  constexpr GstClockTime duration = 120 * GST_SECOND;
  assert(!source_pts_duration_elapsed(first_pts, first_pts, 120));
  assert(!source_pts_duration_elapsed(first_pts, first_pts + duration - 1, 120));
  assert(source_pts_duration_elapsed(first_pts, first_pts + duration, 120));
  assert(source_pts_duration_elapsed(first_pts, first_pts + duration + 50'000'000, 120));
  assert(!source_pts_duration_elapsed(first_pts + 1, first_pts, 120));
  assert(!source_pts_duration_elapsed(GST_CLOCK_TIME_NONE, first_pts + duration, 120));
  assert(!source_pts_duration_elapsed(first_pts, GST_CLOCK_TIME_NONE, 120));
}

static void test_metadata_writer_fails_closed_on_disk_full() {
  Branch branch;
  branch.rows_path = "/dev/full";
  assert(RecorderTestAccess::start_writer(&branch));
  for (guint64 i = 0; i < 64; ++i) RecorderTestAccess::enqueue(&branch, test_row(i));
  RecorderTestAccess::stop_writer(&branch);
  assert(branch.metadata_writer_done.load());
  assert(branch.metadata_writer_failed.load());
  assert(branch.metadata_rows_enqueued.load() == 64);
  assert(branch.metadata_rows_written.load() < branch.metadata_rows_enqueued.load());
}

static void test_slow_writer_queue_is_bounded_and_enqueue_never_waits() {
  const auto path = std::filesystem::temp_directory_path() /
      ("sixcam-recorder-slow-writer-" + std::to_string(getpid()));
  std::filesystem::remove(path);
  assert(::mkfifo(path.c_str(), 0600) == 0);
  const int reader = ::open(path.c_str(), O_RDONLY | O_NONBLOCK);
  assert(reader >= 0);

  Branch branch;
  branch.rows_path = path.string();
  assert(RecorderTestAccess::start_writer(&branch));
  const auto started = std::chrono::steady_clock::now();
  for (guint64 i = 0; i < METADATA_QUEUE_CAPACITY * 2; ++i)
    RecorderTestAccess::enqueue(&branch, test_row(i));
  const auto enqueue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started).count();
  assert(enqueue_ms < 10'000);
  assert(branch.metadata_queue_drops.load() > 0);
  {
    std::lock_guard<std::mutex> guard(branch.rows_mutex);
    assert(branch.metadata_queue_size <= METADATA_QUEUE_CAPACITY);
  }

  std::thread drain([reader] {
    char buffer[16384];
    for (;;) {
      const ssize_t bytes = ::read(reader, buffer, sizeof(buffer));
      if (bytes > 0) continue;
      if (bytes == 0) break;
      if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  RecorderTestAccess::stop_writer(&branch);
  drain.join();
  ::close(reader);
  std::filesystem::remove(path);
  assert(branch.metadata_writer_done.load());
  assert(branch.metadata_queue_size == 0);
}

static std::unique_ptr<Session> failed_session(const std::filesystem::path &directory,
                                               const std::string &id) {
  std::filesystem::create_directories(directory);
  auto session = std::make_unique<Session>();
  session->id = id;
  session->room = "entrance";
  session->directory = directory.string();
  session->duration_sec = 120;
  for (int i = 0; i < 2; ++i) {
    auto branch = std::make_unique<Branch>();
    branch->camera_id = i == 0 ? "CAM-02" : "CAM-05";
    branch->camera_name = i == 0 ? "Entrance 2" : "Entrance 1";
    branch->video_path = (directory / ("camera_" + std::to_string(i + 1) + ".mp4")).string();
    branch->rows_path = (directory / ("frames-" + branch->camera_id + ".jsonl")).string();
    branch->input_frames = 1;
    branch->queue_output_frames = 1;
    branch->converted_frames = 1;
    branch->encoder_input_frames = 1;
    branch->encoder_output_frames = 1;
    branch->encoded_frames = 1;
    branch->metadata_rows_enqueued = 2;
    branch->metadata_rows_written = 2;
    branch->metadata_writer_done = true;
    branch->eos_seen = true;
    session->branches.push_back(std::move(branch));
  }
  session->start_monotonic_ns = 1;
  session->end_monotonic_ns = 2;
  session->requested_duration_completed = false;
  return session;
}

static std::unique_ptr<Session> complete_session(const std::filesystem::path &directory,
                                                  const std::string &id) {
  std::filesystem::create_directories(directory);
  auto session = std::make_unique<Session>();
  session->id = id;
  session->room = "entrance";
  session->directory = directory.string();
  session->duration_sec = 120;
  session->start_monotonic_ns = 1;
  session->end_monotonic_ns = 120 * GST_SECOND;
  session->start_utc = "start";
  session->end_utc = "end";
  session->requested_duration_completed = true;
  for (int i = 0; i < 2; ++i) {
    auto branch = std::make_unique<Branch>();
    branch->camera_id = i == 0 ? "CAM-02" : "CAM-05";
    branch->camera_name = i == 0 ? "Entrance 2" : "Entrance 1";
    branch->video_path = (directory / ("camera_" + std::to_string(i + 1) + ".mp4")).string();
    branch->rows_path = (directory / ("frames-" + branch->camera_id + ".jsonl")).string();
    branch->width = i == 0 ? 3200 : 2560;
    branch->height = i == 0 ? 1800 : 1440;
    branch->fps_num = 20;
    branch->fps_den = 1;
    branch->pixel_format = "NV12";
    branch->first_pts_ns = 50'000'000;
    branch->last_pts_ns = 120'000'000'000;
    branch->first_encoded_pts_ns = 50'000'000;
    branch->last_encoded_pts_ns = 120'000'000'000;
    branch->first_monotonic_ns = 10;
    branch->last_monotonic_ns = 120 * GST_SECOND;
    branch->input_frames = 3;
    branch->queue_output_frames = 3;
    branch->converted_frames = 3;
    branch->encoder_input_frames = 3;
    branch->encoder_output_frames = 3;
    branch->encoded_frames = 3;
    branch->metadata_rows_enqueued = 6;
    branch->metadata_rows_written = 6;
    branch->metadata_writer_done = true;
    branch->eos_seen = true;
    session->branches.push_back(std::move(branch));
  }
  return session;
}

static void test_interrupted_session_is_review_required_not_complete() {
  const auto directory = std::filesystem::temp_directory_path() /
      ("sixcam-recorder-interrupted-" + std::to_string(getpid()));
  std::filesystem::remove_all(directory);
  GstElement *pipeline = gst_pipeline_new("recorder-interrupted-test");
  assert(pipeline);
  Controller controller(pipeline, {}, "", "", directory.string(), nullptr);
  RecorderTestAccess::set_session(&controller, failed_session(directory, "interrupted"));
  RecorderTestAccess::finalize(&controller);
  assert(!RecorderTestAccess::session_active(controller));
  assert(RecorderTestAccess::last_state(controller) == "REVIEW_REQUIRED");
  std::ifstream metadata(directory / "recording_metadata.json");
  const std::string content((std::istreambuf_iterator<char>(metadata)), {});
  assert(content.find("\"status\":\"REVIEW_REQUIRED\"") != std::string::npos);
  assert(content.find("\"stop_reason\":\"\"") != std::string::npos);
  gst_object_unref(pipeline);
  std::filesystem::remove_all(directory);
}

static void test_encoder_mux_error_marks_session_failed() {
  const auto directory = std::filesystem::temp_directory_path() /
      ("sixcam-recorder-error-" + std::to_string(getpid()));
  std::filesystem::remove_all(directory);
  GstElement *pipeline = gst_pipeline_new("recorder-error-test");
  assert(pipeline);
  Controller controller(pipeline, {}, "", "", directory.string(), nullptr);
  RecorderTestAccess::set_session(&controller, failed_session(directory, "encoder-error"));
  RecorderTestAccess::fail(&controller, "simulated encoder/mux error");
  while (g_main_context_iteration(nullptr, FALSE)) {}
  assert(!RecorderTestAccess::session_active(controller));
  assert(RecorderTestAccess::last_state(controller) == "FAILED");
  std::ifstream metadata(directory / "recording_metadata.json");
  const std::string content((std::istreambuf_iterator<char>(metadata)), {});
  assert(content.find("\"status\":\"FAILED\"") != std::string::npos);
  gst_object_unref(pipeline);
  std::filesystem::remove_all(directory);
}

static void test_early_stop_is_review_required_and_next_session_can_complete() {
  const auto root = std::filesystem::temp_directory_path() /
      ("sixcam-recorder-restart-" + std::to_string(getpid()));
  const auto interrupted_dir = root / "interrupted";
  const auto completed_dir = root / "completed";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  GstElement *pipeline = gst_pipeline_new("recorder-restart-test");
  assert(pipeline);
  Controller controller(pipeline, {}, "", "", root.string(), nullptr);

  RecorderTestAccess::set_session(&controller, failed_session(interrupted_dir, "early-stop"));
  RecorderTestAccess::request_stop(&controller, "operator cancellation");
  RecorderTestAccess::finalize(&controller);
  assert(!RecorderTestAccess::session_active(controller));
  assert(RecorderTestAccess::last_state(controller) == "REVIEW_REQUIRED");
  assert(std::filesystem::exists(interrupted_dir / "recording_metadata.json"));

  RecorderTestAccess::set_session(&controller, complete_session(completed_dir, "next-session"));
  RecorderTestAccess::finalize(&controller);
  assert(!RecorderTestAccess::session_active(controller));
  assert(RecorderTestAccess::last_state(controller) == "COMPLETE");
  std::ifstream metadata(completed_dir / "recording_metadata.json");
  const std::string content((std::istreambuf_iterator<char>(metadata)), {});
  assert(content.find("\"session_id\":\"next-session\"") != std::string::npos);
  assert(content.find("\"status\":\"COMPLETE\"") != std::string::npos);
  gst_object_unref(pipeline);
  std::filesystem::remove_all(root);
}

int main() {
  gst_init(nullptr, nullptr);
  test_duration_boundary_uses_source_pts_without_rewriting_it();
  test_metadata_writer_fails_closed_on_disk_full();
  test_slow_writer_queue_is_bounded_and_enqueue_never_waits();
  test_interrupted_session_is_review_required_not_complete();
  test_encoder_mux_error_marks_session_failed();
  test_early_stop_is_review_required_and_next_session_can_complete();
  g_print("PASS: source-PTS duration bound, disk-full, slow-writer saturation, interrupted stop/restart, and encoder/mux failure are contained\n");
  return 0;
}
