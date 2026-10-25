// Built and run inside the pinned DeepStream 9.1 container. It includes the
// actual validator translation unit but never invokes its application entry.
#define PGIE_DIAG_TRACE_PATH "/work/pgie_diag_lifecycle_test.jsonl"
#define main cam_six_validator_application_main
#include "/src/main.cpp"
#undef main

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static unsigned long long no_parser_count() { return 0; }

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "PGIE_DIAG_TEST_FAIL %s\n", message);
    std::exit(1);
  }
}

static std::string read_file(const char *path) {
  std::ifstream input(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input), {});
}

static guint64 hash_file(const char *path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "trace file opens for hash");
  guint64 hash = 1469598103934665603ULL;
  char byte = 0;
  while (input.get(byte)) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

static size_t file_size(const char *path) {
  std::ifstream input(path, std::ios::binary);
  require(bool(input), "trace file opens for size");
  input.seekg(0, std::ios::end);
  return static_cast<size_t>(input.tellg());
}

static std::vector<std::string> lines(const std::string &contents) {
  std::vector<std::string> output;
  // Keep newline handling simple without a second copy of the trace file.
  size_t start = 0;
  while (start < contents.size()) {
    const size_t end = contents.find('\n', start);
    if (end == std::string::npos) {
      output.push_back(contents.substr(start));
      break;
    }
    output.push_back(contents.substr(start, end - start));
    start = end + 1;
  }
  return output;
}

static void reset_ring_for_next_case() {
  pgie_diag_events.fill(PgieDiagEvent{});
  pgie_diag_claimed.store(0);
  pgie_diag_dropped.store(0);
  pgie_diag_export_ready.store(false);
  pgie_diag_dump_started.store(false);
}

static PgieDiagEvent make_event(guint64 index) {
  PgieDiagEvent event{};
  event.pts_ns = 1000 + index;
  event.buffer_cookie = 0xabc000 + index;
  event.batch_no = 20 + index;
  event.source_mask = guint64(1) << (index % 6);
  event.frame_count = guint32(index % 6) + 1;
  event.frame_key = 0xdef000 + index;
  event.stage = PGIE_DIAG_PGIE;
  event.edge = guint8(index % 2);
  pgie_diag_append(event);
  return event;
}

static guint64 json_uint(const std::string &line, const char *key) {
  const std::string token = std::string("\"") + key + "\":";
  const size_t position = line.find(token);
  require(position != std::string::npos, "expected numeric JSON field exists");
  return std::strtoull(line.c_str() + position + token.size(), nullptr, 10);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--setup-error-exit") {
    std::remove(PGIE_DIAG_TRACE_PATH);
    reset_ring_for_next_case();
    pgie_diag_enabled.store(true);
    pgie_diag_pipeline_started.store(false);
    make_event(9);
    require(std::atexit(pgie_diag_atexit_export) == 0,
            "setup failure registers final trace hook");
    std::exit(17);
  }

  std::remove(PGIE_DIAG_TRACE_PATH);
  pgie_diag_enabled.store(false);
  pgie_diag_dump();
  require(pgie_diag_claimed.load() == 0 && pgie_diag_dropped.load() == 0,
          "disabled append/dump leaves ring counters unchanged");
  require(!std::ifstream(PGIE_DIAG_TRACE_PATH).good(),
          "disabled tracing writes no diagnostic file");

  pgie_diag_enabled.store(true);
  parser_errors = no_parser_count;
  parser_rejected = no_parser_count;
  parser_calls = no_parser_count;
  parser_proposals = no_parser_count;
  detection_stats();
  require(!std::ifstream(PGIE_DIAG_TRACE_PATH).good(),
          "periodic statistics do not export the ring");

  for (guint64 i = 0; i < 4; ++i) make_event(i);
  PgieDiagEvent queue_event{};
  queue_event.stage = PGIE_DIAG_CAMERA_QUEUE_BASE + 4;
  queue_event.kind = 1;
  queue_event.value_a = 7;
  queue_event.value_b = 4096;
  queue_event.pts_ns = 800000000;
  pgie_diag_append(queue_event);
  pgie_diag_export_ready.store(true);
  pgie_diag_dump();  // Shutdown export call.
  require(std::ifstream(PGIE_DIAG_TRACE_PATH).good(),
          "shutdown writes the final trace");
  pgie_diag_dump();
  const std::string small_trace = read_file(PGIE_DIAG_TRACE_PATH);
  const auto small_lines = lines(small_trace);
  require(small_lines.size() == 6, "small trace has one header and five events");
  require(small_lines[0].find("\"claimed\":5,\"recorded\":5,\"dropped\":0") !=
              std::string::npos,
          "small trace counters are accurate");
  for (guint64 i = 0; i < 4; ++i) {
    require(json_uint(small_lines[i + 1], "pts_ns") == 1000 + i,
            "retained events preserve append order and PTS");
    require(json_uint(small_lines[i + 1], "buffer_cookie") == 0xabc000 + i,
            "buffer identity is retained");
    require(json_uint(small_lines[i + 1], "batch_no") == 20 + i,
            "batch metadata is retained");
  }
  require(small_lines[1].find("\"stage\":\"nvinfer\"") != std::string::npos,
          "stage name is retained");
  require(small_lines[5].find("\"type\":\"queue\"") != std::string::npos &&
              small_lines[5].find("\"source_id\":4") != std::string::npos &&
              small_lines[5].find("\"buffers\":7") != std::string::npos &&
              small_lines[5].find("\"bytes\":4096") != std::string::npos &&
              small_lines[5].find("\"level_time_ns\":800000000") != std::string::npos,
          "queue events preserve stage, source, and queue metadata");
  const guint64 small_hash = hash_file(PGIE_DIAG_TRACE_PATH);
  const size_t small_size = file_size(PGIE_DIAG_TRACE_PATH);
  pgie_diag_dump();
  pgie_diag_dump();
  require(hash_file(PGIE_DIAG_TRACE_PATH) == small_hash &&
              file_size(PGIE_DIAG_TRACE_PATH) == small_size,
          "repeated shutdown export does not truncate or duplicate output");

  reset_ring_for_next_case();
  std::remove(PGIE_DIAG_TRACE_PATH);
  pgie_diag_export_ready.store(true);
  constexpr guint64 overflow = 17;
  for (guint64 i = 0; i < PGIE_DIAG_CAPACITY + overflow; ++i) make_event(i);
  require(pgie_diag_claimed.load() == PGIE_DIAG_CAPACITY + overflow &&
              pgie_diag_dropped.load() == overflow,
          "ring overflow counter counts each rejected event");
  pgie_diag_dump();

  std::ifstream trace(PGIE_DIAG_TRACE_PATH);
  require(bool(trace), "overflow trace opens");
  std::string line;
  require(bool(std::getline(trace, line)), "overflow header exists");
  require(line.find("\"claimed\":131089,\"recorded\":131072,\"dropped\":17") !=
              std::string::npos,
          "overflow header records claimed, retained, and dropped counts");
  guint64 retained = 0;
  guint64 last_monotonic = 0;
  while (std::getline(trace, line)) {
    require(json_uint(line, "pts_ns") == 1000 + retained,
            "all retained overflow events remain in ring order");
    const guint64 monotonic = json_uint(line, "monotonic_us");
    require(monotonic >= last_monotonic, "sequential append timestamps are monotonic");
    last_monotonic = monotonic;
    if (retained == 0) {
      require(json_uint(line, "source_mask") == 1 &&
                  json_uint(line, "frame_count") == 1 &&
                  json_uint(line, "frame_key") == 0xdef000,
              "source and frame metadata are retained");
    }
    ++retained;
  }
  require(retained == PGIE_DIAG_CAPACITY,
          "trace exports every retained ring slot and no overflow slot");
  const guint64 overflow_hash = hash_file(PGIE_DIAG_TRACE_PATH);
  const size_t overflow_size = file_size(PGIE_DIAG_TRACE_PATH);
  pgie_diag_dump();
  require(hash_file(PGIE_DIAG_TRACE_PATH) == overflow_hash &&
              file_size(PGIE_DIAG_TRACE_PATH) == overflow_size,
          "repeated overflow shutdown does not rewrite the trace");

  trace.close();
  const pid_t child = fork();
  require(child >= 0, "fork setup-error trace test");
  if (child == 0) {
    execl(argv[0], argv[0], "--setup-error-exit", static_cast<char *>(nullptr));
    _exit(127);
  }
  int child_status = 0;
  require(waitpid(child, &child_status, 0) == child, "wait for setup-error trace test");
  require(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 17,
          "setup-error child preserves its failure exit code");
  const auto exit_lines = lines(read_file(PGIE_DIAG_TRACE_PATH));
  require(exit_lines.size() == 2 &&
              exit_lines[0].find("\"claimed\":1,\"recorded\":1,\"dropped\":0") !=
                  std::string::npos &&
              json_uint(exit_lines[1], "pts_ns") == 1009,
          "early error exit writes a final trace through the registered shutdown hook");

  std::puts("PGIE_DIAG_TEST_PASS periodic disabled once order overflow idempotency");
  return 0;
}
