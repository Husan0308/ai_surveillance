#include "services/camera_v11/native/six_camera_v13/source_reconnect_telemetry.hpp"

#include <cassert>
#include <cstdio>

int main() {
  assert(is_nvurisrcbin_reconnect_attempt_warning(
      "CAM-05-source: No data from source since last 5 sec. Trying reconnection"));
  assert(!is_nvurisrcbin_reconnect_attempt_warning(
      "src: Could not write to resource. Received end-of-file"));
  assert(!is_nvurisrcbin_reconnect_attempt_warning(
      "source warning: decoder did not produce a frame"));
  std::puts("PASS: RTSP retry telemetry counts explicit nvurisrcbin retry warnings only");
}
