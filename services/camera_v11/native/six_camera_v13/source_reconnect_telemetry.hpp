#pragma once

#include <string_view>

// DeepStream 9.1's nvurisrcbin reports its automatic RTSP retry through this
// forwarded GStreamer warning. Keep the classifier narrow: a generic EOF or
// socket warning is not by itself proof that a reconnect was attempted.
inline bool is_nvurisrcbin_reconnect_attempt_warning(std::string_view message) {
  return message.find("No data from source since last") != std::string_view::npos &&
      message.find("Trying reconnection") != std::string_view::npos;
}
