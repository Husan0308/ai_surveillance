#include "../../scripts/cam_six_validation/credential_redaction.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
  const std::vector<std::string> credentials = {
      "native-user/name", "native-pass+marker",
      "native-user/name:native-pass+marker",
  };
  const std::string input =
      "CAM-02 WARNING: username=native-user/name password=native-pass+marker "
      "URI=rtsp://native-user%2Fname:native-pass%2Bmarker@example.invalid/private "
      "WWW-Authenticate=Basic " +
      credential_redaction::base64_encode("native-user/name:native-pass+marker");
  const std::string output = credential_redaction::sanitize(input, credentials);
  const std::vector<std::string> forbidden = {
      "native-user/name", "native-user%2Fname", "native-pass+marker",
      "native-pass%2Bmarker", "example.invalid", "private",
      credential_redaction::base64_encode("native-user/name:native-pass+marker"),
  };
  for (const auto &marker : forbidden) {
    if (output.find(marker) != std::string::npos) {
      std::cerr << "credential redaction regression: sensitive marker retained\n";
      return 1;
    }
  }
  if (output.find("CAM-02 WARNING:") == std::string::npos ||
      output.find("<redacted-rtsp-uri>") == std::string::npos) {
    std::cerr << "credential redaction regression: useful diagnostic lost\n";
    return 1;
  }
  const auto empty = credential_redaction::sanitize("decoder error=EINVAL", {});
  if (empty != "decoder error=EINVAL") {
    std::cerr << "credential redaction regression: empty metadata altered output\n";
    return 1;
  }
  const auto message = credential_redaction::sanitize(
      "bad URI rtsp://broken-%ZZ/camera warning=4", {});
  if (message.find("<redacted-rtsp-uri>") == std::string::npos ||
      message.find("warning=4") == std::string::npos) {
    std::cerr << "credential redaction regression: malformed URI handling failed\n";
    return 1;
  }
  const std::vector<std::string> encoded_config = {"configured%2Fuser", "configured%2Bpass"};
  const auto decoded_log = credential_redaction::sanitize(
      "auth configured/user configured+pass; harmless=metadata",
      encoded_config);
  if (decoded_log.find("configured/user") != std::string::npos ||
      decoded_log.find("configured+pass") != std::string::npos ||
      decoded_log.find("harmless=metadata") == std::string::npos) {
    std::cerr << "credential redaction regression: decoded configured credential leaked\n";
    return 1;
  }
  const auto short_username = credential_redaction::sanitize(
      "username=a decoder=nvh264decoder", {"a"});
  if (short_username.find("username=<redacted>") == std::string::npos ||
      short_username.find("decoder=nvh264decoder") == std::string::npos) {
    std::cerr << "credential redaction regression: short token damaged unrelated diagnostics\n";
    return 1;
  }
  std::cout << "PASS: native warning/error text redacts usernames, passwords, encoded forms and RTSP URLs\n";
  return 0;
}
