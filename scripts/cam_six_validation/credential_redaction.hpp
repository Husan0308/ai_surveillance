#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace credential_redaction {

inline std::string lower_ascii(std::string value) {
  for (char &ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value;
}

inline bool is_uri_delimiter(char ch) {
  return std::isspace(static_cast<unsigned char>(ch)) || ch == '"' || ch == '\'' ||
         ch == '<' || ch == '>';
}

inline std::string redact_rtsp_uris(const std::string &input) {
  std::string result;
  result.reserve(input.size());
  for (size_t i = 0; i < input.size();) {
    size_t scheme_size = 0;
    if (i + 7 <= input.size() && lower_ascii(input.substr(i, 7)) == "rtsp://") {
      scheme_size = 7;
    } else if (i + 8 <= input.size() && lower_ascii(input.substr(i, 8)) == "rtsps://") {
      scheme_size = 8;
    }
    if (scheme_size == 0 ||
        (i > 0 && (std::isalnum(static_cast<unsigned char>(input[i - 1])) || input[i - 1] == '_'))) {
      result.push_back(input[i++]);
      continue;
    }
    result += "\x01" "CAM_SIX_RTSP_URI" "\x02";
    i += scheme_size;
    while (i < input.size() && !is_uri_delimiter(input[i])) ++i;
  }
  return result;
}

inline std::string percent_encode(const std::string &value, bool plus_for_space) {
  static constexpr char HEX[] = "0123456789ABCDEF";
  std::string result;
  for (unsigned char ch : value) {
    const bool unreserved = std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~';
    if (unreserved) {
      result.push_back(static_cast<char>(ch));
    } else if (ch == ' ' && plus_for_space) {
      result.push_back('+');
    } else {
      result.push_back('%');
      result.push_back(HEX[ch >> 4]);
      result.push_back(HEX[ch & 0x0f]);
    }
  }
  return result;
}

inline int hex_value(char ch) {
  if (ch >= '0' && ch <= '9') return ch - '0';
  ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
  return -1;
}

inline std::string percent_decode(const std::string &value, bool plus_as_space) {
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '%' && i + 2 < value.size()) {
      const int high = hex_value(value[i + 1]);
      const int low = hex_value(value[i + 2]);
      if (high >= 0 && low >= 0) {
        result.push_back(static_cast<char>((high << 4) | low));
        i += 2;
        continue;
      }
    }
    if (value[i] == '+' && plus_as_space) result.push_back(' ');
    else result.push_back(value[i]);
  }
  return result;
}

inline std::string base64_encode(const std::string &value) {
  static constexpr char TABLE[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  unsigned int accumulator = 0;
  int bits = -6;
  for (unsigned char ch : value) {
    accumulator = (accumulator << 8) | ch;
    bits += 8;
    while (bits >= 0) {
      result.push_back(TABLE[(accumulator >> bits) & 0x3f]);
      bits -= 6;
    }
  }
  if (bits > -6) result.push_back(TABLE[((accumulator << 8) >> (bits + 8)) & 0x3f]);
  while (result.size() % 4) result.push_back('=');
  return result;
}

inline std::string replace_all(std::string text, const std::string &needle,
                               const std::string &replacement, bool ignore_case = false) {
  if (needle.empty()) return text;
  const bool short_token = needle.size() <= 2 &&
      std::all_of(needle.begin(), needle.end(), [](unsigned char ch) { return std::isalnum(ch); });
  if (!ignore_case) {
    size_t at = 0;
    while ((at = text.find(needle, at)) != std::string::npos) {
      if (short_token &&
          ((at > 0 && (std::isalnum(static_cast<unsigned char>(text[at - 1])) || text[at - 1] == '_')) ||
           (at + needle.size() < text.size() &&
            (std::isalnum(static_cast<unsigned char>(text[at + needle.size()])) ||
             text[at + needle.size()] == '_')))) {
        at += needle.size();
        continue;
      }
      text.replace(at, needle.size(), replacement);
      at += replacement.size();
    }
    return text;
  }
  const std::string lower_needle = lower_ascii(needle);
  size_t at = 0;
  while (at < text.size()) {
    const auto found = lower_ascii(text.substr(at)).find(lower_needle);
    if (found == std::string::npos) break;
    at += found;
    if (short_token &&
        ((at > 0 && (std::isalnum(static_cast<unsigned char>(text[at - 1])) || text[at - 1] == '_')) ||
         (at + needle.size() < text.size() &&
          (std::isalnum(static_cast<unsigned char>(text[at + needle.size()])) ||
           text[at + needle.size()] == '_')))) {
      at += needle.size();
      continue;
    }
    text.replace(at, needle.size(), replacement);
    at += replacement.size();
  }
  return text;
}

inline std::string sanitize(std::string text, const std::vector<std::string> &secrets) {
  text = redact_rtsp_uris(text);
  std::vector<std::string> ordered;
  ordered.reserve(secrets.size() * 4);
  for (const auto &secret : secrets) {
    if (secret.empty()) continue;
    for (const auto &decoded : {secret, percent_decode(secret, false), percent_decode(secret, true)}) {
      if (decoded.empty()) continue;
      ordered.push_back(decoded);
      const auto percent = percent_encode(decoded, false);
      const auto plus = percent_encode(decoded, true);
      const auto b64 = base64_encode(decoded);
      if (percent != decoded) ordered.push_back(percent);
      if (plus != decoded && plus != percent) ordered.push_back(plus);
      if (!b64.empty()) ordered.push_back(b64);
    }
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const std::string &left, const std::string &right) {
              return left.size() > right.size();
            });
  for (const auto &secret : ordered) {
    // Percent escape hex digits are case-insensitive. Base64 and raw
    // credentials remain case-sensitive.
    bool percent_encoded = secret.find('%') != std::string::npos;
    text = replace_all(std::move(text), secret, "<redacted>", percent_encoded);
  }
  return replace_all(std::move(text), "\x01" "CAM_SIX_RTSP_URI" "\x02",
                     "<redacted-rtsp-uri>");
}

}  // namespace credential_redaction
