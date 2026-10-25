#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace stage_nv12 {
constexpr std::size_t max_bytes = 64u * 1024u * 1024u;
struct Plane {
  std::size_t width{}, height{}, pitch{}, bytes_per_pixel{}, offset{}, allocation{};
};
struct Layout {
  std::size_t width{}, height{}, allocation{};
  std::array<Plane, 2> planes{};
  std::size_t plane_count = 2;
  bool pitch_linear = true;
};
inline bool add(std::size_t a, std::size_t b, std::size_t &out) {
  if (b > std::numeric_limits<std::size_t>::max() - a) return false;
  out = a + b;
  return true;
}
inline bool mul(std::size_t a, std::size_t b, std::size_t &out) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}
inline bool validate(const Layout &s, std::size_t &packed_bytes, std::string &error) {
  packed_bytes = 0;
  auto fail = [&](const char *reason) { error = reason; return false; };
  if (s.plane_count != 2 || !s.pitch_linear)
    return fail("unsupported plane count or non-pitch-linear layout");
  if (!s.width || !s.height || s.width % 2 || s.height % 2)
    return fail("NV12 requires positive even visible dimensions");
  std::size_t y_bytes{}, uv_bytes{}, total{};
  if (!mul(s.width, s.height, y_bytes) || !mul(s.width, s.height / 2, uv_bytes) ||
      !add(y_bytes, uv_bytes, total) || total > max_bytes)
    return fail("packed size overflow or capture limit exceeded");
  if (!s.allocation || s.allocation > max_bytes)
    return fail("invalid surface allocation size");
  std::size_t previous_end = 0;
  for (std::size_t i = 0; i < 2; ++i) {
    const auto &p = s.planes[i];
    if (p.width != (i ? s.width / 2 : s.width) ||
        p.height != (i ? s.height / 2 : s.height) ||
        p.bytes_per_pixel != (i ? 2u : 1u))
      return fail("unsupported plane geometry or bytes per pixel");
    std::size_t row_bytes{}, footprint{}, end{};
    if (!mul(p.width, p.bytes_per_pixel, row_bytes) || row_bytes != s.width ||
        p.pitch < row_bytes || !mul(p.pitch, p.height, footprint) ||
        footprint > p.allocation || !add(p.offset, p.allocation, end) ||
        end > s.allocation || p.offset < previous_end)
      return fail("plane pitch, overlap, or allocation bounds invalid");
    previous_end = end;
  }
  packed_bytes = total;
  error.clear();
  return true;
}

// CPU-only routine. Caller must supply a synchronized, CPU-accessible allocation.
// Validate every extent before doing any pointer arithmetic or writing a byte.
inline bool pack(const Layout &s, const std::uint8_t *src, std::size_t src_size,
                 std::uint8_t *dst, std::size_t dst_size, std::string &error) {
  std::size_t total{};
  if (!validate(s, total, error)) return false;
  if (!src || !dst || src_size < s.allocation || dst_size < total) {
    error = "null pointer or short source/destination allocation";
    return false;
  }
  const auto a = reinterpret_cast<std::uintptr_t>(src);
  const auto b = reinterpret_cast<std::uintptr_t>(dst);
  if (src_size > std::numeric_limits<std::uintptr_t>::max() - a ||
      dst_size > std::numeric_limits<std::uintptr_t>::max() - b ||
      (a < b + dst_size && b < a + src_size)) {
    error = "pointer range overflow or overlapping allocations";
    return false;
  }
  std::size_t output = 0;
  for (const auto &p : s.planes) {
    for (std::size_t row = 0; row < p.height; ++row) {
      // validate() proved pitch*height, offset+allocation, and total bounds.
      std::memcpy(dst + output, src + p.offset + row * p.pitch, s.width);
      output += s.width;
    }
  }
  return true;
}
}  // namespace stage_nv12
