#include "../../scripts/cam_six_validation/nv12_capture.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using namespace stage_nv12;
static Layout layout(std::size_t w, std::size_t h, std::size_t padding) {
  const auto pitch = w + padding;
  const auto y_size = pitch * h;
  const auto uv_size = pitch * (h / 2);
  return {w, h, y_size + uv_size,
          {{{w, h, pitch, 1, 0, y_size},
            {w / 2, h / 2, pitch, 2, y_size, uv_size}}}};
}
static void resolution(std::size_t w, std::size_t h, std::size_t padding,
                       std::size_t expected) {
  auto s = layout(w, h, padding);
  std::string error;
  std::size_t total{};
  assert(validate(s, total, error) && total == expected);
  std::vector<std::uint8_t> src(s.allocation, 0xee);
  for (std::size_t y = 0; y < h; ++y)
    for (std::size_t x = 0; x < w; ++x)
      src[y * s.planes[0].pitch + x] = (x + y) % 223;
  for (std::size_t y = 0; y < h / 2; ++y)
    for (std::size_t x = 0; x < w; x += 2) {
      auto pos = s.planes[1].offset + y * s.planes[1].pitch + x;
      src[pos] = 37; src[pos + 1] = 193;  // distinct interleaved U and V
    }
  std::vector<std::uint8_t> dst(total + 32, 0xa7);
  assert(pack(s, src.data(), src.size(), dst.data() + 16, total, error));
  for (std::size_t y = 0; y < h; ++y)
    for (std::size_t x = 0; x < w; ++x)
      assert(dst[16 + y * w + x] == (x + y) % 223);
  for (std::size_t i = w * h; i < total; i += 2) {
    assert(dst[16 + i] == 37 && dst[16 + i + 1] == 193);
  }
  assert(std::all_of(dst.begin(), dst.begin() + 16, [](auto x) { return x == 0xa7; }));
  assert(std::all_of(dst.end() - 16, dst.end(), [](auto x) { return x == 0xa7; }));
  std::cout << w << "x" << h << " padding=" << padding << " bytes=" << total << "\n";
}
static void rejected(Layout s) {
  std::vector<std::uint8_t> src(1024, 1), dst(1024, 0xa7);
  std::string error;
  assert(!pack(s, src.data(), src.size(), dst.data(), dst.size(), error));
  assert(!error.empty());
  assert(std::all_of(dst.begin(), dst.end(), [](auto x) { return x == 0xa7; }));
}
int main() {
  for (auto padding : {0u, 128u}) {
    resolution(2560, 1440, padding, 5529600);
    resolution(3200, 1800, padding, 8640000);
    resolution(1920, 1620, padding, 4665600);
  }
  auto s = layout(16, 8, 4);
  auto unsupported = s; unsupported.plane_count = 3; rejected(unsupported);
  unsupported = s; unsupported.pitch_linear = false; rejected(unsupported);
  auto bad = s; bad.planes[1].bytes_per_pixel = 1; rejected(bad);
  bad = s; bad.planes[1].width = 16; rejected(bad);
  bad = s; bad.planes[0].pitch = 15; rejected(bad);
  bad = s; bad.planes[0].allocation--; rejected(bad);
  bad = s; bad.planes[1].offset = 0; rejected(bad);
  bad = s; bad.allocation--; rejected(bad);
  bad = s; bad.width = 15; rejected(bad);
  bad = s; bad.height = 0; rejected(bad);
  bad = s; bad.planes[1].offset = SIZE_MAX; rejected(bad);
  bad = s; bad.planes[0].pitch = SIZE_MAX; rejected(bad);
  bad = s; bad.width = SIZE_MAX - 1; bad.height = SIZE_MAX - 1; rejected(bad);
  bad = s; bad.allocation = max_bytes + 1; rejected(bad);
  std::string error; std::size_t total{};
  assert(validate(s, total, error));
  std::vector<std::uint8_t> src(s.allocation, 1), dst(total, 0xa7);
  assert(!pack(s, src.data(), src.size() - 1, dst.data(), dst.size(), error));
  assert(!pack(s, src.data(), src.size(), dst.data(), dst.size() - 1, error));
  assert(!pack(s, nullptr, src.size(), dst.data(), dst.size(), error));
  assert(!pack(s, src.data(), src.size(), nullptr, dst.size(), error));
  assert(!pack(s, src.data(), src.size(), src.data(), src.size(), error));
  auto *invalid = reinterpret_cast<const std::uint8_t*>(UINTPTR_MAX - 1);
  assert(!pack(s, invalid, src.size(), dst.data(), dst.size(), error));
  assert(std::all_of(dst.begin(), dst.end(), [](auto x) { return x == 0xa7; }));
  std::cout << "negative bounds and metadata cases PASS\n";
}
