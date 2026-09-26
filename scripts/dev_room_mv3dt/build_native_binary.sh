#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR=/tmp/mv3dt-crop-build
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR"/*.o
cd "$BUILD_DIR"

read -r -a PKG_CFLAGS <<< "$(pkg-config --cflags gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0)"
read -r -a PKG_LIBS <<< "$(pkg-config --libs gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0)"
INCLUDES=(
  -I/workspace/room-native
  -I/workspace/ds-src/apps/common/includes
  -I/workspace/ds-src/apps/sample_apps/deepstream-app
  -I/workspace/ds-includes
  -I/opt/nvidia/deepstream/deepstream/sources/includes
  -I/usr/local/cuda/include
)
CFLAGS=(-O2 -D_GNU_SOURCE -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 "${INCLUDES[@]}" "${PKG_CFLAGS[@]}")

for source in \
  /workspace/room-native/deepstream_test5_app_main.c \
  /workspace/old-source/deepstream_utc.c \
  /workspace/room-native/bbox_correction.c \
  /workspace/room-native/bbox_history_lifecycle.c \
  /workspace/room-native/deepstream_crop_encoder.c \
  /workspace/ds-src/apps/sample_apps/deepstream-app/deepstream_app.c \
  /workspace/ds-src/apps/sample_apps/deepstream-app/deepstream_app_config_parser.c \
  /workspace/ds-src/apps/common/src/*.c; do
  object="$(basename "${source%.*}").o"
  gcc "${CFLAGS[@]}" -c "$source" -o "$object"
done

for source in \
  /workspace/ds-src/apps/sample_apps/deepstream-app/deepstream_app_config_parser_yaml.cpp \
  /workspace/ds-src/apps/common/src/deepstream-yaml/*.cpp \
  /opt/nvidia/deepstream/deepstream/sources/libs/nvds_msgapi_common_src/nvds_utils.cpp; do
  object="$(basename "${source%.*}").o"
  g++ "${CFLAGS[@]}" -std=c++17 -c "$source" -o "$object"
done

g++ -o /workspace/bin/deepstream-test5-pn263-global-id-proto *.o \
  -L/usr/local/cuda/lib64 -lcudart \
  -L/opt/nvidia/deepstream/deepstream/lib \
  -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_customhelper \
  -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lnvds_batch_jpegenc \
  -lm -lyaml-cpp \
  -lcuda -lgstrtspserver-1.0 -lnvbufsurface -lnvbufsurftransform -ldl \
  -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib -lnvds_logger -lcrypto \
  "${PKG_LIBS[@]}"
chmod +x /workspace/bin/deepstream-test5-pn263-global-id-proto
