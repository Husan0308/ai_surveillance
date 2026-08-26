CXX ?= g++
DS_ROOT ?= /opt/nvidia/deepstream/deepstream
CUDA_HOME ?= /usr/local/cuda
TARGET := libnvdsinfer_custom_yolo26_e2e.so
SRC := nvdsparsebbox_yolo26_e2e.cpp

# Keep warnings enabled for our parser, but do not turn warnings originating from
# DeepStream 7.1 headers / CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE into fatal errors.
# NVIDIA's installed nvdsinfer.h uses aggregate initialization that triggers
# -Wmissing-field-initializers with newer host GCC versions, and the prototype
# check macro intentionally has an unused parameter.
CXXFLAGS += -O2 -std=c++17 -fPIC -Wall -Wextra \
            -Wno-missing-field-initializers -Wno-unused-parameter
CPPFLAGS += -isystem $(DS_ROOT)/sources/includes -isystem $(CUDA_HOME)/include
LDFLAGS += -shared

.PHONY: all clean check

all: $(TARGET)

check:
	@test -f "$(DS_ROOT)/sources/includes/nvdsinfer_custom_impl.h" || \
	  (echo "DeepStream headers not found under $(DS_ROOT)/sources/includes" >&2; exit 1)
	@test -f "$(CUDA_HOME)/include/cuda_runtime_api.h" || \
	  (echo "CUDA header not found: $(CUDA_HOME)/include/cuda_runtime_api.h" >&2; exit 1)
	@echo "YOLO26_PARSER_BUILD ds_root=$(DS_ROOT) cuda_home=$(CUDA_HOME)"

$(TARGET): check $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
