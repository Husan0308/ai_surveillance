CXX ?= g++
DS_ROOT ?= /opt/nvidia/deepstream/deepstream
TARGET := libnvdsinfer_custom_yolo26_e2e.so
SRC := nvdsparsebbox_yolo26_e2e.cpp

CXXFLAGS += -O2 -std=c++17 -fPIC -Wall -Wextra -Werror
CPPFLAGS += -I$(DS_ROOT)/sources/includes
LDFLAGS += -shared

.PHONY: all clean check

all: $(TARGET)

check:
	@test -f "$(DS_ROOT)/sources/includes/nvdsinfer_custom_impl.h" || \
	  (echo "DeepStream headers not found under $(DS_ROOT)/sources/includes" >&2; exit 1)

$(TARGET): check $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)
