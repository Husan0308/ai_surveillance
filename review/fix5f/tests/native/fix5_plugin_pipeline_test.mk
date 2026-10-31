# SPDX-License-Identifier: Apache-2.0
# Build the Fix 5F test binary only; this target does not execute a pipeline.
FIX5_MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
FIX5_ROOT := $(abspath $(dir $(FIX5_MK_PATH))/../..)
FIX5_APP_DIR := $(CURDIR)

include $(FIX5_APP_DIR)/Makefile

FIX5_OUT ?= $(FIX5_ROOT)/.runtime/deepstream-fix5-generation-20261031/review/fix5_plugin_pipeline_test
FIX5_TEST_OBJ := /tmp/fix5_plugin_pipeline_test.o
FIX5_RENAMED_ENTRY_OBJ := /tmp/fix5_deepstream_entrypoint.o
FIX5_LINK_OBJS := $(filter-out deepstream_app_main.o six_live_monitor.o,$(OBJS))

.PHONY: fix5_plugin_pipeline_test
fix5_plugin_pipeline_test: $(FIX5_OUT)

$(FIX5_TEST_OBJ): $(FIX5_ROOT)/tests/native/fix5_plugin_pipeline_test.c
	$(CC) -c -o $@ $(CFLAGS) $<

$(FIX5_RENAMED_ENTRY_OBJ): deepstream_app_main.c Makefile
	$(CC) -Dmain=deepstream_app_entrypoint -c -o $@ $(CFLAGS) $<

$(FIX5_OUT): $(FIX5_TEST_OBJ) $(FIX5_RENAMED_ENTRY_OBJ) $(FIX5_LINK_OBJS)
	mkdir -p "$(dir $@)"
	$(CXX) -Wl,--no-undefined -o $@ $(FIX5_TEST_OBJ) $(FIX5_RENAMED_ENTRY_OBJ) $(FIX5_LINK_OBJS) $(LIBS)
