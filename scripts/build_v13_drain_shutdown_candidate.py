"""Build an isolated live candidate that drains admitted frames before FLUSH_START."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import shutil
import subprocess

from scripts.build_room_candidate import ROOT, SDK, IMAGE, BINARY, input_hashes, diagnostic_source
from scripts.build_v13_shutdown_candidate import EOS, LIVE_CLOSE
from scripts.dev_room_mv3dt.verify_validated_assets import sha256

BASE = ROOT / "services/mv3dt_room/native/deepstream_test5_app_main.c"
SDK_APP = SDK / "src/apps/sample_apps/deepstream-app/deepstream_app.c"
ANCHOR = "#define SOURCE_HEALTH_MAX_RECONNECT_ATTEMPTS 3\n"
NATIVE_DESTROY = """    if (appCtx[i]->return_value == -1)
      return_value = -1;

    destroy_pipeline (appCtx[i]);"""
NATIVE_DESTROY_DRAIN = """    if (appCtx[i]->return_value == -1)
      return_value = -1;

    if (f5_has_rtsp_source (appCtx[i])) {
      if (f5_drain_admitted_frames (appCtx[i])) {
        gint64 flush_start_ns = f5_monotonic_ns ();
        f5_flush_start_ns = flush_start_ns;
        f5_drain_log_event ("flush_start", flush_start_ns);
        if (!gst_element_send_event (appCtx[i]->pipeline.pipeline,
                gst_event_new_flush_start ())) {
          g_printerr ("F5 drain: FLUSH_START rejected after completed drain\\n");
          return_value = -1;
        }
      } else {
        g_printerr ("F5 drain: incomplete; refusing success-path FLUSH_START\\n");
        return_value = -1;
      }
    }

    if (f5_drain_success) {
      gint64 destroy_pipeline_ns = f5_monotonic_ns ();
      f5_destroy_pipeline_ns = destroy_pipeline_ns;
      f5_drain_log_event ("destroy_pipeline", destroy_pipeline_ns);
    }
    destroy_pipeline (appCtx[i]);"""

C_DRAIN = r'''
typedef struct {
  guint source_id;
  GstPad *pad;
  gulong buffer_probe;
  gulong idle_probe;
  guint64 admitted_buffers;
  guint64 last_admitted_pts;
  gint blocked;
  gint64 idle_ns;
} F5DrainSource;

static F5DrainSource f5_drain_sources[MAX_SOURCE_BINS];
static guint f5_drain_source_count;
static GMutex f5_drain_mutex;
static gboolean f5_drain_mutex_initialized;
static gboolean f5_drain_success;
static gint64 f5_shutdown_requested_ns;
static gint64 f5_source_admission_stopped_ns;
static gint64 f5_drain_completed_ns;
static gint64 f5_flush_start_ns;
static gint64 f5_destroy_pipeline_ns;
static gint64 f5_native_exit_ns;

static gint64
f5_monotonic_ns (void)
{
  struct timespec value;
  if (clock_gettime (CLOCK_MONOTONIC, &value) != 0)
    return 0;
  return ((gint64) value.tv_sec * G_GINT64_CONSTANT (1000000000)) + value.tv_nsec;
}

static gboolean
f5_has_rtsp_source (AppCtx *app_ctx)
{
  if (!app_ctx) return FALSE;
  for (guint i = 0; i < app_ctx->config.num_source_sub_bins; i++)
    if (app_ctx->config.multi_source_config[i].type == NV_DS_SOURCE_RTSP)
      return TRUE;
  return FALSE;
}

static void
f5_drain_log_event (const gchar *event, gint64 mono_ns)
{
  const gchar *path = g_getenv ("MV3DT_SHUTDOWN_DRAIN_LOG");
  gchar *directory;
  FILE *out;
  if (!path || !*path) return;
  directory = g_path_get_dirname (path);
  g_mkdir_with_parents (directory, 0755);
  g_free (directory);
  out = fopen (path, "a");
  if (!out) {
    g_printerr ("F5 drain: cannot append drain evidence: %s\n", g_strerror (errno));
    return;
  }
  setvbuf (out, NULL, _IOLBF, 0);
  fprintf (out, "{\"event\":\"%s\",\"mono_ns\":%" G_GINT64_FORMAT
      ",\"shutdown_requested_ns\":%" G_GINT64_FORMAT
      ",\"source_admission_stopped_ns\":%" G_GINT64_FORMAT
      ",\"drain_completed_ns\":%" G_GINT64_FORMAT
      ",\"flush_start_ns\":%" G_GINT64_FORMAT
      ",\"destroy_pipeline_ns\":%" G_GINT64_FORMAT
      ",\"native_exit_ns\":%" G_GINT64_FORMAT "",
      event, mono_ns, f5_shutdown_requested_ns, f5_source_admission_stopped_ns,
      f5_drain_completed_ns, f5_flush_start_ns, f5_destroy_pipeline_ns,
      f5_native_exit_ns);
  if (f5_drain_mutex_initialized) {
    g_mutex_lock (&f5_drain_mutex);
    g_mutex_lock (&source_health_mutex);
    fputc (',', out);
    fputs ("\"sources\":[", out);
    for (guint i = 0; i < f5_drain_source_count; i++) {
      F5DrainSource *drain = &f5_drain_sources[i];
      SourceHealth *health = drain->source_id < MAX_SOURCE_BINS ?
          &source_health[drain->source_id] : NULL;
      gchar mapped[128] = "unknown";
      frame_audit_camera_id (drain->source_id, drain->source_id, mapped, sizeof (mapped));
      if (i) fputc (',', out);
      fprintf (out, "{\"camera_id\":\"%s\",\"source_id\":%u"
          ",\"blocked\":%s,\"idle_ns\":%" G_GINT64_FORMAT
          ",\"admission_buffers_since_barrier\":%" G_GUINT64_FORMAT
          ",\"last_admitted_pts\":%" G_GUINT64_FORMAT
          ",\"mux_frames\":%" G_GUINT64_FORMAT
          ",\"mux_last_frame\":%" G_GUINT64_FORMAT
          ",\"mux_last_pts\":%" G_GUINT64_FORMAT
          ",\"pgie_frames\":%" G_GUINT64_FORMAT
          ",\"pgie_last_frame\":%" G_GUINT64_FORMAT
          ",\"pgie_last_pts\":%" G_GUINT64_FORMAT
          ",\"tracker_frames\":%" G_GUINT64_FORMAT
          ",\"tracker_last_frame\":%" G_GUINT64_FORMAT
          ",\"tracker_last_pts\":%" G_GUINT64_FORMAT "}",
          mapped, drain->source_id, g_atomic_int_get (&drain->blocked) ? "true" : "false",
          drain->idle_ns, drain->admitted_buffers,
          drain->last_admitted_pts == GST_CLOCK_TIME_NONE ? G_GUINT64_CONSTANT (0) : drain->last_admitted_pts,
          health ? health->frames[SOURCE_HEALTH_MUX] : 0,
          health ? health->last_frame_num[SOURCE_HEALTH_MUX] : 0,
          health ? health->last_pts[SOURCE_HEALTH_MUX] : 0,
          health ? health->frames[SOURCE_HEALTH_PGIE] : 0,
          health ? health->last_frame_num[SOURCE_HEALTH_PGIE] : 0,
          health ? health->last_pts[SOURCE_HEALTH_PGIE] : 0,
          health ? health->frames[SOURCE_HEALTH_TRACKER] : 0,
          health ? health->last_frame_num[SOURCE_HEALTH_TRACKER] : 0,
          health ? health->last_pts[SOURCE_HEALTH_TRACKER] : 0);
    }
    fputs ("]", out);
    g_mutex_unlock (&source_health_mutex);
    g_mutex_unlock (&f5_drain_mutex);
  }
  fputs ("}\n", out);
  fclose (out);
}

static GstPadProbeReturn
f5_count_source_frame (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  F5DrainSource *source = (F5DrainSource *) user_data;
  GstBuffer *buffer;
  (void) pad;
  if (!info || !(buffer = GST_PAD_PROBE_INFO_BUFFER (info)))
    return GST_PAD_PROBE_OK;
  g_mutex_lock (&f5_drain_mutex);
  source->admitted_buffers++;
  source->last_admitted_pts = GST_BUFFER_PTS (buffer);
  g_mutex_unlock (&f5_drain_mutex);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
f5_idle_source_admission (GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
  F5DrainSource *source = (F5DrainSource *) user_data;
  (void) pad;
  (void) info;
  source->idle_ns = f5_monotonic_ns ();
  g_atomic_int_set (&source->blocked, 1);
  return GST_PAD_PROBE_OK; /* IDLE probe remains installed and keeps the pad idle. */
}

static gboolean
f5_pts_drained (F5DrainSource *source)
{
  SourceHealth *health = source->source_id < MAX_SOURCE_BINS ?
      &source_health[source->source_id] : NULL;
  return health && g_atomic_int_get (&source->blocked) &&
      source->last_admitted_pts != GST_CLOCK_TIME_NONE &&
      health->last_pts[SOURCE_HEALTH_MUX] == source->last_admitted_pts &&
      health->last_pts[SOURCE_HEALTH_PGIE] == source->last_admitted_pts &&
      health->last_pts[SOURCE_HEALTH_TRACKER] == source->last_admitted_pts &&
      health->frames[SOURCE_HEALTH_PGIE] == health->frames[SOURCE_HEALTH_TRACKER];
}

static gboolean
f5_drain_admitted_frames (AppCtx *app_ctx)
{
  guint timeout_sec = 10;
  const gchar *timeout_env = g_getenv ("MV3DT_SHUTDOWN_DRAIN_TIMEOUT_SEC");
  gchar *end = NULL;
  guint64 parsed;
  gint64 deadline;
  gboolean complete = FALSE;
  if (!f5_has_rtsp_source (app_ctx)) return TRUE;
  if (!f5_drain_mutex_initialized) {
    g_mutex_init (&f5_drain_mutex);
    f5_drain_mutex_initialized = TRUE;
  }
  if (timeout_env && *timeout_env) {
    parsed = g_ascii_strtoull (timeout_env, &end, 10);
    if (!end || *end || parsed < 1 || parsed > 30) {
      g_printerr ("F5 drain: timeout must be an integer in [1,30]\n");
      return FALSE;
    }
    timeout_sec = (guint) parsed;
  }
  if (f5_shutdown_requested_ns == 0)
    f5_shutdown_requested_ns = f5_monotonic_ns ();
  f5_drain_log_event ("shutdown_requested", f5_shutdown_requested_ns);
  deadline = g_get_monotonic_time () + ((gint64) timeout_sec * G_USEC_PER_SEC);
  f5_drain_source_count = 0;
  for (guint i = 0; i < app_ctx->pipeline.multi_src_bin.num_bins; i++) {
    F5DrainSource *source = &f5_drain_sources[f5_drain_source_count];
    NvDsSrcBin *bin = &app_ctx->pipeline.multi_src_bin.sub_bins[i];
    f5_drain_source_count++;
    source->source_id = bin->source_id;
    source->last_admitted_pts = GST_CLOCK_TIME_NONE;
    source->admitted_buffers = 0;
    source->blocked = 0;
    source->idle_ns = 0;
    source->pad = bin->bin ? gst_element_get_static_pad (bin->bin, "src") : NULL;
    if (!source->pad) {
      g_printerr ("F5 drain: cannot resolve source-bin output pad source=%u\n", source->source_id);
      return FALSE;
    }
    source->buffer_probe = gst_pad_add_probe (source->pad, GST_PAD_PROBE_TYPE_BUFFER,
        f5_count_source_frame, source, NULL);
    source->idle_probe = gst_pad_add_probe (source->pad, GST_PAD_PROBE_TYPE_IDLE,
        f5_idle_source_admission, source, NULL);
    if (!source->buffer_probe || !source->idle_probe) {
      g_printerr ("F5 drain: failed to install source admission barrier source=%u\n", source->source_id);
      return FALSE;
    }
  }
  while (g_get_monotonic_time () < deadline) {
    complete = TRUE;
    for (guint i = 0; i < f5_drain_source_count; i++)
      if (!g_atomic_int_get (&f5_drain_sources[i].blocked)) {
        complete = FALSE;
        break;
      }
    if (complete) break;
    g_usleep (1000);
  }
  if (!complete) {
    f5_drain_log_event ("source_admission_barrier_timeout_failed", f5_monotonic_ns ());
    return FALSE;
  }
  f5_source_admission_stopped_ns = f5_monotonic_ns ();
  f5_drain_log_event ("source_admission_stopped", f5_source_admission_stopped_ns);
  while (g_get_monotonic_time () < deadline) {
    complete = TRUE;
    g_mutex_lock (&f5_drain_mutex);
    g_mutex_lock (&source_health_mutex);
    for (guint i = 0; i < f5_drain_source_count; i++)
      if (!f5_pts_drained (&f5_drain_sources[i])) {
        complete = FALSE;
        break;
      }
    g_mutex_unlock (&source_health_mutex);
    g_mutex_unlock (&f5_drain_mutex);
    if (complete) break;
    g_usleep (1000); /* Observer cadence only; drain completion is frame/PTS based. */
  }
  if (complete) {
    f5_drain_success = TRUE;
    f5_drain_completed_ns = f5_monotonic_ns ();
    f5_drain_log_event ("drain_completed", f5_drain_completed_ns);
  } else {
    f5_drain_log_event ("drain_timeout_failed", f5_monotonic_ns ());
  }
  return complete;
}

static void
f5_release_drain_probes (void)
{
  for (guint i = 0; i < f5_drain_source_count; i++) {
    F5DrainSource *source = &f5_drain_sources[i];
    if (source->pad) {
      if (source->buffer_probe) gst_pad_remove_probe (source->pad, source->buffer_probe);
      if (source->idle_probe) gst_pad_remove_probe (source->pad, source->idle_probe);
      gst_object_unref (source->pad);
      source->pad = NULL;
      source->buffer_probe = source->idle_probe = 0;
    }
  }
}

'''


def drain_source(source: str) -> str:
    source = diagnostic_source(source)
    if source.count(ANCHOR) != 1:
        raise ValueError("expected one health constants insertion point")
    source = source.replace(ANCHOR, ANCHOR + C_DRAIN, 1)
    frame_num_anchor = "  guint64 last_frame_num[3];\n"
    if source.count(frame_num_anchor) != 1:
        raise ValueError("expected one per-stage frame counter declaration")
    source = source.replace(frame_num_anchor,
        frame_num_anchor + "  guint64 last_pts[3];\n", 1)
    update = "    health->last_frame_num[stage_index] = frame_meta->frame_num;\n"
    if source.count(update) != 1:
        raise ValueError("expected one health last-frame assignment")
    source = source.replace(update, update +
        "    health->last_pts[stage_index] = frame_meta->buf_pts;\n", 1)
    signal = "_intr_handler (int signum)\n{\n"
    if source.count(signal) != 1:
        raise ValueError("expected one interrupt handler entry")
    source = source.replace(signal, signal +
        "  f5_shutdown_requested_ns = f5_monotonic_ns ();\n", 1)
    if source.count(NATIVE_DESTROY) != 1:
        raise ValueError("expected one pipeline destroy call")
    source = source.replace(NATIVE_DESTROY, NATIVE_DESTROY_DRAIN, 1)
    release_anchor = "    destroy_pipeline (appCtx[i]);\n\n    if (appCtx[i]->ota_handler_thread"
    if source.count(release_anchor) != 1:
        raise ValueError("expected post-destroy probe cleanup insertion point")
    source = source.replace(release_anchor,
        "    destroy_pipeline (appCtx[i]);\n    f5_release_drain_probes ();\n\n    if (appCtx[i]->ota_handler_thread", 1)
    exit_anchor = "  gst_deinit ();\n\n  return return_value;"
    if source.count(exit_anchor) != 1:
        raise ValueError("expected native exit timestamp insertion point")
    source = source.replace(exit_anchor,
        "  gst_deinit ();\n  f5_native_exit_ns = f5_monotonic_ns ();\n"
        "  if (f5_drain_source_count > 0) f5_drain_log_event (\"native_exit\", f5_native_exit_ns);\n\n"
        "  return return_value;", 1)
    return source


def sdk_source(source: str) -> str:
    if source.count(EOS) != 1:
        raise ValueError("expected exact DeepStream 9.1 live EOS call")
    return source.replace(EOS, LIVE_CLOSE, 1)


def build(output: Path) -> dict:
    output = output.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    base_inputs = input_hashes(ROOT / "services/mv3dt_room/native")
    staged_native = output / "native"
    shutil.copytree(ROOT / "services/mv3dt_room/native", staged_native)
    main = staged_native / "deepstream_test5_app_main.c"
    main.write_text(drain_source(main.read_text()))
    source_inputs = input_hashes(staged_native)
    sdk_original = SDK_APP.read_text()
    staged_sdk = output / "deepstream_app.c"
    staged_sdk.write_text(sdk_source(sdk_original))
    binary_dir = output / "bin"
    binary_dir.mkdir()
    mounts = [(staged_native, "/workspace/room-native", True),
        (SDK / "src/apps/sample_apps/deepstream-test5", "/workspace/old-source", True),
        (SDK / "src", "/workspace/ds-src", True), (SDK / "includes", "/workspace/ds-includes", True),
        (staged_sdk, "/workspace/ds-src/apps/sample_apps/deepstream-app/deepstream_app.c", True),
        (ROOT / "scripts/dev_room_mv3dt/build_native_binary.sh", "/workspace/build-native.sh", True),
        (binary_dir, "/workspace/bin", False)]
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--user",
        f"{__import__('os').getuid()}:{__import__('os').getgid()}"]
    for local, target, read_only in mounts:
        command.extend(["-v", f"{local}:{target}" + (":ro" if read_only else "")])
    command.extend(["--entrypoint", "bash", IMAGE, "/workspace/build-native.sh"])
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if input_hashes(staged_native) != source_inputs or input_hashes(ROOT / "services/mv3dt_room/native") != base_inputs:
        raise RuntimeError("native source or frozen build inputs changed during candidate build")
    binary = binary_dir / BINARY
    build_id = re.search(r"Build ID: ([0-9a-f]+)", subprocess.check_output(["readelf", "-n", str(binary)], text=True))
    record = {"status": "EXPERIMENTAL_NOT_ACCEPTED", "candidate": "F5_DRAIN_BEFORE_FLUSH",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "base_source_sha256": base_inputs["native/deepstream_test5_app_main.c"],
        "staged_source_sha256": sha256(main), "binary": str(binary), "binary_sha256": sha256(binary),
        "elf_build_id": build_id.group(1), "toolchain_image": IMAGE,
        "base_native_inputs": base_inputs, "staged_native_inputs": source_inputs,
        "sdk_source_sha256": sha256(SDK_APP), "sdk_staged_sha256": sha256(staged_sdk),
        "sdk_delta": "RTSP configs skip injected EOS; URI replay remains original",
        "behavior": "source-bin src IDLE barriers; PTS match admitted source/mux/PGIE/tracker; timeout fails and withholds FLUSH_START",
        "drain_timeout_sec": 10, "production_accepted": False,
        "build_command": command}
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def drained(source: dict) -> bool:
    return (source.get("blocked") is True
        and source.get("last_admitted_pts") not in (None, 0)
        and source.get("last_admitted_pts") == source.get("mux_last_pts")
        and source.get("last_admitted_pts") == source.get("pgie_last_pts")
        and source.get("last_admitted_pts") == source.get("tracker_last_pts")
        and source.get("pgie_frames") == source.get("tracker_frames"))


def verify_build(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record.get("status") != "EXPERIMENTAL_NOT_ACCEPTED" or record.get("production_accepted"):
        raise ValueError("drain candidate must remain experimental")
    if input_hashes(ROOT / "services/mv3dt_room/native") != record["base_native_inputs"]:
        raise ValueError("current native source differs from candidate base")
    if input_hashes(path.parent / "native") != record["staged_native_inputs"]:
        raise ValueError("staged native source changed")
    if sha256(Path(record["binary"])) != record["binary_sha256"]:
        raise ValueError("candidate binary changed")
    if sha256(path.parent / "native/deepstream_test5_app_main.c") != record["staged_source_sha256"]:
        raise ValueError("candidate source hash mismatch")
    if sha256(SDK_APP) != record["sdk_source_sha256"] or sha256(path.parent / "deepstream_app.c") != record["sdk_staged_sha256"]:
        raise ValueError("DeepStream SDK staged/original source mismatch")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({key: value for key, value in build(args.output).items()
        if key in ("candidate", "revision", "staged_source_sha256", "binary_sha256", "elf_build_id")}))
