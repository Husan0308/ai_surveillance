"""Build an isolated DeepStream candidate that logs public NvDCF shadow ages."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from scripts.build_room_candidate import IMAGE, ROOT, SDK, BINARY, input_hashes, diagnostic_source
from scripts.dev_room_mv3dt.verify_validated_assets import sha256

OUT = ROOT / ".runtime/sixcam-local6-20261004-b/shadow-age-diagnostic-build"

SHADOW_LOG_DECL = "static FILE *history_log;\n"
SHADOW_LOG_DECL_NEW = SHADOW_LOG_DECL + "static FILE *shadow_state_log;\n"
PROBE_END = "  if (post_log) fflush (post_log);\n  if (tracker_state_log) fflush (tracker_state_log);\n"
PROBE_END_NEW = '''  for (NvDsMetaList *u = batch_meta->batch_user_meta_list; u; u = u->next) {
    NvDsUserMeta *user = (NvDsUserMeta *) u->data;
    if (!user || user->base_meta.meta_type != NVDS_TRACKER_SHADOW_LIST_META ||
        !user->user_meta_data || !shadow_state_log) continue;
    NvDsTargetMiscDataBatch *shadow = (NvDsTargetMiscDataBatch *) user->user_meta_data;
    for (guint si = 0; si < shadow->numFilled; si++) {
      NvDsTargetMiscDataStream *stream = &shadow->list[si];
      for (guint ti = 0; ti < stream->numFilled; ti++) {
        NvDsTargetMiscDataObject *target = &stream->list[ti];
        for (guint oi = 0; target->list && oi < target->numObj; oi++) {
          NvDsTargetMiscDataFrame *sample = &target->list[oi];
          fprintf (shadow_state_log,
              "%s,%u,%u,%" G_GUINT64_FORMAT ",%u,%s,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f\\n",
              source_name (stream->streamID), stream->streamID, sample->frameNum,
              (guint64) target->uniqueId, sample->age,
              tracker_state_name (sample->trackerState), sample->confidence,
              sample->visibility, sample->tBbox.left, sample->tBbox.top,
              sample->tBbox.width, sample->tBbox.height);
        }
      }
    }
  }
  if (post_log) fflush (post_log);
  if (tracker_state_log) fflush (tracker_state_log);
  if (shadow_state_log) fflush (shadow_state_log);
'''
OPEN_LOG = '''  tracker_state_log = open_csv (directory, "tracker_state.csv",
      "camera_id,source_id,frame,label,native_track_id,nvdcf_state,track_age,tracker_confidence,left,top,width,height,visibility,world_foot_x,world_foot_y");
'''
OPEN_LOG_NEW = OPEN_LOG + '''  const char *shadow_path = g_getenv ("PN263_SHADOW_META_LOG");
  if (shadow_path && *shadow_path) {
    shadow_state_log = fopen (shadow_path, "w");
    if (shadow_state_log) {
      (void) setvbuf (shadow_state_log, NULL, _IOLBF, 0);
      fprintf (shadow_state_log, "camera_id,source_id,frame,native_track_id,age,state,tracker_confidence,visibility,left,top,width,height\\n");
    } else {
      g_printerr ("PN263_SHADOW ERROR cannot open %s\\n", shadow_path);
      return FALSE;
    }
  }
'''


def instrument_shadow(source: str) -> str:
    if source.count(SHADOW_LOG_DECL) != 1 or source.count(PROBE_END) != 1 or source.count(OPEN_LOG) != 1:
        raise ValueError("native shadow metadata instrumentation anchors changed")
    return (source.replace(SHADOW_LOG_DECL, SHADOW_LOG_DECL_NEW, 1)
            .replace(PROBE_END, PROBE_END_NEW, 1)
            .replace(OPEN_LOG, OPEN_LOG_NEW, 1))


def build() -> dict:
    output = OUT.resolve()
    output.relative_to(ROOT / ".runtime")
    output.mkdir(parents=True, exist_ok=False)
    current = ROOT / "services/mv3dt_room/native"
    base = input_hashes(current)
    native = output / "native"
    shutil.copytree(current, native)
    main = native / "deepstream_test5_app_main.c"
    main.write_text(diagnostic_source(main.read_text()))
    correction = native / "bbox_correction.c"
    correction.write_text(instrument_shadow(correction.read_text()))
    inputs = input_hashes(native)
    changed = [key for key in base if base[key] != inputs[key]]
    allowed = ["native/bbox_correction.c", "native/deepstream_test5_app_main.c"]
    if changed != allowed:
        raise RuntimeError(f"unexpected staged source delta: {changed}")
    binary_dir = output / "bin"
    binary_dir.mkdir()
    mounts = [(native, "/workspace/room-native", True),
        (SDK / "src/apps/sample_apps/deepstream-test5", "/workspace/old-source", True),
        (SDK / "src", "/workspace/ds-src", True), (SDK / "includes", "/workspace/ds-includes", True),
        (ROOT / "scripts/dev_room_mv3dt/build_native_binary.sh", "/workspace/build-native.sh", True),
        (binary_dir, "/workspace/bin", False)]
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--user",
               f"{os.getuid()}:{os.getgid()}"]
    for source_path, destination, readonly in mounts:
        command.extend(["-v", f"{source_path}:{destination}" + (":ro" if readonly else "")])
    command.extend(["--entrypoint", "bash", IMAGE, "/workspace/build-native.sh"])
    with (output / "build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if input_hashes(native) != inputs or input_hashes(current) != base:
        raise RuntimeError("source changed during diagnostic build")
    binary = binary_dir / BINARY
    build_id = re.search(r"Build ID: ([0-9a-f]+)", subprocess.check_output(["readelf", "-n", str(binary)], text=True))
    manifest = {
        "purpose": "diagnostic-only public NvDCF shadow state and age logging",
        "production_accepted": False,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "toolchain_image": IMAGE,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "elf_build_id": build_id.group(1),
        "base_inputs": base,
        "staged_inputs": inputs,
        "staged_source_delta": changed,
        "source_sha256": {key: inputs[key] for key in changed},
        "command": command,
        "logger": "NVDS_TRACKER_SHADOW_LIST_META -> age/state/confidence/visibility/bbox CSV, enabled only by PN263_SHADOW_META_LOG",
    }
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
