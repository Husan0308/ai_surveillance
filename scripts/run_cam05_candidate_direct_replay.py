#!/usr/bin/env python3
"""Run a six-input, file-only replay against the isolated CAM-05 cascade candidate."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
DEFAULT_CANDIDATE = ROOT / ".runtime/sixcam-cam05-cascade-20261007-r6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed {path.name}:{number}: {exc}") from exc
    return rows


def source_config(inputs: list[Path]) -> tuple[str, list[str]]:
    if len(inputs) == 1:
        mapping = [inputs[0]] * 6
    elif len(inputs) == 2:
        mapping = [inputs[0]] * 3 + [inputs[1]] * 3
    elif len(inputs) == 6:
        mapping = inputs
    else:
        raise ValueError("supply 1 file (repeated 6x), 2 files (3x each), or 6 files")
    lines: list[str] = []
    for index, path in enumerate(mapping, 1):
        uri = f"file:///workspace/inputs/{path.name}"
        lines.extend((f"[CAM-{index:02d}]", f"uri={base64.b64encode(uri.encode()).decode()}",
                      "username=", "password=", "latency_ms=0", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines), [str(path) for path in mapping]


def source_integrity(rows: list[dict], expected: int) -> dict:
    by_source = {source: [] for source in range(6)}
    for row in rows:
        source = int(row["source_id"])
        if source not in by_source:
            raise RuntimeError(f"unexpected direct-file source id {source}")
        by_source[source].append(row)
    result = {}
    for source, source_rows in by_source.items():
        local = [int(row["source_local_frame_num"]) for row in source_rows]
        sequences = [int(row["decoded_sequence"]) for row in source_rows]
        pts = [int(row["source_pts_ns"]) for row in source_rows]
        if (len(local) != expected or local != list(range(expected))
                or sequences != list(range(1, expected + 1))
                or len(set(pts)) != expected or any(b <= a for a, b in zip(pts, pts[1:]))):
            raise RuntimeError(f"source {source}: replay did not produce unique monotonic 0..N-1 frames")
        result[str(source)] = {
            "frames": len(local), "local_first": local[0], "local_last": local[-1],
            "unique_frames": len(set(local)), "first_pts_ns": pts[0], "last_pts_ns": pts[-1],
        }
    return result


def native_runtime_budget_seconds(input_metadata: list[dict], margin_seconds: int = 30) -> int:
    """Allow the direct-file runner to drain EOS after the longest input."""
    if not input_metadata or margin_seconds < 0:
        raise ValueError("runtime budget requires inputs and a non-negative margin")
    durations = [float(row["duration_sec"]) for row in input_metadata]
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("direct-file input duration must be positive")
    return max(120, int(max(durations) + 0.999999) + margin_seconds)


def verifier_runtime_contract(manifest: dict, verifier_config: str) -> tuple[str, int]:
    """Return the staged verifier identity and exact runtime batch setting."""
    if verifier_config == "peoplenet_lowconf_verifier.txt":
        # The PeopleNet profile is batch-1 and must not inherit the selected
        # candidate's default verifier contract.
        return "PeopleNet-2.6.3", 1
    if verifier_config == "yolo26m_context_verifier.txt":
        return "YOLO26m-640-FP16", 1
    return "YOLO26m-640-FP16", int(
        manifest.get("yolo_context_verifier", {}).get("runtime_batch_size", 1))


def resolve_verifier_config(manifest: dict, requested: str | None) -> str:
    """Use the selected candidate's manifested verifier unless explicitly overridden."""
    config = requested or (manifest.get("yolo_context_verifier", {}).get("config"))
    if config is None:
        config = "yolo26m_context_verifier.txt"
    if not isinstance(config, str) or Path(config).name != config:
        raise ValueError("verifier config must be a manifested plain filename")
    return config


def resolve_verifier_batch_size(configured: int, override: int | None,
                                variant: dict | None) -> int:
    """Allow only a diagnostic batch inside the staged engine's declared profile."""
    if override is None:
        return configured
    if override < 1:
        raise ValueError("diagnostic verifier batch size must be positive")
    if variant is None:
        if override != configured:
            raise ValueError("batch override requires a manifest-declared dynamic verifier profile")
        return configured
    profile = variant.get("batch", {})
    minimum = int(profile.get("engine_min", configured))
    maximum = int(profile.get("engine_max", configured))
    if not minimum <= override <= maximum:
        raise ValueError("diagnostic verifier batch override is outside the TensorRT engine profile")
    return override


def validate_stream_copy_provenance(source: dict, remux: dict,
                                    source_packet_sha256: str,
                                    remux_packet_sha256: str) -> None:
    """Require unchanged compressed video packets and frame geometry after PTS normalization."""
    for key in ("width", "height", "fps", "frames"):
        if source.get(key) != remux.get(key):
            raise RuntimeError(f"timestamp remux changed input {key}: {source.get(key)} != {remux.get(key)}")
    if not source_packet_sha256 or source_packet_sha256 != remux_packet_sha256:
        raise RuntimeError("timestamp remux changed compressed video packet payloads")


def ffprobe_video(path: Path) -> dict:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,start_time",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ], check=True, text=True, capture_output=True)
    data = json.loads(probe.stdout)
    stream = data["streams"][0]
    return {
        "codec": stream["codec_name"], "width": int(stream["width"]),
        "height": int(stream["height"]), "fps": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]), "start_time": float(stream.get("start_time", 0.0)),
        "duration_sec": float(data["format"]["duration"]),
    }


def video_packet_payload_sha256(path: Path) -> str:
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-hash", "sha256", "-",
    ], check=True, text=True, capture_output=True)
    return result.stdout.strip().split("=", 1)[-1]


def stage_pts_normalized_stream_copy(source: Path, target: Path) -> dict:
    """Reset MP4 timeline origin without re-encoding or changing video packets."""
    before = ffprobe_video(source)
    before_packets = video_packet_payload_sha256(source)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-map", "0:v:0", "-an", "-c:v", "copy", "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(target),
    ], check=True, cwd=ROOT)
    after = ffprobe_video(target)
    after_packets = video_packet_payload_sha256(target)
    validate_stream_copy_provenance(before, after, before_packets, after_packets)
    target.chmod(0o444)
    return {
        "source_sha256": sha256(source), "source_start_time_sec": before["start_time"],
        "staged_sha256": sha256(target), "staged_start_time_sec": after["start_time"],
        "video_packet_payload_sha256": before_packets,
        "frame_payloads_unchanged": True, "reencoded": False,
        "frames": after["frames"], "width": after["width"], "height": after["height"],
        "fps": after["fps"], "duration_sec": after["duration_sec"],
    }


def auxiliary_detector_environment(enabled: bool, profile: dict | None = None) -> list[str]:
    """Return only the explicitly selected auxiliary branch environment.

    The isolated CAM-05 verifier experiment can compare the unchanged 640
    primary path with/without the explicitly manifested supplemental detector.
    """
    profile = profile or {
        "aux_config": "config_infer_aux_yolo26m_896.txt",
        "metamux_config": "config_metamux_yolo26m_aux896.txt",
        "aux_input": [896, 896], "aux_unique_id": 3,
        "aux_engine_sha256": "legacy", "aux_parser_sha256": "legacy",
    }
    if enabled:
        from scripts.run_six_camera_v13_wall_candidate import auxiliary_detector_environment as render
        return render(profile)
    if profile.get("aux_environment_contract") == "generic-v1":
        return ["V11_AUX_YOLO_ENABLE=0"]
    return ["V11_AUX_YOLO896_ENABLE=0"]


def tracker_nan_diagnostic_environment(enabled: bool) -> list[str]:
    """Opt into source/frame-addressed shadow diagnostics without changing defaults."""
    return ["V11_TRACKER_NAN_DIAGNOSTICS=1"] if enabled else []


def tracker_frame_state_diagnostic_environment(enabled: bool) -> list[str]:
    """Opt into a per-source/frame public NvDCF state dump for isolated replay."""
    return ["V11_TRACKER_FRAME_STATE_DIAGNOSTICS=1"] if enabled else []


def candidate_centered_crop_environment(enabled: bool) -> list[str]:
    """Opt into the diagnostic candidate-centered context crop policy."""
    return ["V11_CONTEXT_CANDIDATE_CROP=1"] if enabled else []


def run(candidate: Path, inputs: list[Path], output: Path, timeout_sec: int,
        verifier_config: str | None = None,
        dump_verifier_camera: int | None = None,
        dump_verifier_frame: int | None = None,
        dump_verifier_native_id: int | None = None,
        enable_auxiliary: bool = True,
        diagnostic_verifier_batch_size: int | None = None,
        realtime_pace: bool = False,
        tracker_nan_diagnostics: bool = False,
        tracker_frame_state_diagnostics: bool = False,
        shadow_context_refresh_diagnostic: bool = False,
        candidate_centered_crop: bool = False,
        isolation_sleep_per_frame_us: int = 0,
        runtime_budget_override_sec: int | None = None) -> dict:
    from scripts.run_six_camera_v13_wall_candidate import validate_candidate, validate_yolo_context_verifier

    candidate, output = candidate.resolve(), output.resolve()
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate and output must be under .runtime")
    manifest = validate_candidate(candidate)
    validate_yolo_context_verifier(candidate, manifest)
    verifier_config = resolve_verifier_config(manifest, verifier_config)
    verifier_name, verifier_batch_size = verifier_runtime_contract(manifest, verifier_config)
    variant = None
    if verifier_config not in ("yolo26m_context_verifier.txt", "peoplenet_lowconf_verifier.txt"):
        variant = manifest.get("yolo_context_verifier_variants", {}).get(verifier_config)
        if not variant and verifier_config == "yolo26m_context_verifier_512.txt":
            # r44 staged and hashed the 512 TensorRT/ONNX/parser artifacts as a
            # diagnostic-only shape, but its selected-candidate manifest only
            # promoted 448 into the variant map. Reconstruct this exact staged
            # variant from those recorded hashes; do not accept an unmanifested
            # or altered artifact.
            model_hashes = manifest.get("model_sha256", {})
            variant = {
                "config": verifier_config,
                "config_sha256": manifest.get("config_sha256", {}).get(verifier_config),
                "engine": "yolo26m_context_verifier_512_b2_fp16.engine",
                "engine_sha256": model_hashes.get("yolo26m_context_verifier_512_b2_fp16.engine"),
                "parser": "libyolo26rawperson_context_dynamic.so",
                "parser_sha256": model_hashes.get("libyolo26rawperson_context_dynamic.so"),
                "onnx": "yolo26m_context_512.onnx",
                "onnx_sha256": model_hashes.get("yolo26m_context_512.onnx"),
                "input": [3, 512, 512], "threshold": 0.20, "nms_iou": 0.45,
                "batch": {"runtime": 2},
            }
        config_path = candidate / "config" / verifier_config
        if not variant or not config_path.is_file() or sha256(config_path) != variant.get("config_sha256"):
            raise ValueError("unsupported or unproven diagnostic verifier config")
        engine_path = candidate / "models" / variant["engine"]
        parser_path = candidate / "models" / variant["parser"]
        onnx_path = candidate / "models" / variant["onnx"]
        text = config_path.read_text()
        verifier_batch_size = int(variant["batch"]["runtime"])
        required_variant = (
            f"batch-size={verifier_batch_size}", f"infer-dims=3;{variant['input'][1]};{variant['input'][2]}",
            f"model-engine-file=/models/{variant['engine']}",
            f"custom-lib-path=/models/{variant['parser']}",
            "parse-bbox-func-name=NvDsInferParseYolo26RawPerson",
            f"pre-cluster-threshold={variant['threshold']:.2f}",
            f"nms-iou-threshold={variant['nms_iou']:.2f}",
            "maintain-aspect-ratio=1", "symmetric-padding=1")
        if (any(text.count(field) != 1 for field in required_variant)
                or not engine_path.is_file() or sha256(engine_path) != variant.get("engine_sha256")
                or not parser_path.is_file() or sha256(parser_path) != variant.get("parser_sha256")
                or not onnx_path.is_file() or sha256(onnx_path) != variant.get("onnx_sha256")
                or manifest.get("model_sha256", {}).get(variant["engine"]) != variant.get("engine_sha256")):
            raise RuntimeError("diagnostic verifier variant artifact/config provenance mismatch")
        verifier_name = f"YOLO26m-{variant['input'][1]}-FP16"
    configured_verifier_batch_size = verifier_batch_size
    verifier_batch_size = resolve_verifier_batch_size(
        verifier_batch_size, diagnostic_verifier_batch_size, variant)
    inputs = [path.resolve() for path in inputs]
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("a replay input file does not exist")
    input_metadata = []
    for path in inputs:
        media = ffprobe_video(path)
        input_metadata.append({"path": str(path), "sha256": sha256(path), **media})
    frame_counts = {entry["frames"] for entry in input_metadata}
    if len(frame_counts) != 1:
        raise ValueError("all supplied files must have identical frame counts for the six-input diagnostic")
    expected_frames = frame_counts.pop()
    runtime_budget_sec = native_runtime_budget_seconds(input_metadata)
    if not 0 <= isolation_sleep_per_frame_us <= 1_000_000:
        raise ValueError("isolation sleep must be in the diagnostic range 0..1000000 us")
    if realtime_pace and isolation_sleep_per_frame_us:
        raise ValueError("choose either source-PTS pacing or fixed per-frame pacing")
    if runtime_budget_override_sec is not None:
        minimum_runtime = int(max(row["duration_sec"] for row in input_metadata) + 0.999999) + 30
        if runtime_budget_override_sec < minimum_runtime:
            raise ValueError("runtime budget must exceed the longest source duration plus 30s EOS margin")
        runtime_budget_sec = runtime_budget_override_sec
    output.mkdir(parents=True, exist_ok=False)
    image_dir = output / "input-mount"
    image_dir.mkdir()
    effective_verifier_config = f"/config/{verifier_config}"
    effective_verifier_config_path = candidate / "config" / verifier_config
    if diagnostic_verifier_batch_size is not None and diagnostic_verifier_batch_size != configured_verifier_batch_size:
        original_config = effective_verifier_config_path.read_text()
        original_batch_line = f"batch-size={configured_verifier_batch_size}"
        if original_config.count(original_batch_line) != 1:
            raise RuntimeError("diagnostic config staging requires one exact configured batch-size line")
        diagnostic_config = output / "verifier-config-diagnostic.txt"
        diagnostic_config.write_text(original_config.replace(
            original_batch_line, f"batch-size={diagnostic_verifier_batch_size}", 1))
        diagnostic_config.chmod(0o444)
        effective_verifier_config = "/work/verifier-config-diagnostic.txt"
        effective_verifier_config_path = diagnostic_config
    staged_by_source: dict[Path, Path] = {}
    staged_metadata: dict[str, dict] = {}
    for index, path in enumerate(dict.fromkeys(inputs)):
        # Some extracted MP4 segments retain a large parent-recording PTS
        # origin. GstQTDemux may reject those edit lists even when ffprobe can
        # read them. Normalize only the container timeline, then prove packet
        # payload identity; no decode/re-encode or frame selection is allowed.
        staged_input = image_dir / f"input{index:02d}-{path.name}"
        staged_metadata[str(path)] = stage_pts_normalized_stream_copy(path, staged_input)
        staged_by_source[path] = staged_input
    staged_inputs = [staged_by_source[path] for path in inputs]
    sources, _ = source_config(staged_inputs)
    _, mapping = source_config(inputs)
    source_file = output / "sources.ini"
    source_file.write_text(sources)
    source_file.chmod(0o600)

    command = [
        "docker", "run", "--rm", "--pull=never", "--init", "--gpus", "all", "--network=none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", "PGIE_ISOLATION_SOURCE_COUNT=6", "-e", "PGIE_ISOLATION_MUX_BATCH=6",
        "-e", "PGIE_ISOLATION_MUX_WIDTH=1920", "-e", "PGIE_ISOLATION_MUX_HEIGHT=1080",
        "-e", "PGIE_ISOLATION_MUX_MEMORY_TYPE=0", "-e", "PGIE_ISOLATION_LIVE_SOURCE=0",
        "-e", "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US=-1",
        "-e", "PGIE_ISOLATION_CONFIG=/config/config_infer_primary_yolo26m_raw_otm.txt",
        "-e", "PGIE_ISOLATION_MIN_CONFIDENCE=0.23",
        "-e", "V11_TRACKER_WIDTH=1920", "-e", "V11_TRACKER_HEIGHT=1080",
        "-e", "V11_VERBOSE_EVIDENCE=1", "-e", "V11_SOURCE_STALL_RECOVERY=0",
        "-e", "V11_LOWCONF_VERIFIER_ENABLE=1", "-e", "V11_ACTIVE_TRACK_VERIFIER_ENABLE=1",
        "-e", "V11_ACTIVE_TRACK_VERIFIER_FAIL_CLOSED=1",
        "-e", "V11_LOWCONF_PRIMARY_CONFIG=/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
        "-e", f"V11_LOWCONF_VERIFIER_CONFIG={effective_verifier_config}",
        "-e", f"V11_CONTEXT_VERIFIER_NAME={verifier_name}",
        "-e", f"V11_CONTEXT_VERIFIER_BATCH_SIZE={verifier_batch_size}",
        "-e", "V11_CONTEXT_VERIFIER_EVIDENCE=1",
        *[value for item in auxiliary_detector_environment(
            enable_auxiliary, manifest.get("detector_profile", {}))
          for value in ("-e", item)],
        "-e", "V11_TRACKER_DECISION_DIAGNOSTICS=1",
        "--mount", f"type=bind,src={candidate / 'bin'},dst=/candidate,readonly",
        "--mount", f"type=bind,src={candidate / 'config'},dst=/config,readonly",
        "--mount", f"type=bind,src={candidate / 'models'},dst=/models,readonly",
        "--mount", f"type=bind,src={image_dir},dst=/workspace/inputs,readonly",
        "--mount", f"type=bind,src={output},dst=/work",
        IMAGE, "/candidate/cam-six-yolo-local6-direct-gatea", "/work/sources.ini",
        str(runtime_budget_sec),
    ]
    if isolation_sleep_per_frame_us:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", f"PGIE_ISOLATION_SLEEP_PER_FRAME_US={isolation_sleep_per_frame_us}"]
    if tracker_nan_diagnostics:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", *tracker_nan_diagnostic_environment(True)]
    if tracker_frame_state_diagnostics:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", *tracker_frame_state_diagnostic_environment(True)]
    if manifest.get("shadow_context_refresh_enabled") is True or shadow_context_refresh_diagnostic:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_SHADOW_CONTEXT_REFRESH_ENABLE=1"]
    if candidate_centered_crop:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", *candidate_centered_crop_environment(True)]
    if manifest.get("gpu_context_crop") is True:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_CONTEXT_VERIFIER_GPU_CROP=1"]
    if manifest.get("native_trt_runtime_backend") is True:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_CONTEXT_VERIFIER_BACKEND=trt-runtime",
            "-e", "V11_CONTEXT_VERIFIER_TRACE=1"]
    elif manifest.get("native_context_backend") is True:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_CONTEXT_VERIFIER_BACKEND=nvdsinfer-context",
            "-e", "V11_CONTEXT_VERIFIER_TRACE=1"]
    if manifest.get("per_camera_context_backend") is True:
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_CONTEXT_VERIFIER_PER_CAMERA=1"]
    if manifest.get("native_preprocessed_context_backend") is True:
        preprocessor = manifest.get("context_preprocessor", {})
        library = candidate / "lib" / Path(str(preprocessor.get("library", ""))).name
        if (not library.is_file() or sha256(library) != preprocessor.get("library_sha256")
                or not str(preprocessor.get("library", "")).startswith("/contextlib/")):
            raise RuntimeError("preprocessed-context library is missing or differs from candidate manifest")
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "V11_CONTEXT_VERIFIER_PREPROCESSED=1",
            "-e", f"V11_CONTEXT_PREPROCESS_LIBRARY={preprocessor['library']}",
            "--mount", f"type=bind,src={candidate / 'lib'},dst=/contextlib,readonly",
        ]
    if realtime_pace:
        command[command.index("--mount"):command.index("--mount")] = [
            "-e", "PGIE_ISOLATION_REALTIME_PACE=1"]
    if dump_verifier_camera is not None:
        if dump_verifier_camera not in range(1, 7):
            raise ValueError("verifier crop dump camera must be 1..6")
        command[command.index(IMAGE):command.index(IMAGE)] = [
            "-e", "V11_CONTEXT_VERIFIER_DUMP_DIR=/work/verifier-crops",
            "-e", f"V11_CONTEXT_VERIFIER_DUMP_CAMERA={dump_verifier_camera}",
        ]
        if dump_verifier_frame is not None:
            command[command.index(IMAGE):command.index(IMAGE)] = [
                "-e", f"V11_CONTEXT_VERIFIER_DUMP_FRAME={dump_verifier_frame}"]
        if dump_verifier_native_id is not None:
            command[command.index(IMAGE):command.index(IMAGE)] = [
                "-e", f"V11_CONTEXT_VERIFIER_DUMP_NATIVE_ID={dump_verifier_native_id}"]
    (output / "command-redacted.json").write_text(json.dumps({
        "candidate": str(candidate), "candidate_binary_sha256": manifest["direct_file_binary_sha256"],
        "command": ["docker", "run", "--gpus all", "--network=none", "<six direct-file sources>",
                    "DeepStream 9.1", "640 primary + local NvDCF",
                    f"{manifest['detector_profile'].get('aux_input', [896])[0]} auxiliary enabled"
                    if enable_auxiliary else "auxiliary disabled"],
        "input_metadata": input_metadata, "source_mapping": mapping,
        "staged_stream_copies": staged_metadata,
        "mux": {"width": 1920, "height": 1080, "batch_size": 6, "live_source": 0,
                "timeout_us": -1, "loop": False},
        "primary": {"engine_sha256": manifest["detector_profile"]["primary_engine_sha256"],
                    "config_sha256": manifest["config_sha256"]["config_infer_primary_yolo26m_raw_otm_lowconf023.txt"],
                    "threshold": 0.23, "normal_nvdcf_floor": 0.25},
        "auxiliary": {"enabled": enable_auxiliary,
                       "input": manifest["detector_profile"].get("aux_input") if enable_auxiliary else None,
                       "config": manifest["detector_profile"].get("aux_config") if enable_auxiliary else None,
                       "engine_sha256": manifest["detector_profile"]["aux_engine_sha256"] if enable_auxiliary else None,
                       "interval": manifest["detector_profile"]["aux_interval"] if enable_auxiliary else None,
                       "pts_tolerance_us": 0 if enable_auxiliary else None},
        "active_track_verifier_config": verifier_config,
        "active_track_verifier_batch_size": verifier_batch_size,
        "configured_verifier_batch_size": configured_verifier_batch_size,
        "diagnostic_verifier_batch_override": diagnostic_verifier_batch_size,
        "effective_verifier_config_path": str(effective_verifier_config_path),
        "effective_verifier_config_sha256": sha256(effective_verifier_config_path),
        "shadow_context_refresh_enabled": manifest.get("shadow_context_refresh_enabled", False),
        "shadow_context_refresh_diagnostic_override": shadow_context_refresh_diagnostic,
        "candidate_centered_crop": candidate_centered_crop,
        "native_runtime_budget_sec": runtime_budget_sec,
        "realtime_paced_file_sources": realtime_pace,
        "diagnostic_sleep_per_frame_us": isolation_sleep_per_frame_us,
    }, indent=2) + "\n")
    started = time.monotonic()
    with (output / "native.log").open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=timeout_sec, check=False)
    elapsed = time.monotonic() - started
    native_log = (output / "native.log").read_text(errors="replace")
    if (completed.returncode != 0 or "PGIE_ISOLATION EOS all file sources drained" not in native_log
            or "GROUP END" not in native_log or "fatal=0" not in native_log):
        result = {"status": "FAIL", "return_code": completed.returncode, "elapsed_sec": elapsed,
                  "reason": "candidate direct-file pipeline did not complete clean EOS without fatal errors"}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"candidate replay failed; inspect {output / 'native.log'}")

    source_rows = read_jsonl(output / "source-frames.jsonl")
    source_audit = source_integrity(source_rows, expected_frames)
    frames = read_jsonl(output / "frames.jsonl")
    detections = read_jsonl(output / "detections.jsonl")
    canonical = read_jsonl(output / "canonical-objects.jsonl")
    lowconf = read_jsonl(output / "lowconf-candidates.jsonl")
    verifier = read_jsonl(output / "lowconf-verifier.jsonl")
    per_source = {}
    for source in range(6):
        camera_id = f"CAM-{source + 1:02d}"
        det_rows = [row for row in detections if int(row["source_id"]) == source]
        canon_rows = [row for row in canonical if row.get("camera_id") == camera_id]
        verify_rows = [row for row in verifier if row.get("camera_id") == camera_id]
        per_source[str(source)] = {
            "camera_id": camera_id,
            "pgie_frames": sum(1 for row in frames if int(row["source_id"]) == source),
            "primary_ge_025": sum(1 for row in det_rows if int(row.get("component_id", 1)) == 1 and float(row["confidence"]) >= .25),
            "primary_023_to_025": sum(1 for row in lowconf if row.get("camera_id") == camera_id),
            "auxiliary_proposals": sum(1 for row in det_rows if int(row.get("component_id", 1)) == 3),
            "canonical_object_rows": len(canon_rows),
            "canonical_ids": sorted({int(obj["native_track_id"])
                                      for row in canon_rows for obj in row.get("objects", [])}),
            "active_birth_verifier": len(verify_rows),
            "active_birth_verifier_status": {
                status: sum(1 for row in verify_rows if row.get("status") == status)
                for status in sorted({row.get("status", "UNKNOWN") for row in verify_rows})
            },
        }
    result = {
        "status": "PASS_DIAGNOSTIC_REPLAY",
        "diagnostic_only": True,
        "candidate_binary_sha256": manifest["direct_file_binary_sha256"],
        "auxiliary_enabled": enable_auxiliary,
        "active_track_verifier_config": verifier_config,
        "active_track_verifier_batch_size": verifier_batch_size,
        "configured_verifier_batch_size": configured_verifier_batch_size,
        "diagnostic_verifier_batch_override": diagnostic_verifier_batch_size,
        "effective_verifier_config_path": str(effective_verifier_config_path),
        "effective_verifier_config_sha256": sha256(effective_verifier_config_path),
        "native_source_sha256": manifest["source_sha256"],
        "inputs": input_metadata,
        "source_mapping": mapping,
        "staged_stream_copies": staged_metadata,
        "unique_source_frame_integrity": source_audit,
        "expected_frames_per_source": expected_frames,
        "native_runtime_budget_sec": runtime_budget_sec,
        "observed": {"frame_rows": len(frames), "detector_rows": len(detections),
                     "canonical_rows": len(canonical), "lowconf_candidate_rows": len(lowconf),
                     "verifier_rows": len(verifier)},
        "per_source": per_source,
        "eos": True,
        "rtsp_connections": 0,
        "elapsed_sec": elapsed,
        "diagnostic_sleep_per_frame_us": isolation_sleep_per_frame_us,
        "primary_input_note": "0.23-0.25 proposals were copied to verifier sidecar and removed before NvDCF; >=0.25 primary detections retained original box/confidence",
        "auxiliary_note": ("supplemental proposals were merged only at exact same-source PTS; auxiliary-born publication requires the frozen-640 context verifier"
                           if enable_auxiliary else "auxiliary branch disabled; only frozen 640 primary proposals enter NvDCF"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="repeat once, twice, or six times; one input is duplicated as a diagnostic source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--verifier-config",
                        help="diagnostic verifier config declared in candidate build-manifest.json")
    parser.add_argument("--dump-verifier-camera", type=int, help="opt-in crop dump source ID (1..6)")
    parser.add_argument("--dump-verifier-frame", type=int, help="optional source-local frame filter for crop dump")
    parser.add_argument("--dump-verifier-native-id", type=int, help="optional native target ID filter for crop dump")
    parser.add_argument("--disable-auxiliary", action="store_true",
                        help="diagnostic A/B only: disable the supplemental branch while preserving 640 primary and verifier")
    parser.add_argument("--diagnostic-verifier-batch-size", type=int,
                        help="diagnostic-only batch override within the selected engine profile")
    parser.add_argument("--realtime-pace", action="store_true",
                        help="pace direct-file sources at their recorded rate for live-equivalent latency testing")
    parser.add_argument("--isolation-sleep-per-frame-us", type=int, default=0,
                        help="diagnostic-only fixed sleep on each decoded file frame to limit GPU load")
    parser.add_argument("--runtime-budget-sec", type=int,
                        help="diagnostic-only wall-clock cap; must exceed source duration plus EOS margin")
    parser.add_argument("--tracker-nan-diagnostics", action="store_true",
                        help="emit source/frame-addressed invalid shadow geometry rows; diagnostic only")
    parser.add_argument("--tracker-frame-state-diagnostics", action="store_true",
                        help="emit per-frame active/shadow NvDCF metadata; diagnostic only")
    parser.add_argument("--shadow-context-refresh-diagnostic", action="store_true",
                        help="opt in to exact-frame shadow-context refresh for this diagnostic replay only")
    parser.add_argument("--candidate-centered-crop", action="store_true",
                        help="diagnostic only: use a tight candidate-centered verifier crop")
    args = parser.parse_args()
    run(args.candidate, args.input, args.output, args.timeout_sec, args.verifier_config,
        args.dump_verifier_camera, args.dump_verifier_frame, args.dump_verifier_native_id,
        enable_auxiliary=not args.disable_auxiliary,
        diagnostic_verifier_batch_size=args.diagnostic_verifier_batch_size,
        realtime_pace=args.realtime_pace,
        tracker_nan_diagnostics=args.tracker_nan_diagnostics,
        tracker_frame_state_diagnostics=args.tracker_frame_state_diagnostics,
        shadow_context_refresh_diagnostic=args.shadow_context_refresh_diagnostic,
        candidate_centered_crop=args.candidate_centered_crop,
        isolation_sleep_per_frame_us=args.isolation_sleep_per_frame_us,
        runtime_budget_override_sec=args.runtime_budget_sec)


if __name__ == "__main__":
    main()
