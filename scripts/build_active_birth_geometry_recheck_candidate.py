#!/usr/bin/env python3
"""Build an isolated r6b candidate with one bounded active-birth geometry recheck."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-verified-track-20261008-r6b-aux768-sparse20"
OUT = ROOT / ".runtime/sixcam-cam05-verified-track-20261009-active-birth-geometry-recheck-dedupe-r3"
SOURCE = ROOT / "services/camera_v11/native/six_camera_v13"
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"

PROTECTED = {
    "live_binary": ("bin/cam-six-yolo-local6-posttracker", "3f76f807a8401e35fa33a3811ef612f70af78f87dd51101580c4399c9e409a07"),
    "primary_engine": ("models/yolo26m_raw_otm_b6_fp16.engine", "e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"),
    "primary_parser": ("models/libyolo26rawperson.so", "d9d007b4eb0b80ab0c5aba552cc07ebf2e508d9b4ca649ef0e3bb156b292eb19"),
    "local_tracker": ("config/config_tracker_NvDCF_v13_local6.yml", "5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"),
}
V13 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml"
V13_SHA256 = "5d319ec9ce71ac2cef744c412ad55f91ee8bb64ee69c492901a1f6d0d98f9834"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}


def runtime_description(profile: dict) -> str:
    """Describe the effective detector profile, not inherited parent prose."""
    aux_input = profile.get("aux_input")
    aux_config = profile.get("aux_config")
    aux_interval = profile.get("aux_interval")
    if (not isinstance(aux_input, list) or len(aux_input) != 2
            or aux_input[0] != aux_input[1]
            or not isinstance(aux_interval, int)
            or not isinstance(aux_config, str)):
        raise RuntimeError("candidate manifest lacks a complete effective auxiliary profile")
    return (
        "DeepStream 9.1; frozen YOLO26m 640 FP16 primary batch-6 per frame; "
        f"YOLO26m {aux_input[0]} auxiliary batch-6 interval={aux_interval} "
        f"via {aux_config}; exact-PTS sidecar join; local NvDCF; "
        "async frozen-640 context verifier; no MV3DT/calibration"
    )


def compile_binary(name: str, *, direct_file: bool) -> None:
    # Historical feature-gate name retained for binary compatibility. The
    # effective auxiliary resolution/cadence is selected by the profile's
    # explicit V11_AUX_YOLO896_CONFIG runtime value and validated by the
    # candidate runner; this macro does not select 896-pixel inference.
    macros = "-DYOLO26_PERSON -DYOLO26_TRACKER -DYOLO26_AUXILIARY_896"
    if direct_file:
        macros += " -DPGIE_ISOLATION"
    command = (
        "g++ -std=c++17 -O2 " + macros
        + " -I/host-cuda-include -I/opt/nvidia/deepstream/deepstream/sources/includes"
        + " $(pkg-config --cflags gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " /src/main.cpp -o /work/bin/" + name
        + " $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)"
        + " -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta"
        + " -lnvbufsurface -lnvbufsurftransform -lnvds_infer -lnvinfer"
        + " -lnvinfer_plugin -L/usr/local/cuda/lib64 -lcudart -ldl -pthread"
        + " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
    )
    subprocess.run([
        "docker", "run", "--rm", "--pull=never", "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{SOURCE}:/src:ro", "-v", f"{OUT}:/work",
        "-v", "/usr/local/cuda/include:/host-cuda-include:ro", IMAGE,
        "bash", "-lc", command,
    ], cwd=ROOT, check=True)


def main() -> dict[str, str]:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    if sha256(V13) != V13_SHA256:
        raise RuntimeError("frozen V13 tracker hash changed")
    manifest_path = PARENT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("binary_sha256") != PROTECTED["live_binary"][1]:
        raise RuntimeError("r6b parent manifest does not name the expected binary")
    if manifest.get("deepstream_image") != IMAGE:
        raise RuntimeError("r6b parent is not pinned to the expected DeepStream image")
    for label, (relative, expected) in PROTECTED.items():
        if sha256(PARENT / relative) != expected:
            raise RuntimeError(f"r6b parent protected asset changed: {label}")

    shutil.copytree(PARENT, OUT)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True)

    out_manifest_path = OUT / "build-manifest.json"
    out_manifest = json.loads(out_manifest_path.read_text())
    live_sha = sha256(OUT / "bin/cam-six-yolo-local6-posttracker")
    direct_sha = sha256(OUT / "bin/cam-six-yolo-local6-direct-gatea")
    out_manifest["candidate_label"] = (
        "r6b sparse-768/interval-19 plus one exact-frame active-track verifier recheck "
        "and high-containment, bottom-aligned same-person proposal dedupe"
    )
    out_manifest["source_sha256"] = source_hashes()
    out_manifest["binary_sha256"] = live_sha
    out_manifest["direct_file_binary_sha256"] = direct_sha
    out_manifest["runtime"] = runtime_description(out_manifest["detector_profile"])
    out_manifest["build_provenance"] = {
        **out_manifest.get("build_provenance", {}),
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": PROTECTED["live_binary"][1],
        "change": (
            "a definitive active-track context-verifier negative is scoped to the checked "
            "detector geometry; permit one recheck for the same native ID only after at "
            "least five frames and detector-box IoU below 0.30, publish only on a "
            "positive exact-frame verifier result, and suppress only strict same-GIE "
            "nested duplicate proposals before NvDCF"
        ),
        "invariants": [
            "primary detector, parser, thresholds, inference cadence, and all non-duplicate NvDCF inputs unchanged",
            "no native track is published from an old verifier result or shadow geometry",
            "recheck is one-shot per native ID, same source, and requires current exact-frame PGIE association",
            "no geometry change is accepted without positive context verification",
            "only a unique, strongly nested duplicate proposal may be removed before NvDCF; ambiguous pairs fail open",
            "YOLO engine/parser/config and local NvDCF/V13/calibration assets are byte-identical to r6b",
        ],
    }
    out_manifest["active_track_verifier_geometry_recheck"] = {
        "enabled": True,
        "max_rechecks_per_native_id": 1,
        "minimum_delay_frames": 5,
        "current_detection_box_iou_vs_rejected_box_must_be_below": 0.30,
        "publication_requires": "positive exact-source-frame-and-PTS context-verifier result",
        "primary_nvdcf_input_changed": False,
    }
    out_manifest["same_component_nested_duplicate_policy"] = {
        "enabled": True,
        "placement": "after DeepStream per-GIE NMS and before NvDCF input",
        "base_gate": {
            "iou_min": 0.30,
            "smaller_box_containment_min": 0.95,
            "center_distance_over_max_height_max": 0.25,
            "width_ratio_max": 2.0,
            "height_ratio_max": 2.0,
            "confidence_difference_min": 0.05,
        },
        "strict_nested_extension": {
            "smaller_box_containment_min": 0.99,
            "center_distance_over_max_height_max": 0.28,
            "area_ratio_min": 1.25,
            "confidence_difference_min": 0.05,
            "close_confidence_tighter_box_preferred_below": 0.10,
        },
        "same_person_nested_extension": {
            "smaller_box_containment_min": 0.99,
            "center_distance_over_max_height_max": 0.35,
            "center_distance_over_max_width_max": 0.30,
            "bottom_edge_delta_over_max_height_max": 0.04,
            "width_ratio_max": 2.40,
            "height_ratio_max": 2.20,
            "area_ratio_min": 1.25,
            "area_ratio_max": 6.0,
            "confidence_difference_min": 0.05,
            "unique_match_required": True,
            "ambiguous_match_policy": "retain all candidates",
        },
        "ambiguous_match_policy": "retain all candidates",
        "detector_thresholds_nms_engine_parser_unchanged": True,
        "NvDCF_inputs_change_only_when_a_same_component_duplicate_is_suppressed": True,
    }
    out_manifest["recovery"] = {
        **out_manifest.get("recovery", {}),
        "candidate_binary_sha256": live_sha,
        "status": "REQUIRES_REVALIDATION_ON_EXACT_BINARY",
        "validated_cycles": 0,
    }
    out_manifest_path.write_text(json.dumps(out_manifest, indent=2) + "\n")
    return {
        "status": "BUILT_ISOLATED_CANDIDATE_REQUIRES_FULL_REGRESSION",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": live_sha,
        "direct_file_binary_sha256": direct_sha,
        "parent_binary_sha256": PROTECTED["live_binary"][1],
        "engine_sha256": PROTECTED["primary_engine"][1],
        "parser_sha256": PROTECTED["primary_parser"][1],
        "tracker_sha256": PROTECTED["local_tracker"][1],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
