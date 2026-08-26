from __future__ import annotations

from pathlib import Path

from . import person_tracking_native_deepstream as native


def _section_header(line: str) -> str | None:
    if not line or line[0].isspace():
        return None
    body = line.split("#", 1)[0].strip()
    if body.endswith(":") and body[:-1].strip():
        return body[:-1].strip()
    return None


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = None
    for index, line in enumerate(lines):
        if _section_header(line) == section:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _section_header(lines[index]) is not None:
            end = index
            break
    return start, end


def _set_section_key(lines: list[str], section: str, key: str, value: str) -> None:
    bounds = _section_bounds(lines, section)
    if bounds is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{section}:")
        start, end = len(lines) - 1, len(lines)
    else:
        start, end = bounds

    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if not stripped.startswith(key + ":"):
            continue
        indent = lines[index][: len(lines[index]) - len(stripped)] or "  "
        comment = ""
        if "#" in stripped:
            comment = "  #" + stripped.split("#", 1)[1]
        lines[index] = f"{indent}{key}: {value}{comment}"
        return

    lines.insert(end, f"  {key}: {value}")


def _native_tracker_config_v2(stock: Path) -> Path:
    lines = Path(stock).read_text(encoding="utf-8").splitlines()

    # These keys are present in NVIDIA's NvDCF perf profile and are tuned only
    # for sparse primary-GIE refresh. Missing required lifecycle keys indicate an
    # unexpected/incompatible tracker config and remain fatal.
    native._rewrite_key(lines, "minDetectorConfidence", "0.10")
    native._rewrite_key(lines, "minTrackerConfidence", "0.12")
    native._rewrite_key(lines, "probationAge", "0")
    native._rewrite_key(lines, "maxShadowTrackingAge", "80")
    native._rewrite_key(lines, "earlyTerminationAge", "6")

    # NVIDIA documents outputShadowTracks as an optional TargetManagement key;
    # config_tracker_NvDCF_perf.yml may omit it. Add it in the correct section
    # instead of treating absence as a malformed stock config.
    _set_section_key(lines, "TargetManagement", "outputShadowTracks", "1")

    # Keep this baseline camera-local; these settings are optional across stock
    # DeepStream tracker profiles, so absence is not an error.
    native._rewrite_key(lines, "enableReAssoc", "0", required=False)
    native._rewrite_key(lines, "reidType", "0", required=False)
    native._rewrite_key(lines, "outputReidTensor", "0", required=False)

    native.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    native.TRACKER_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Fail fast if the generated deployment config does not contain the policy
    # we intend the low-level NvDCF library to consume.
    generated = native.TRACKER_CONFIG.read_text(encoding="utf-8")
    required = (
        "probationAge: 0",
        "maxShadowTrackingAge: 80",
        "earlyTerminationAge: 6",
        "outputShadowTracks: 1",
    )
    missing = [item for item in required if item not in generated]
    if missing:
        raise RuntimeError(
            "Native NvDCF generated config verification failed: " + ", ".join(missing)
        )

    print(
        "CAMERA_NATIVE_NVDCF_CONFIG "
        "probationAge=0 maxShadowTrackingAge=80 earlyTerminationAge=6 "
        "outputShadowTracks=1 section=TargetManagement verified=1",
        flush=True,
    )
    return native.TRACKER_CONFIG


def main() -> int:
    # CameraPersonTrackingNativeDeepStream resolves this function through its
    # defining module at runtime, so replace only the config generator while
    # preserving the already-audited native DeepStream graph implementation.
    native._native_tracker_config = _native_tracker_config_v2
    return native.CameraPersonTrackingNativeDeepStream().run()


if __name__ == "__main__":
    raise SystemExit(main())
