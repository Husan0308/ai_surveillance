from __future__ import annotations

import glob
import os
from pathlib import Path


def _find_intel_render_node() -> tuple[str | None, list[str]]:
    """Find an Intel DRM render node without guessing renderD128/renderD129."""
    diagnostics: list[str] = []
    override = os.environ.get("AI_INTEL_DRM_DEVICE", "").strip()
    if override:
        vendor_path = Path("/sys/class/drm") / Path(override).name / "device/vendor"
        vendor = "?"
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except Exception:
            pass
        diagnostics.append(f"override={override} vendor={vendor} access={int(os.access(override, os.R_OK | os.W_OK))}")
        if Path(override).exists() and vendor == "0x8086":
            return override, diagnostics
        return None, diagnostics

    for dev in sorted(glob.glob("/dev/dri/renderD*")):
        name = Path(dev).name
        vendor_path = Path("/sys/class/drm") / name / "device/vendor"
        driver_link = Path("/sys/class/drm") / name / "device/driver"
        vendor = "?"
        driver = "?"
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except Exception:
            pass
        try:
            driver = driver_link.resolve().name
        except Exception:
            pass
        access = os.access(dev, os.R_OK | os.W_OK)
        diagnostics.append(f"{dev}:vendor={vendor},driver={driver},access={int(access)}")
        if vendor == "0x8086":
            return dev, diagnostics
    return None, diagnostics


INTEL_DRM_DEVICE, DRM_DIAGNOSTICS = _find_intel_render_node()

# All VA/GStreamer environment must be fixed BEFORE Gst.init().
os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")
if INTEL_DRM_DEVICE:
    # Official gstreamer-vaapi DRM selector. Important on hybrid Intel+NVIDIA hosts.
    os.environ["GST_VAAPI_DRM_DEVICE"] = INTEL_DRM_DEVICE

# Use a dedicated registry so an earlier failed VA scan against the NVIDIA DRM
# device cannot leave this surveillance process with stale zero-feature entries.
os.environ.setdefault(
    "GST_REGISTRY",
    f"/tmp/gstreamer-surveillance-intel-v4-{os.getuid()}.bin",
)

# Never PRIME-offload the visible Intel wall back to NVIDIA.
os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

# Preserve the current smoothness/detection profile.
os.environ.setdefault("AI_YOLO_INFER_WIDTH", "448")
os.environ.setdefault("AI_YOLO_INFER_HEIGHT", "256")
os.environ.setdefault("AI_YOLO_BATCH_FPS", "0.80")
os.environ.setdefault("AI_YOLO_CONF", "0.16")

from . import intel_display_yolo26m_v3 as v3


NEW_VA_REQUIRED = ("vah264dec", "vah265dec", "vapostproc")
LEGACY_VA_REQUIRED = ("vaapih264dec", "vaapih265dec", "vaapipostproc")
COMMON_REQUIRED = (
    "rtspsrc",
    "rtph264depay",
    "rtph265depay",
    "h264parse",
    "h265parse",
    "cairooverlay",
    "compositor",
    "videoconvert",
)


def _available(Gst, names) -> bool:
    return all(Gst.ElementFactory.find(name) is not None for name in names)


def _select_backend(Gst) -> str | None:
    if _available(Gst, NEW_VA_REQUIRED):
        return "va"
    if _available(Gst, LEGACY_VA_REQUIRED):
        return "vaapi"
    return None


class IntelDisplayYolo26mV4(v3.IntelDisplayYolo26mV3):
    """V3 pipeline with hybrid-GPU-safe Intel VA backend selection.

    Preferred backend: modern gst-plugins-bad `va` elements.
    Fallback backend: Ubuntu's gstreamer1.0-vaapi elements pinned to the Intel
    DRM render node. The fallback is still Intel hardware decode; it never
    falls back to NVIDIA rendering.
    """

    def __init__(self, va_backend: str):
        self.va_backend = va_backend
        super().__init__()

    def _make(self, factory: str, name: str):
        actual = factory
        if self.va_backend == "vaapi":
            actual = {
                "vah264dec": "vaapih264dec",
                "vah265dec": "vaapih265dec",
                "vapostproc": "vaapipostproc",
            }.get(factory, factory)
        element = self.Gst.ElementFactory.make(actual, name)
        if element is None:
            raise RuntimeError(
                f"missing GStreamer element: requested={factory} actual={actual} backend={self.va_backend}"
            )
        return element

    def run(self) -> int:
        print(
            "INTEL_V4 preflight "
            f"drm={INTEL_DRM_DEVICE} va_backend={self.va_backend} "
            f"LIBVA_DRIVER_NAME={os.environ.get('LIBVA_DRIVER_NAME')} "
            f"GST_VAAPI_DRM_DEVICE={os.environ.get('GST_VAAPI_DRM_DEVICE', '-')}",
            flush=True,
        )
        return super().run()


def run() -> int:
    if INTEL_DRM_DEVICE is None:
        raise RuntimeError(
            "No Intel DRM render node (vendor 0x8086) was found. DRM scan: "
            + " | ".join(DRM_DIAGNOSTICS)
            + ". Check `ls -l /dev/dri` and Intel i915/iHD availability."
        )

    if not os.access(INTEL_DRM_DEVICE, os.R_OK | os.W_OK):
        raise RuntimeError(
            f"Intel DRM node {INTEL_DRM_DEVICE} exists but is not accessible by this user. "
            "Add the user to the render/video groups and re-login. DRM scan: "
            + " | ".join(DRM_DIAGNOSTICS)
        )

    Gst = v3._gstreamer()
    common_missing = [name for name in COMMON_REQUIRED if Gst.ElementFactory.find(name) is None]
    if common_missing:
        raise RuntimeError(
            "Missing common GStreamer plugins: " + ", ".join(common_missing)
        )

    backend = _select_backend(Gst)
    if backend is None:
        new_missing = [name for name in NEW_VA_REQUIRED if Gst.ElementFactory.find(name) is None]
        legacy_missing = [name for name in LEGACY_VA_REQUIRED if Gst.ElementFactory.find(name) is None]
        raise RuntimeError(
            "Intel DRM was pinned correctly but no usable VA decoder backend is installed. "
            f"intel_drm={INTEL_DRM_DEVICE}; modern_missing={','.join(new_missing)}; "
            f"legacy_missing={','.join(legacy_missing)}. "
            "On Ubuntu 24.04 install: gstreamer1.0-plugins-bad gstreamer1.0-vaapi "
            "intel-media-va-driver vainfo."
        )

    print(
        "INTEL_V4 backend selected "
        f"drm={INTEL_DRM_DEVICE} backend={backend} registry={os.environ.get('GST_REGISTRY')}; "
        "scan=" + " | ".join(DRM_DIAGNOSTICS),
        flush=True,
    )
    return IntelDisplayYolo26mV4(backend).run()


if __name__ == "__main__":
    raise SystemExit(run())
