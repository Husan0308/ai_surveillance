from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_service_directories_exist() -> None:
    for path in (
        ROOT / "services" / "ml_service",
        ROOT / "services" / "api_service",
        ROOT / "services" / "frontend",
    ):
        assert path.is_dir(), f"required production service missing: {path.relative_to(ROOT)}"


def test_required_service_entrypoints_exist() -> None:
    required = (
        ROOT / "services" / "ml_service" / "app" / "main.py",
        ROOT / "services" / "api_service" / "app" / "main.py",
        ROOT / "services" / "frontend" / "app" / "main.py",
    )
    for path in required:
        assert path.is_file(), f"required production entrypoint missing: {path.relative_to(ROOT)}"


def test_frontend_does_not_own_ml_runtime() -> None:
    frontend = ROOT / "services" / "frontend"
    forbidden = (
        "DeepStreamRuntime(",
        "from ultralytics import YOLO",
        "import pyds",
        "nvstreammux",
        "nvinfer",
        "nvtracker",
    )
    for path in frontend.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"frontend crossed ML boundary: {path.relative_to(ROOT)} -> {marker}"


def test_api_does_not_own_camera_decode() -> None:
    api = ROOT / "services" / "api_service"
    forbidden = (
        "DeepStreamRuntime(",
        "from ultralytics import YOLO",
        "import pyds",
        "nvstreammux",
        "nvinfer",
        "nvtracker",
    )
    for path in api.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"API crossed ML boundary: {path.relative_to(ROOT)} -> {marker}"
