import ast
from pathlib import Path

from shared.schemas import Camera, Detection, Event, Events, HeatmapPoint, Person, Room, Track


def test_shared_contracts_construct():
    assert Camera(id="cam-1", name="Entrance").id == "cam-1"
    assert Room(id="room-1", name="Lobby").name == "Lobby"
    assert Person(name="Unknown").status == "active"
    assert Track(id="t1", camera_id="cam-1", bbox_xyxy=(0, 0, 10, 10)).id == "t1"
    assert Detection(camera_id="cam-1", frame_id="1", confidence=.9, bbox_xyxy=(0, 0, 1, 1))
    assert HeatmapPoint(camera_id="cam-1", x=.5, y=.5)
    assert Events(items=[]).total == 0


def test_frontend_has_no_ml_imports():
    frontend = Path(__file__).parents[1] / "services" / "frontend"
    forbidden = ("backend.ai", "backend.cameras", "ultralytics", "torch", "insightface")
    for source_file in frontend.glob("*.py"):
        tree = ast.parse(source_file.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith(forbidden) for name in imports)
