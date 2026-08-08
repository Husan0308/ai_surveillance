import ast
import unittest
from pathlib import Path

class RealtimeVisualizationTests(unittest.TestCase):
    def test_frontend_has_no_generated_camera_or_person_visuals(self):
        source=Path("services/frontend/ui.py").read_text();tree=ast.parse(source)
        names={node.name for node in ast.walk(tree) if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
        forbidden={"SimPerson","CameraSim","render_scene","render_noise","spawn_person","capture_one"}
        self.assertTrue(names.isdisjoint(forbidden));self.assertNotIn("import random",source)
