from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "services/camera_v2/sentinel_exact.py"

REQUIRED_CLASSES = {
    "Panel", "ScrollPage", "StatCard", "FaceAvatar", "CameraTile", "LiveWall",
    "BarChart", "MonitoringPage", "PeoplePage", "EventsPage", "RoomsPage",
    "EnrollmentPage", "ReportsPage", "MainWindow",
}

REQUIRED_TEXT = (
    "setFixedWidth(224)",
    "setFixedHeight(70)",
    "Recent Views",
    "Heatmap",
    "fullscreenRequested",
    "Barchasi",
    "Known",
    "Unknown",
    "Ism berish",
    "Birlashtirish",
    "Ajratish",
    "Barcha turlar",
    "Barcha xonalar",
    "Barcha odamlar",
    "dd.mm.yyyy",
    "KAMERALAR",
    "HOZIR XONADA",
    "10 ta yuz rasmi",
    "PROFILE PHOTO",
    "10 ta rasm tanlash",
    "Soatlik kirish va chiqish",
    "Known / Unknown",
    "Xonalar bo'yicha bandlik",
    "Bugungi xulosa",
    "CSV",
    "PDF",
    "face re-id · 6 cam",
    "build 2026.08 · edge worker",
)


def main() -> int:
    text = UI.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(UI))
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    missing_classes = sorted(REQUIRED_CLASSES - classes)
    missing_text = [token for token in REQUIRED_TEXT if token not in text]

    print("SENTINEL_EXACT_CONTRACT")
    print(f"ui={UI}")
    print(f"classes={len(classes)}")
    if missing_classes:
        print("MISSING_CLASSES=" + ",".join(missing_classes))
    if missing_text:
        print("MISSING_UI_TOKENS=" + " | ".join(missing_text))

    ok = not missing_classes and not missing_text
    print("SENTINEL_EXACT_CONTRACT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
