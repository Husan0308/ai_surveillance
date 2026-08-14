from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QFrame, QLabel, QHBoxLayout, QTabWidget, QVBoxLayout, QWidget

from . import operator_dashboard_face as face


TH = face.TH
CAMERA_SPECS = face.CAMERA_SPECS


class CudaSettingsPage(face.base.Page):
    def __init__(self):
        super().__init__()
        self.title_row("Settings", "runtime summary")
        tabs = QTabWidget()
        self.v.addWidget(tabs, 1)

        cameras = QWidget()
        cv = QVBoxLayout(cameras)
        for camera_id, name, location in CAMERA_SPECS:
            row = QFrame()
            row.setObjectName("chartCard")
            h = QHBoxLayout(row)
            h.addWidget(QLabel(f"🎥 {camera_id}  —  {name}"))
            h.addStretch(1)
            loc = QLabel(location)
            loc.setStyleSheet(f"color:{TH.DIM};")
            h.addWidget(loc)
            cv.addWidget(row)
        cv.addStretch(1)
        tabs.addTab(cameras, "🎥 Cameras")

        ai = QWidget()
        form = QFormLayout(ai)
        form.addRow("Detector", QLabel("YOLO26m · CUDA · person-only"))
        form.addRow("Tracking", QLabel("Camera-local Hungarian ownership tracker"))
        form.addRow("Cross-camera ReID", QLabel("OSNet-AIN · CPU"))
        form.addRow("Pose / Heatmap", QLabel("Not enabled"))
        tabs.addTab(ai, "🤖 AI")

        recognition = QWidget()
        rform = QFormLayout(recognition)
        rform.addRow("Face model", QLabel("InsightFace buffalo_m"))
        rform.addRow("Provider", QLabel("CUDAExecutionProvider · CPU fallback"))
        rform.addRow("Search", QLabel("upper person crop · 320×320 · low-rate"))
        rform.addRow("VRAM guard", QLabel("768 MB ORT arena cap"))
        rform.addRow("Enrollment", QLabel("10 quality-gated samples"))
        tabs.addTab(recognition, "🆔 Recognition")


def run():
    face.SettingsPage = CudaSettingsPage
    return face.run()
