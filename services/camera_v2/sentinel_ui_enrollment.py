from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)

from .sentinel_ui_base import C, Panel, ScrollPage, label, make_button, panel_layout


class EnrollmentPage(ScrollPage):
    def __init__(self):
        super().__init__()
        self.image_paths: list[str] = []
        self.profile_index: int | None = None

        body = QHBoxLayout()
        body.setSpacing(16)

        form = Panel()
        form.setMaximumWidth(370)
        form_layout = panel_layout(form, (18, 18, 18, 18), 10)
        form_layout.addWidget(label("Shaxs ma'lumotlari", "sectionTitle"))
        form_layout.addWidget(label("Ism", "muted"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("To'liq ism")
        form_layout.addWidget(self.name)
        form_layout.addWidget(label("Qo'shimcha ma'lumot", "muted"))
        self.note = QTextEdit()
        self.note.setPlaceholderText("Lavozim, bo'lim, ruxsatlar")
        self.note.setMaximumHeight(82)
        form_layout.addWidget(self.note)

        profile_box = Panel()
        profile_layout = QVBoxLayout(profile_box)
        profile_layout.setContentsMargins(12, 12, 12, 12)
        profile_layout.addWidget(label("PROFILE PHOTO", "eyebrow"), 0, Qt.AlignCenter)
        self.profile_preview = QLabel("Tanlanmagan")
        self.profile_preview.setAlignment(Qt.AlignCenter)
        self.profile_preview.setFixedSize(170, 170)
        self.profile_preview.setStyleSheet(
            f"background:{C['field']};border:1px dashed {C['border']};"
            f"border-radius:7px;color:{C['muted']};"
        )
        profile_layout.addWidget(self.profile_preview, 0, Qt.AlignCenter)
        self.profile_name = label("Rasmlardan birini tanlang", "muted")
        self.profile_name.setAlignment(Qt.AlignCenter)
        profile_layout.addWidget(self.profile_name)
        form_layout.addWidget(profile_box)

        self.count = label("Rasmlar: 0/10 · Profile photo: tanlanmagan", "muted")
        self.count.setStyleSheet(
            f"border:1px solid {C['border']};border-radius:5px;padding:10px;color:{C['muted']};"
        )
        form_layout.addWidget(self.count)
        self.finish_button = make_button("✓  Enroll", "primary")
        self.finish_button.clicked.connect(self.finish)
        form_layout.addWidget(self.finish_button)
        form_layout.addStretch()
        body.addWidget(form, 1)

        photos = Panel()
        photos_layout = panel_layout(photos, (18, 18, 18, 18), 10)
        photos_header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.addWidget(label("10 ta yuz rasmi", "sectionTitle"))
        header_text.addWidget(
            label("Bitta odamning aniq, turli burchakdan olingan rasmlarini tanlang.", "muted")
        )
        photos_header.addLayout(header_text)
        photos_header.addStretch()
        choose = make_button("＋  10 ta rasm tanlash", "primary")
        choose.clicked.connect(self.select_images)
        photos_header.addWidget(choose)
        photos_layout.addLayout(photos_header)

        self.photo_group = QButtonGroup(self)
        self.photo_group.setExclusive(True)
        self.photo_buttons: list[QPushButton] = []
        self.photo_labels: list[QLabel] = []
        grid = QGridLayout()
        grid.setSpacing(12)
        for index in range(10):
            cell = QVBoxLayout()
            tile = make_button("＋")
            tile.setCheckable(True)
            tile.setEnabled(False)
            tile.setMinimumSize(125, 112)
            tile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            tile.clicked.connect(lambda _, i=index: self.select_profile(i))
            self.photo_group.addButton(tile, index)
            self.photo_buttons.append(tile)
            cell.addWidget(tile)
            caption = label(f"Rasm {index + 1} · bo'sh", "mono")
            caption.setAlignment(Qt.AlignCenter)
            self.photo_labels.append(caption)
            cell.addWidget(caption)
            grid.addLayout(cell, index // 5, index % 5)
            grid.setColumnStretch(index % 5, 1)
        photos_layout.addLayout(grid)

        hint = label(
            "10 ta rasm yuklang, keyin eng yaxshi tushgan rasm ustiga bosib profile photo sifatida tanlang.",
            "muted",
        )
        hint.setWordWrap(True)
        photos_layout.addWidget(hint)
        photos_layout.addStretch()
        body.addWidget(photos, 3)
        self.layout.addLayout(body)
        self.layout.addStretch()

    def select_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "10 ta yuz rasmini tanlang",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if not paths:
            return
        if len(paths) != 10:
            QMessageBox.warning(self, "Rasmlar soni", "Aynan 10 ta rasm tanlash kerak.")
            return
        valid_paths = [path for path in paths if not QPixmap(path).isNull()]
        if len(valid_paths) != 10:
            QMessageBox.warning(
                self,
                "Noto'g'ri fayl",
                "Tanlangan fayllarning barchasi ochiladigan rasm bo'lishi kerak.",
            )
            return

        self.image_paths = valid_paths
        self.profile_index = None
        self.profile_preview.clear()
        self.profile_preview.setText("Tanlanmagan")
        self.profile_name.setText("Rasmlardan birini tanlang")
        for index, path in enumerate(self.image_paths):
            button = self.photo_buttons[index]
            button.setEnabled(True)
            button.setChecked(False)
            button.setText("")
            button.setIcon(QIcon(path))
            button.setIconSize(QSize(160, 104))
            button.setStyleSheet(
                f"border:1px solid {C['border']};border-radius:6px;padding:3px;"
            )
            self.photo_labels[index].setText(f"Rasm {index + 1}")
        self.update_enrollment_status()

    def select_profile(self, index):
        if index >= len(self.image_paths):
            return
        self.profile_index = index
        for current, button in enumerate(self.photo_buttons):
            selected = current == index
            button.setChecked(selected)
            border = C["primary"] if selected else C["border"]
            width = 3 if selected else 1
            button.setStyleSheet(
                f"border:{width}px solid {border};border-radius:6px;padding:3px;"
            )
            self.photo_labels[current].setText(
                f"Rasm {current + 1} · PROFILE" if selected else f"Rasm {current + 1}"
            )
            self.photo_labels[current].setStyleSheet(
                f"color:{C['primary'] if selected else C['muted']};"
                "font:10px 'DejaVu Sans Mono';"
            )

        pixmap = QPixmap(self.image_paths[index])
        self.profile_preview.setPixmap(
            pixmap.scaled(
                self.profile_preview.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )
        self.profile_name.setText(f"Rasm {index + 1} tanlandi")
        self.update_enrollment_status()

    def update_enrollment_status(self):
        profile = (
            f"Rasm {self.profile_index + 1}"
            if self.profile_index is not None
            else "tanlanmagan"
        )
        self.count.setText(
            f"Rasmlar: {len(self.image_paths)}/10 · Profile photo: {profile}"
        )

    def finish(self):
        if not self.name.text().strip():
            QMessageBox.warning(self, "Enrollment", "Shaxsning to'liq ismini kiriting.")
            return
        if len(self.image_paths) != 10:
            QMessageBox.warning(
                self,
                "Enrollment",
                "Enrollment uchun aynan 10 ta yuz rasmi kerak.",
            )
            return
        if self.profile_index is None:
            QMessageBox.warning(
                self,
                "Enrollment",
                "Eng yaxshi rasmni profile photo sifatida tanlang.",
            )
            return
        QMessageBox.information(
            self,
            "Enrollment",
            f"{self.name.text().strip()} bazaga qo'shildi.\n"
            f"Profile photo: Rasm {self.profile_index + 1}",
        )
