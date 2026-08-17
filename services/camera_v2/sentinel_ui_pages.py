from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QVBoxLayout,
)

from .data import CAMERAS, EVENTS, PEOPLE, ROOMS, TYPE_LABEL, camera_name, fmt, room_name
from .sentinel_ui_base import C, FaceAvatar, Panel, ScrollPage, clear_layout, label, make_button, panel_layout


class PeoplePage(ScrollPage):
    def __init__(self):
        super().__init__(); self.filter = "all"
        top = QHBoxLayout(); self.buttons = QButtonGroup(self); self.buttons.setExclusive(True)
        for key, text in [("all","Barchasi"),("known","Known"),("unknown","Unknown")]:
            b=make_button(text); b.setCheckable(True); b.setChecked(key=="all"); b.clicked.connect(lambda _, k=key: self.set_filter(k)); self.buttons.addButton(b); top.addWidget(b)
        self.search = QLineEdit(); self.search.setPlaceholderText("Ism yoki Unknown_XX qidirish"); self.search.setMaximumWidth(320); self.search.textChanged.connect(self.rebuild); top.addWidget(self.search); top.addStretch(); self.layout.addLayout(top)
        self.grid = QGridLayout(); self.grid.setSpacing(12); self.layout.addLayout(self.grid); self.layout.addStretch(); self.rebuild()

    def set_filter(self, value): self.filter=value; self.rebuild()

    def rebuild(self):
        clear_layout(self.grid); q=self.search.text().lower() if hasattr(self, 'search') else ""
        items=[p for p in PEOPLE if (self.filter=="all" or p.known==(self.filter=="known")) and q in p.label.lower()]
        for i,p in enumerate(items): self.grid.addWidget(self.person_card(p), i//3, i%3)
        for col in range(3): self.grid.setColumnStretch(col,1)

    def person_card(self, p):
        card=Panel(); card.setMinimumWidth(295); lay=panel_layout(card, (12,12,12,12), 8)
        top=QHBoxLayout(); top.addWidget(FaceAvatar(p)); info=QVBoxLayout(); name=make_button(p.label,"ghost"); name.setStyleSheet("text-align:left;font-weight:700;padding:0;border:0;"); info.addWidget(name); info.addWidget(label(p.id,"mono")); badge=label("KNOWN" if p.known else "UNKNOWN"); badge.setStyleSheet(f"background:{C['known'] if p.known else C['unknown']};color:{C['bg']};padding:3px 6px;border-radius:3px;font:8px 'DejaVu Sans Mono';"); info.addWidget(badge,0,Qt.AlignLeft); info.addStretch(); top.addLayout(info,1); lay.addLayout(top)
        details=[("Birinchi",fmt(p.first_seen)),("Oxirgi",fmt(p.last_seen)),("Xona",room_name(p.room_id) if p.in_building else "Binoda emas"),("Kameralar",", ".join(camera_name(x) for x in p.cameras))]
        for k,v in details:
            row=QHBoxLayout(); row.addWidget(label(k,"muted")); val=label(v,"mono"); val.setWordWrap(True); row.addWidget(val,1,Qt.AlignRight); lay.addLayout(row)
        actions=QHBoxLayout()
        if not p.known:
            b=make_button("⌑  Ism berish","secondary"); b.clicked.connect(lambda _,p=p:self.rename_person(p)); actions.addWidget(b)
        merge=make_button("⇉  Birlashtirish"); merge.clicked.connect(lambda _,p=p:self.merge_person(p)); actions.addWidget(merge)
        if len(p.cameras)>1: actions.addWidget(make_button("⑂  Ajratish","ghost"))
        actions.addStretch(); lay.addLayout(actions); return card

    def rename_person(self,p):
        dlg=QDialog(self); dlg.setWindowTitle("Unknown odamga ism berish"); l=QVBoxLayout(dlg); l.addWidget(label("Unknown odamga ism berish","title")); inp=QLineEdit(); inp.setPlaceholderText("To'liq ism"); l.addWidget(inp); bb=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); l.addWidget(bb)
        if dlg.exec() and inp.text().strip(): p.name=inp.text().strip(); p.known=True; self.rebuild()

    def merge_person(self,p):
        QMessageBox.information(self,"Noto'g'ri ajralgan ID'ni birlashtirish",f"Tanlangan ID {p.id} bilan birlashtiriladi va bitta global ID qoladi.")


class EventsPage(ScrollPage):
    TYPE_COLORS={"entry":C['known'],"exit":C['muted'],"transition":C['blue'],"unknown":C['unknown'],"restricted":C['offline'],"camera_offline":C['offline'],"service":C['violet']}
    def __init__(self):
        super().__init__(); filters=QGridLayout(); filters.setSpacing(8)
        self.kind=QComboBox(); self.kind.addItem("Barcha turlar","all"); [self.kind.addItem(v,k) for k,v in TYPE_LABEL.items()]
        self.room=QComboBox(); self.room.addItem("Barcha xonalar","all"); [self.room.addItem(r.name,r.id) for r in ROOMS]
        self.person=QComboBox(); self.person.addItem("Barcha odamlar","all"); [self.person.addItem(p.label,p.id) for p in PEOPLE]
        self.date=QLineEdit(); self.date.setPlaceholderText("dd.mm.yyyy")
        for i,w in enumerate([self.kind,self.room,self.person,self.date]): filters.addWidget(w,0,i)
        self.kind.currentIndexChanged.connect(self.rebuild); self.room.currentIndexChanged.connect(self.rebuild); self.person.currentIndexChanged.connect(self.rebuild); self.date.textChanged.connect(self.rebuild)
        self.layout.addLayout(filters); self.list=QVBoxLayout(); self.list.setSpacing(8); self.layout.addLayout(self.list); self.layout.addStretch(); self.rebuild()

    def rebuild(self):
        clear_layout(self.list); kind=self.kind.currentData(); room=self.room.currentData(); person=self.person.currentData(); date=self.date.text().strip()
        rows=[e for e in EVENTS if (kind=='all' or e.type==kind) and (room=='all' or e.room_id==room) and (person=='all' or e.person_id==person) and (not date or date in fmt(e.at))]
        for e in rows:
            card=Panel(); card.setMinimumHeight(80); lay=QHBoxLayout(card); lay.setContentsMargins(12,10,12,10); lay.setSpacing(12)
            thumb=QLabel(); thumb.setFixedSize(64,48); thumb.setStyleSheet(f"background:{self.TYPE_COLORS[e.type]}18;border:1px solid {C['border']};border-radius:4px;"); lay.addWidget(thumb)
            info=QVBoxLayout(); top=QHBoxLayout(); tag=label(TYPE_LABEL[e.type]); tag.setStyleSheet(f"background:{self.TYPE_COLORS[e.type]};color:{C['bg']};padding:3px 6px;border-radius:3px;font:8px 'DejaVu Sans Mono';"); top.addWidget(tag); top.addWidget(label(fmt(e.at),"mono")); top.addStretch(); info.addLayout(top); info.addWidget(label(e.message)); info.addWidget(label(f"{camera_name(e.camera_id)} · {room_name(e.room_id)}","mono")); lay.addLayout(info,1)
            if e.person_id: lay.addWidget(make_button("Profil","ghost"))
            self.list.addWidget(card)


class RoomsPage(ScrollPage):
    def __init__(self):
        super().__init__(); grid=QGridLayout(); grid.setSpacing(16); inside=[p for p in PEOPLE if p.in_building]
        for i,room in enumerate(ROOMS):
            occupants=[p for p in inside if p.room_id==room.id]; cams=[c for c in CAMERAS if c.room_id==room.id]; load=round(len(occupants)/room.capacity*100)
            card=Panel(); lay=panel_layout(card); top=QHBoxLayout(); top.addWidget(label(room.name,"sectionTitle")); top.addStretch(); top.addWidget(label(str(len(occupants)),"metric",C['primary'])); lay.addLayout(top)
            bar=QProgressBar(); bar.setMaximum(100); bar.setValue(load); lay.addWidget(bar); lay.addWidget(label(f"sig'im {room.capacity} · {load}% band","mono")); lay.addSpacing(8); lay.addWidget(label("KAMERALAR","eyebrow"))
            for cam in cams:
                row=QHBoxLayout(); row.addWidget(label(cam.name)); row.addStretch(); row.addWidget(label(f"{cam.fps:.1f} fps" if cam.online else "offline","mono",C['known'] if cam.online else C['offline'])); lay.addLayout(row)
            lay.addSpacing(8); lay.addWidget(label("HOZIR XONADA","eyebrow"))
            if occupants:
                for p in occupants:
                    row=QHBoxLayout(); row.addWidget(FaceAvatar(p,32)); row.addWidget(label(p.label)); row.addStretch(); lay.addLayout(row)
            else: lay.addWidget(label("Xona bo'sh","muted"))
            lay.addStretch(); grid.addWidget(card,0,i); grid.setColumnStretch(i,1)
        self.layout.addLayout(grid); self.layout.addStretch()
