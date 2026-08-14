from __future__ import annotations

import http.client
import threading


def _post(path: str) -> None:
    try:
        from . import dashboard
        conn = http.client.HTTPConnection(dashboard.ML_HOST, dashboard.ML_PORT, timeout=2.0)
        conn.request("POST", path, body=b"", headers={"Connection": "close"})
        response = conn.getresponse()
        response.read()
        conn.close()
    except Exception:
        pass


def _send(kind: str, enabled: bool) -> None:
    state = "on" if enabled else "off"
    threading.Thread(
        target=_post,
        args=(f"/overlays/{kind}/{state}",),
        name=f"ui-overlay-{kind}",
        daemon=True,
    ).start()


def install(dashboard_module) -> None:
    """Add presentation-only Heatmap/Pose toggles to Live View.

    The server keeps heat accumulation and pose inference running regardless of
    these button states. Buttons only control what gets painted into live JPEGs.
    """

    LivePage = dashboard_module.LivePage
    if getattr(LivePage, "_overlay_controls_installed", False):
        return

    original_init = LivePage.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        layout = self.title_row.layout()

        heat = dashboard_module.QPushButton("Heatmap")
        heat.setCheckable(True)
        heat.setChecked(True)
        heat.setObjectName("actionButton")
        heat.setToolTip("Show/hide camera heatmap. Accumulation continues when hidden.")
        heat.setCursor(dashboard_module.Qt.CursorShape.PointingHandCursor)
        heat.toggled.connect(lambda checked: _send("heatmap", bool(checked)))

        pose = dashboard_module.QPushButton("Pose")
        pose.setCheckable(True)
        pose.setChecked(False)
        pose.setObjectName("actionButton")
        pose.setToolTip("Show/hide pose skeleton. Pose inference continues for heatmap.")
        pose.setCursor(dashboard_module.Qt.CursorShape.PointingHandCursor)
        pose.toggled.connect(lambda checked: _send("pose", bool(checked)))

        # Insert immediately before the existing fullscreen button.
        index = max(0, layout.count() - 1)
        layout.insertWidget(index, heat)
        layout.insertWidget(index + 1, pose)
        self.heatmap_toggle = heat
        self.pose_toggle = pose

    LivePage.__init__ = patched_init
    LivePage._overlay_controls_installed = True
