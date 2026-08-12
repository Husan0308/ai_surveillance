"""Measure the real six-camera Qt transport and GUI event loop without user input."""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from services.frontend.ui import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--fullscreen-cameras", type=int, default=2)
    parser.add_argument("--display-fps", type=float, choices=(10.0,12.0,15.0), default=12.0)
    parser.add_argument("--fullscreen-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, help="write the final JSON metrics atomically")
    args = parser.parse_args()
    os.environ["SURVEILLANCE_DISPLAY_FPS"]=str(args.display_fps)
    app = QApplication.instance() or QApplication(sys.argv)
    process = psutil.Process()
    process.cpu_percent(None)
    window = MainWindow()
    window.show()
    system = window.sys

    fullscreen_result={"opened":[],"closed":[]}
    def exercise_fullscreen():
        if not system.sims:return
        for camera in system.sims[:max(0,args.fullscreen_cameras)]:
            fullscreen_result["opened"].append(camera.id);started=time.monotonic()
            QTimer.singleShot(round(args.fullscreen_seconds*1000),lambda:window.fs.accept() if window.fs is not None else None)
            window.open_fullscreen(camera)
            fullscreen_result["closed"].append({"camera_id":camera.id,"elapsed_seconds":time.monotonic()-started,"closed":window.fs is None})
    def finish():
        metrics=system.frontend_runtime_metrics();metrics["fullscreen"]=fullscreen_result
        metrics["frontend_process"]={"cpu_percent":process.cpu_percent(None),"rss_bytes":process.memory_info().rss,"thread_count":process.num_threads()}
        payload=json.dumps(metrics, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            temporary=args.output.with_suffix(args.output.suffix+".tmp")
            temporary.write_text(payload+"\n",encoding="utf-8")
            temporary.replace(args.output)
        print(payload)
        system.shutdown()
        app.quit()

    QTimer.singleShot(min(10000,round(args.duration*150)),exercise_fullscreen)
    QTimer.singleShot(round(args.duration * 1000), finish)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
