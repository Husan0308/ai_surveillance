#!/usr/bin/env python3
"""Read-only runtime prerequisite diagnostic."""
from __future__ import annotations
import importlib.util,os,shutil,sqlite3,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check(name,ok,detail):print(f"{'OK' if ok else 'ERROR'} {name}: {detail}");return bool(ok)
def main():
    results=[check("python",sys.version_info>=(3,10),sys.version.split()[0])]
    for package in ("fastapi","uvicorn","numpy","cv2","torch","PySide6","yaml"):
        results.append(check(f"package {package}",importlib.util.find_spec(package) is not None,"installed" if importlib.util.find_spec(package) else "missing"))
    for element in ("gst-launch-1.0","gst-inspect-1.0"):
        results.append(check(element,shutil.which(element) is not None,shutil.which(element) or "not in PATH"))
    inspect=shutil.which("gst-inspect-1.0")
    for plugin in ("nvv4l2decoder","nvvideoconvert"):
        ok=bool(inspect and subprocess.run([inspect,plugin],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0);results.append(check(plugin,ok,"available" if ok else "missing"))
    try:import torch;gpu=torch.cuda.is_available();detail=torch.cuda.get_device_name(0) if gpu else "CUDA unavailable"
    except Exception as exc:gpu=False;detail=str(exc)
    results.append(check("GPU",gpu,detail))
    for path in (ROOT/"config"/"project.yaml",ROOT/"config"/"cameras.yaml",ROOT/"config"/"topology.yaml"):results.append(check(str(path.relative_to(ROOT)),path.is_file(),"present" if path.is_file() else "missing"))
    for path in (ROOT/"data",ROOT/"data"/"snapshots",ROOT/"data"/"enrollment_staging"):
        try:path.mkdir(parents=True,exist_ok=True);fd,name=tempfile.mkstemp(dir=path);os.close(fd);Path(name).unlink();ok=True
        except OSError as exc:ok=False;detail=str(exc)
        results.append(check(f"writable {path.relative_to(ROOT)}",ok,"writable" if ok else detail))
    db=ROOT/"data"/"surveillance.db"
    try:
        with sqlite3.connect(db) as conn:mode=conn.execute("pragma journal_mode").fetchone()[0]
        results.append(check("database WAL",mode.lower()=="wal",mode))
    except sqlite3.Error as exc:results.append(check("database",False,str(exc)))
    return 0 if all(results) else 1
if __name__=="__main__":raise SystemExit(main())
