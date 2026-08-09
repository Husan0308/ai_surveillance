#!/usr/bin/env python3
import sqlite3,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared.settings import ServiceSettings
source=Path(ServiceSettings.from_env().database_path);target=source.parent/"backups"/f"surveillance-{time.strftime('%Y%m%d-%H%M%S')}.db";target.parent.mkdir(parents=True,exist_ok=True)
with sqlite3.connect(source) as src,sqlite3.connect(target) as dst:src.backup(dst)
print(target)
