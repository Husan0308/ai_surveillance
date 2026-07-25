import os
import zipfile
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "data",
    "logs",
    "snapshots",
    "recordings",
    "exports",
    "backups",
    "models",
    "node_modules",
}

IGNORE_FILES = {
    "pack_project.py",
}

IGNORE_EXT = {
    ".pyc",
    ".log",
    ".db",
    ".db-wal",
    ".db-shm",
}


def should_ignore_file(filename: str) -> bool:
    if filename in IGNORE_FILES:
        return True

    _, ext = os.path.splitext(filename)

    if ext.lower() in IGNORE_EXT:
        return True

    return False


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ROOT, f"ai_surveillance_{timestamp}.zip")

    count = 0

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if should_ignore_file(file):
                    continue

                full_path = os.path.join(path, file)
                arcname = os.path.relpath(full_path, ROOT)

                zf.write(full_path, arcname)
                count += 1

                print("added:", arcname)

    print("\n✅ ZIP created")
    print(f"📦 Files: {count}")
    print(f"📁 Path: {out_path}")


if __name__ == "__main__":
    main()