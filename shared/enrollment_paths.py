"""Controlled same-host enrollment staging and validation."""
from __future__ import annotations
import os,shutil,time,uuid
from pathlib import Path

ALLOWED_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp"}
MAX_FILE_BYTES=20*1024*1024
PROJECT_ROOT=Path(__file__).resolve().parents[1]
def staging_root():return Path(os.getenv("SURVEILLANCE_ENROLLMENT_STAGING",PROJECT_ROOT/"data"/"enrollment_staging")).expanduser().resolve()
def _inside(path,root):
    try:path.relative_to(root);return True
    except ValueError:return False

def stage_files(paths,max_files=30):
    root=staging_root();root.mkdir(parents=True,exist_ok=True);session=root/uuid.uuid4().hex;session.mkdir(mode=0o700)
    staged=[]
    try:
        for index,source in enumerate(paths[:max_files]):
            item=Path(source).expanduser().resolve(strict=True)
            if not item.is_file() or item.is_symlink():raise ValueError("Enrollment source must be a regular file")
            if item.suffix.lower() not in ALLOWED_EXTENSIONS:raise ValueError(f"Unsupported image extension: {item.suffix}")
            size=item.stat().st_size
            if size<=0 or size>MAX_FILE_BYTES:raise ValueError("Enrollment image size is invalid")
            target=session/f"sample-{index:02d}{item.suffix.lower()}";shutil.copyfile(item,target);target.chmod(0o600);staged.append(str(target))
        return staged
    except Exception:
        shutil.rmtree(session,ignore_errors=True);raise

def validate_staged_paths(paths):
    import cv2
    root=staging_root();validated=[]
    for raw in paths:
        path=Path(raw).expanduser().resolve(strict=True)
        if not _inside(path,root) or not path.is_file() or path.is_symlink():raise ValueError("Enrollment path is outside controlled staging")
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:raise ValueError("Unsupported enrollment image extension")
        size=path.stat().st_size
        if size<=0 or size>MAX_FILE_BYTES:raise ValueError("Enrollment image size is invalid")
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if image is None or image.size==0:raise ValueError("Enrollment file is not a decodable image")
        validated.append(str(path))
    return validated

def cleanup_staging(max_age_hours=24,max_total_bytes=2*1024*1024*1024):
    root=staging_root();root.mkdir(parents=True,exist_ok=True);now=time.time();entries=[]
    for child in root.iterdir():
        if not child.is_dir():continue
        size=sum(p.stat().st_size for p in child.rglob("*") if p.is_file());mtime=child.stat().st_mtime;entries.append((mtime,size,child))
        if now-mtime>max_age_hours*3600:shutil.rmtree(child,ignore_errors=True)
    remaining=[x for x in entries if x[2].exists()];total=sum(x[1] for x in remaining)
    for _,size,path in sorted(remaining):
        if total<=max_total_bytes:break
        shutil.rmtree(path,ignore_errors=True);total-=size
    return total
