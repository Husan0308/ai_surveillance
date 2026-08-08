from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _project_path(value: str) -> str:
    path=Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT/path).resolve())


@dataclass(frozen=True)
class ServiceSettings:
    database_path: str = "data/surveillance.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_url: str = "http://127.0.0.1:8000"
    ml_host: str = "127.0.0.1"
    ml_port: int = 8001
    ml_url: str = "http://127.0.0.1:8001"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            database_path=_project_path(os.getenv("SURVEILLANCE_DB_PATH", cls.database_path)),
            api_host=os.getenv("SURVEILLANCE_API_HOST", cls.api_host),
            api_port=int(os.getenv("SURVEILLANCE_API_PORT", str(cls.api_port))),
            api_url=os.getenv("SURVEILLANCE_API_URL", cls.api_url),
            ml_host=os.getenv("SURVEILLANCE_ML_HOST", cls.ml_host),
            ml_port=int(os.getenv("SURVEILLANCE_ML_PORT", str(cls.ml_port))),
            ml_url=os.getenv("SURVEILLANCE_ML_URL", cls.ml_url),
            log_level=os.getenv("SURVEILLANCE_LOG_LEVEL", cls.log_level),
        )
