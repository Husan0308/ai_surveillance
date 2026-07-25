import os
import logging
from logging.handlers import RotatingFileHandler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_LOGGER = "ai_surveillance"


def setup_logging(config):
    level_name = str(config.get("logging.level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    log_dir = config.get("logging.dir", "logs")
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(BASE_DIR, log_dir)

    os.makedirs(log_dir, exist_ok=True)

    max_mb = int(config.get("logging.max_mb", 10))
    backup_count = int(config.get("logging.backup_count", 5))

    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(level)

    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")