#!/usr/bin/env python3
"""Run the six-camera 60-second person detection gate using the frozen transport runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_cam_six import main
if __name__ == '__main__':
    raise SystemExit(main(person_detection=True))
