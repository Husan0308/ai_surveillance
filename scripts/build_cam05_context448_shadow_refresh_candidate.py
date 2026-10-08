#!/usr/bin/env python3
"""Build a fresh-source isolated shared-batch CAM-05 verifier diagnostic."""
from __future__ import annotations

from pathlib import Path

from scripts import build_cam05_context448_batch6_candidate as builder


ROOT = Path(__file__).resolve().parents[1]
builder.PARENT = ROOT / ".runtime/sixcam-cam05-context448-direct-context-20261008-r64"
builder.OUT = ROOT / ".runtime/sixcam-cam05-context448-shadow-refresh-20261008-r69"


if __name__ == "__main__":
    import json

    print(json.dumps(builder.main(), indent=2))
