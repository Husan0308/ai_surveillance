#!/usr/bin/env python3
"""Build an isolated DeepStream 9.1 MV3DT+NvDCF Re-ID candidate config.

The only association edits are the NVIDIA documentation's published example
values for minMatchingScore4ReidSimilarity and
matchingScoreWeight4ReidSimilarity. They are a runtime experiment, not a
locally calibrated identity threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one {label} field, found {count}")
    return updated


def build(base: Path, out: Path, model: str, engine: str) -> None:
    text = base.read_text()
    required = (
        "maxShadowTrackingAge: 162",
        "enableReAssoc: 1",
        "multiViewAssociatorType: 1",
        "enableLatePeerReAssoc: 1",
        "enableIDCorrection: 0",
        "communicatorType: 2",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"Base config does not match preserved B baseline: {missing}")
    if "ReID:" in text:
        raise ValueError("Base already contains a ReID section; refusing duplicate")

    # These two numeric values are the documented NvDCF accuracy example. The
    # rest of the tracking/MV3DT configuration remains byte-for-byte unchanged.
    text = replace_one(
        text,
        r"^(  minMatchingScore4ReidSimilarity:)\s*[^\n]+$",
        r"\1 0.4329",
        "minimum Re-ID similarity score",
    )
    text = replace_one(
        text,
        r"^(  matchingScoreWeight4ReidSimilarity:)\s*[^\n]+$",
        r"\1 0.8356",
        "Re-ID similarity weight",
    )

    text = text.rstrip() + f"""

ReID:
  reidType: 2
  batchSize: 8
  workspaceSize: 1000
  reidFeatureSize: 256
  reidHistorySize: 100
  inferDims: [3, 256, 128]
  networkMode: 1
  inputOrder: 0
  colorFormat: 0
  offsets: [123.6750, 116.2800, 103.5300]
  netScaleFactor: 0.01735207
  keepAspc: 1
  useVPICropScaler: 1
  addFeatureNormalization: 1
  minVisibility4GalleryUpdate: 0.6
  outputReidTensor: 1
  tltEncodedModel: {model}
  tltModelKey: nvidia_tao
  modelEngineFile: {engine}
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    build(args.base, args.out, args.model, args.engine)
    print(f"Wrote isolated Re-ID tracker config: {args.out}")


if __name__ == "__main__":
    main()
