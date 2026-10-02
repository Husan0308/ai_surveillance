#!/usr/bin/env python3
"""Invoke unchanged archived acceptance analyzers for a new isolated stage."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / '.runtime/final-detector-ab-20261002-uSpPqjLY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    stage.relative_to((ROOT / '.runtime').resolve())
    spec = importlib.util.spec_from_file_location('unchanged_candidate_analyzer', ARCHIVE / 'analyze_one.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STAGE = stage
    sys.argv = [str(ARCHIVE / 'analyze_one.py'), args.label]
    module.main()


if __name__ == '__main__':
    main()
