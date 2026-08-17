#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from benchmark_utils import load_profile, parse_points, read_yaml, resolve_path, write_json
from run_shandong_targetday_benchmark import _build_lgb_reference_protocol_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a machine-checkable LGB reference protocol audit for a profile/config pair."
    )
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument("--model-config-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile, _profile_path = load_profile(args.profile, stage=args.stage)
    model_config = read_yaml(resolve_path(args.model_config_path, PACKAGE_ROOT))
    audit = _build_lgb_reference_protocol_audit(
        profile=profile,
        model_config=model_config,
        points=parse_points(profile),
    )
    output_path = resolve_path(args.output, PACKAGE_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, audit)
    print(output_path)


if __name__ == "__main__":
    main()
