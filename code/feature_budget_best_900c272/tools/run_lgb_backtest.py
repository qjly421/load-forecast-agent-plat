#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


MODE_PRESETS = {
    "medium": {
        "stage": "stage1",
        "profile": "formal_sep2025",
        "model_config_path": "./config/model/lgb_targetday_medium_strict_safe.yaml",
        "description": "Stage1 LGB strict-safe on 2025-09, 43 origin calls.",
    },
    "full": {
        "stage": "stage4",
        "profile": "formal_allpoints_available",
        "model_config_path": "./config/model/lgb_targetday_full_stage4.yaml",
        "description": "Stage4 LGB on all currently available formal points.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Shandong 14-day target-day LGB backtest with fixed medium/full modes."
    )
    parser.add_argument("--mode", choices=sorted(MODE_PRESETS), default="medium")
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--result-root", type=str, default=None)
    parser.add_argument("--paths-config", type=str, default=None)
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--weather-root", type=str, default=None)
    parser.add_argument("--weather-source", type=str, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to tools/run_shandong_targetday_benchmark.py. Repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODE_PRESETS[args.mode]
    cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "tools" / "run_shandong_targetday_benchmark.py"),
        "--stage",
        preset["stage"],
        "--model-family",
        "lgb",
        "--profile",
        preset["profile"],
        "--model-config-path",
        preset["model_config_path"],
    ]
    optional_pairs = {
        "--machine": args.machine,
        "--output-root": args.output_root,
        "--result-root": args.result_root,
        "--paths-config": args.paths_config,
        "--load-path": args.load_path,
        "--weather-root": args.weather_root,
        "--weather-source": args.weather_source,
        "--cache-root": args.cache_root,
        "--run-tag": args.run_tag,
        "--max-train-samples": args.max_train_samples,
    }
    for flag, value in optional_pairs.items():
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.smoke:
        cmd.append("--smoke")
    if args.force_rebuild_cache:
        cmd.append("--force-rebuild-cache")
    cmd.extend(args.extra_arg)

    print(f"mode: {args.mode}")
    print(f"description: {preset['description']}")
    print("command: " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(PACKAGE_ROOT)))


if __name__ == "__main__":
    main()
