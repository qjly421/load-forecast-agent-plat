#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from benchmark_utils import build_origin_days, load_profile, parse_points  # noqa: E402


@dataclass(frozen=True)
class NativePointConfig:
    profile_id: str
    point_id: str
    target_start: date
    target_end: date
    pred_days: int
    expected_origin_start: date
    expected_origin_end: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild target-day D1-D14 artifacts and leakage audit from native rolling LGB outputs."
    )
    parser.add_argument("--run-root", type=Path, required=True, help="Native rolling run root.")
    parser.add_argument("--site-key", default="shandong_full", help="Native site directory under run-root.")
    parser.add_argument(
        "--profile",
        default="formal_sep2025",
        help="Formal profile id or yaml path used to derive target-day coverage.",
    )
    parser.add_argument(
        "--point-id",
        default=None,
        help="Formal point id. Defaults to the only point in the profile when unambiguous.",
    )
    parser.add_argument(
        "--site-points-csv",
        type=Path,
        default=None,
        help="Optional explicit native per-point csv path. Defaults to the only *_逐点明细.csv under site dir.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where rebuilt target-day artifacts should be written. Defaults to run-root.",
    )
    parser.add_argument(
        "--audit-json-name",
        default=None,
        help="Optional audit json filename. Defaults to targetday_<point_id>_native_protocol_audit.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved paths and audit summary.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _point_config(profile_arg: str, point_id: str | None) -> NativePointConfig:
    profile, _ = load_profile(profile_arg)
    points = parse_points(profile)
    if point_id is None:
        if len(points) != 1:
            raise ValueError("profile has multiple points; --point-id is required")
        point = points[0]
    else:
        matches = [item for item in points if item.point_id == point_id]
        if not matches:
            raise ValueError(f"point_id not found in profile: {point_id}")
        point = matches[0]
    pred_days = int(profile["contract"]["pred_days"])
    origin_days = build_origin_days(point, pred_days=pred_days)
    return NativePointConfig(
        profile_id=str(profile["profile_id"]),
        point_id=point.point_id,
        target_start=point.target_start,
        target_end=point.target_end,
        pred_days=pred_days,
        expected_origin_start=origin_days[0],
        expected_origin_end=origin_days[-1],
    )


def _infer_site_points_csv(run_root: Path, site_key: str) -> Path:
    site_dir = run_root / site_key
    matches = sorted(site_dir.glob("*_逐点明细.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one native per-point csv under {site_dir}, found {len(matches)}: {[str(item) for item in matches]}"
        )
    return matches[0]


def _mape(actual: pd.Series, pred: pd.Series) -> float:
    actual = actual.astype(float)
    pred = pred.astype(float)
    denom = actual.where(actual.abs() >= 1e-8, 1e-8)
    return float(((actual - pred).abs() / denom).mean())


def _segment_metrics(group: pd.DataFrame) -> dict[str, float]:
    ts = pd.to_datetime(group["timestamp"])
    midday = group.loc[ts.dt.hour.between(10, 15)]
    evening = group.loc[ts.dt.hour.between(18, 21)]
    night = group.loc[(ts.dt.hour < 6) | ((ts.dt.hour == 6) & (ts.dt.minute == 0))]

    def build_row(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
        if frame.empty:
            return {f"{prefix}_mape": math.nan, f"{prefix}_acc": math.nan}
        value = _mape(frame["actual"], frame["pred"])
        return {f"{prefix}_mape": value, f"{prefix}_acc": (1.0 - value) * 100.0}

    row = {}
    row.update(build_row(group, "overall"))
    row.update(build_row(midday, "midday"))
    row.update(build_row(evening, "evening"))
    row.update(build_row(night, "night"))
    return row


def build_targetday_points(native_points: pd.DataFrame, cfg: NativePointConfig) -> pd.DataFrame:
    frame = native_points.copy()
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"])
    frame["时间"] = pd.to_datetime(frame["时间"])
    frame["origin_day"] = frame["anchor_time"].dt.strftime("%Y-%m-%d")
    frame["target_day"] = pd.to_datetime(frame["预测日期"])
    frame["horizon"] = frame["相对日序"].astype(int) - 1
    mask = (
        frame["horizon"].between(1, cfg.pred_days)
        & (frame["target_day"] >= pd.Timestamp(cfg.target_start))
        & (frame["target_day"] <= pd.Timestamp(cfg.target_end))
    )
    points = frame.loc[mask, ["origin_day", "target_day", "horizon", "时间", "step", "真实负荷", "预测负荷"]].copy()
    points = points.rename(columns={"时间": "timestamp", "真实负荷": "actual", "预测负荷": "pred"})
    points["target_day"] = pd.to_datetime(points["target_day"]).dt.strftime("%Y-%m-%d")
    return points.sort_values(["target_day", "horizon", "timestamp", "origin_day"]).reset_index(drop=True)


def build_target_horizon_metrics(points: pd.DataFrame, point_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_day, horizon), group in points.groupby(["target_day", "horizon"], sort=True):
        rows.append(
            {
                "benchmark_point": point_id,
                "target_day": target_day,
                "horizon": int(horizon),
                "label": f"D{int(horizon)}",
                "n_points": int(len(group)),
                **_segment_metrics(group),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_day", "horizon"]).reset_index(drop=True)


def build_target_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_day, group in points.groupby("target_day", sort=True):
        rows.append(
            {
                "target_day": target_day,
                "horizon_count": int(group["horizon"].nunique()),
                **_segment_metrics(group),
            }
        )
    return pd.DataFrame(rows).sort_values("target_day").reset_index(drop=True)


def build_horizon_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, group in points.groupby("horizon", sort=True):
        rows.append(
            {
                "horizon": int(horizon),
                "rows": int(group["target_day"].nunique()),
                **_segment_metrics(group),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def build_coverage(points: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        points.groupby("target_day", sort=True)
        .agg(count=("horizon", "nunique"), min=("horizon", "min"), max=("horizon", "max"))
        .reset_index()
    )
    return coverage.sort_values("target_day").reset_index(drop=True)


def build_protocol_audit(run_root: Path, site_key: str, cfg: NativePointConfig, points: pd.DataFrame) -> dict[str, Any]:
    site_dir = run_root / site_key
    anchor_dirs = sorted(site_dir.glob("anchor_*"))
    leakage_rows: list[dict[str, Any]] = []

    for anchor_dir in anchor_dirs:
        time_path = anchor_dir / "time.yaml"
        if not time_path.exists():
            continue
        payload = load_yaml(time_path)
        anchor_start = pd.Timestamp(payload["backtest"][0][0])

        def _ends_after_origin(intervals: list[list[str]]) -> tuple[bool, str | None]:
            latest_end: pd.Timestamp | None = None
            hit = False
            for _, end_value in intervals:
                end_ts = pd.Timestamp(end_value)
                if latest_end is None or end_ts > latest_end:
                    latest_end = end_ts
                if end_ts >= anchor_start:
                    hit = True
            return hit, None if latest_end is None else latest_end.strftime("%Y-%m-%d %H:%M:%S")

        train_hit, train_latest = _ends_after_origin(payload.get("train", []) or [])
        valid_hit, valid_latest = _ends_after_origin(payload.get("valid", []) or [])
        if train_hit or valid_hit:
            leakage_rows.append(
                {
                    "anchor_id": anchor_dir.name,
                    "anchor_start": anchor_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "train_leaks_past_origin": bool(train_hit),
                    "valid_leaks_past_origin": bool(valid_hit),
                    "latest_train_end": train_latest,
                    "latest_valid_end": valid_latest,
                }
            )

    combo_sizes = sorted(points.groupby(["target_day", "horizon"]).size().unique().tolist())
    return {
        "profile_id": cfg.profile_id,
        "point_id": cfg.point_id,
        "run_root": str(run_root),
        "site_key": site_key,
        "target_start": cfg.target_start.isoformat(),
        "target_end": cfg.target_end.isoformat(),
        "pred_days": int(cfg.pred_days),
        "expected_origin_start": cfg.expected_origin_start.isoformat(),
        "expected_origin_end": cfg.expected_origin_end.isoformat(),
        "expected_origin_count": int((cfg.expected_origin_end - cfg.expected_origin_start).days + 1),
        "actual_origin_count": int(points["origin_day"].nunique()),
        "actual_target_count": int(points["target_day"].nunique()),
        "actual_horizon_count": int(points["horizon"].nunique()),
        "combo_point_count_unique": combo_sizes,
        "target_span": {
            "min": str(points["target_day"].min()) if not points.empty else None,
            "max": str(points["target_day"].max()) if not points.empty else None,
        },
        "origin_span": {
            "min": str(points["origin_day"].min()) if not points.empty else None,
            "max": str(points["origin_day"].max()) if not points.empty else None,
        },
        "coverage_ok": bool(
            not points.empty
            and points["origin_day"].nunique() == (cfg.expected_origin_end - cfg.expected_origin_start).days + 1
            and points["target_day"].nunique() == (cfg.target_end - cfg.target_start).days + 1
            and points["horizon"].nunique() == cfg.pred_days
            and combo_sizes == [96]
        ),
        "anchor_count": len(anchor_dirs),
        "anchors_with_future_train_or_valid": len(leakage_rows),
        "all_anchors_flagged": bool(anchor_dirs) and len(leakage_rows) == len(anchor_dirs),
        "sample_anchor_violations": leakage_rows[:5],
    }


def write_outputs(
    *,
    output_root: Path,
    cfg: NativePointConfig,
    points: pd.DataFrame,
    target_horizon_metrics: pd.DataFrame,
    target_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    audit_payload: dict[str, Any],
    audit_json_name: str,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "points_long": output_root / f"targetday_{cfg.point_id}_points_long.csv",
        "target_horizon_metrics": output_root / f"targetday_{cfg.point_id}_target_horizon_metrics.csv",
        "target_summary": output_root / f"targetday_{cfg.point_id}_target_summary.csv",
        "horizon_summary": output_root / f"targetday_{cfg.point_id}_horizon_summary.csv",
        "coverage": output_root / f"targetday_{cfg.point_id}_coverage.csv",
        "audit_json": output_root / audit_json_name,
    }
    points.to_csv(paths["points_long"], index=False)
    target_horizon_metrics.to_csv(paths["target_horizon_metrics"], index=False)
    target_summary.to_csv(paths["target_summary"], index=False)
    horizon_summary.to_csv(paths["horizon_summary"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    write_json(paths["audit_json"], audit_payload)
    return paths


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = (args.output_root or run_root).resolve()
    cfg = _point_config(args.profile, args.point_id)
    site_points_csv = (args.site_points_csv or _infer_site_points_csv(run_root, args.site_key)).resolve()

    native_points = pd.read_csv(site_points_csv)
    points = build_targetday_points(native_points, cfg)
    target_horizon_metrics = build_target_horizon_metrics(points, cfg.point_id)
    target_summary = build_target_summary(points)
    horizon_summary = build_horizon_summary(points)
    coverage = build_coverage(points)
    audit_json_name = args.audit_json_name or f"targetday_{cfg.point_id}_native_protocol_audit.json"
    audit_payload = build_protocol_audit(run_root, args.site_key, cfg, points)

    summary = {
        "run_root": str(run_root),
        "site_points_csv": str(site_points_csv),
        "output_root": str(output_root),
        "profile_id": cfg.profile_id,
        "point_id": cfg.point_id,
        "rows": int(len(points)),
        "origin_count": int(points["origin_day"].nunique()) if not points.empty else 0,
        "target_count": int(points["target_day"].nunique()) if not points.empty else 0,
        "horizon_count": int(points["horizon"].nunique()) if not points.empty else 0,
        "coverage_ok": audit_payload["coverage_ok"],
        "anchors_with_future_train_or_valid": audit_payload["anchors_with_future_train_or_valid"],
        "all_anchors_flagged": audit_payload["all_anchors_flagged"],
        "overall_acc_mean": None if target_summary.empty else float(target_summary["overall_acc"].mean()),
        "midday_acc_mean": None if target_summary.empty else float(target_summary["midday_acc"].mean()),
        "night_acc_mean": None if target_summary.empty else float(target_summary["night_acc"].mean()),
    }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    written = write_outputs(
        output_root=output_root,
        cfg=cfg,
        points=points,
        target_horizon_metrics=target_horizon_metrics,
        target_summary=target_summary,
        horizon_summary=horizon_summary,
        coverage=coverage,
        audit_payload=audit_payload,
        audit_json_name=audit_json_name,
    )
    summary["written_files"] = {key: str(path) for key, path in written.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
