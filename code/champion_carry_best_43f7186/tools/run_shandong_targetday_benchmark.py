#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from benchmark_utils import (
    build_origin_days,
    build_result_dir,
    build_train_groups,
    current_git_info,
    date_range_days,
    dry_run_lines,
    load_profile,
    machine_benchmark_root,
    machine_result_dir,
    parse_points,
    planned_machine,
    read_yaml,
    resolve_path,
    resolve_runtime_paths,
    selected_machine,
    selected_gpu_ids,
    set_global_seed,
    write_json,
    write_yaml,
)
from data_adapter import ShandongD0D14Adapter
from evaluator import (
    build_formal_summary,
    build_horizon_summary,
    build_per_point_summary,
    build_target_day_horizon_metrics,
    build_target_day_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Shandong D1-D14 target-day benchmark.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], default="stage1")
    parser.add_argument("--model-family", choices=["nn", "lgb"], default="nn")
    parser.add_argument("--nn-backbone-id", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--model-config-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
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
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--smoke-max-train-samples", type=int, default=128)
    parser.add_argument("--issue-id", type=str, default=None)
    parser.add_argument("--change-id", type=str, default=None)
    parser.add_argument("--line-id", type=str, default=None)
    parser.add_argument("--compare-to-run", type=str, default=None)
    parser.add_argument("--baseline-run-id", type=str, default=None)
    parser.add_argument("--decision-status", type=str, default=None)
    parser.add_argument("--record-md-relpath", type=str, default=None)
    parser.add_argument("--targeted-eval-set", action="append", default=None)
    parser.add_argument("--control-only", action="store_true")
    return parser.parse_args()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_targeted_eval_sets(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for raw in values or []:
        for item in str(raw).split(","):
            text = item.strip()
            if text and text not in normalized:
                normalized.append(text)
    return normalized


def _audit_context_from_args(args: argparse.Namespace) -> dict:
    return {
        "issue_id": _normalize_optional_text(args.issue_id),
        "change_id": _normalize_optional_text(args.change_id),
        "line_id": _normalize_optional_text(args.line_id),
        "compare_to_run_id": _normalize_optional_text(args.compare_to_run),
        "baseline_run_id": _normalize_optional_text(args.baseline_run_id),
        "decision_status": _normalize_optional_text(args.decision_status),
        "record_md_relpath": _normalize_optional_text(args.record_md_relpath),
        "targeted_eval_set": _normalize_targeted_eval_sets(args.targeted_eval_set),
        "control_only": bool(args.control_only),
    }


def _resolve_nn_backbone(profile: dict, requested_backbone_id: str | None) -> tuple[str | None, dict | None]:
    if "nn" not in profile.get("model_defaults", {}):
        return None, None
    nn_defaults = profile["model_defaults"]["nn"]
    if "backbone_configs" not in nn_defaults:
        return None, None
    default_backbone_id = str(nn_defaults.get("default_backbone_id", "")).strip()
    backbone_id = str(requested_backbone_id or default_backbone_id).strip()
    if not backbone_id:
        raise ValueError("nn backbone id is empty")
    backbone_configs = nn_defaults.get("backbone_configs", {}) or {}
    if backbone_id not in backbone_configs:
        raise ValueError(f"unknown nn backbone id: {backbone_id}")
    return backbone_id, backbone_configs[backbone_id]


def _load_model_config(
    profile: dict,
    model_family: str,
    override_path: str | None = None,
    nn_backbone_id: str | None = None,
) -> tuple[dict, Path, str | None, dict | None]:
    resolved_backbone_id = None
    backbone_entry = None
    if override_path is not None:
        cfg_path = resolve_path(override_path, PACKAGE_ROOT)
    else:
        if model_family == "nn":
            resolved_backbone_id, backbone_entry = _resolve_nn_backbone(profile, nn_backbone_id)
            cfg_path = resolve_path(backbone_entry["config_path"], PACKAGE_ROOT)
        else:
            cfg_path = resolve_path(profile["model_defaults"][model_family]["config_path"], PACKAGE_ROOT)
    cfg = read_yaml(cfg_path)
    return cfg, cfg_path, resolved_backbone_id, backbone_entry


def _stage_segment_ranges(profile: dict, stage: str) -> list[tuple[int, int]]:
    configured = profile.get("stage_defaults", {}).get(stage, {}).get("segment_ranges")
    if configured:
        return [(int(start), int(end)) for start, end in configured]
    pred_days = int(profile["contract"]["pred_days"])
    return [(1, pred_days)]


def _stage_strategy_id(profile: dict, stage: str) -> str:
    return str(profile["stage_defaults"][stage]["strategy_id"])


def _gpu_slot(device: str | None) -> str:
    if not device:
        return ""
    value = str(device)
    if not value.startswith("cuda:"):
        return value
    return value.split(":", 1)[1]


def _build_run_executor(model_family: str, config: dict):
    if model_family == "nn":
        backbone_type = str(config.get("model", {}).get("benchmark_backbone_type", "legacy_targetday_dlinear"))
        if backbone_type in {"baseline", "dlinearbasictime"}:
            from models.native_tsingroc_targetday_model import ShandongNativeTsingRocTargetDayModel

            return ShandongNativeTsingRocTargetDayModel(config)
        from models.nn_targetday_model import ShandongTargetDayNNModel

        return ShandongTargetDayNNModel(config)
    if model_family == "lgb":
        from models.lgb_targetday_model import ShandongTargetDayLGBModel

        return ShandongTargetDayLGBModel(config)
    raise ValueError(model_family)


def _expected_target_horizon_frame(points, pred_days: int) -> pd.DataFrame:
    rows: list[dict] = []
    for point in points:
        for target_day in date_range_days(point.target_start, point.target_end):
            for dayplus in range(1, pred_days + 1):
                rows.append(
                    {
                        "benchmark_point": point.point_id,
                        "target_day": target_day.isoformat(),
                        "dayplus": int(dayplus),
                    }
                )
    return pd.DataFrame(rows)


def _validate_train_group_assignments(*, points, train_groups: dict[str, list], pred_days: int) -> None:
    for point in points:
        expected = [day.isoformat() for day in build_origin_days(point, pred_days=pred_days)]
        assigned: list[str] = []
        for group in train_groups[point.point_id]:
            assigned.extend(day.isoformat() for day in group.assigned_origin_days)
        if sorted(assigned) != sorted(expected):
            missing = sorted(set(expected) - set(assigned))
            extra = sorted(set(assigned) - set(expected))
            raise ValueError(
                f"train group origin coverage mismatch for {point.point_id}: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
        if len(assigned) != len(set(assigned)):
            duplicates = sorted({day for day in assigned if assigned.count(day) > 1})
            raise ValueError(f"duplicate assigned origin days detected for {point.point_id}: {duplicates[:10]}")


def _validate_predictions_contract(
    *,
    predictions: pd.DataFrame,
    points,
    pred_days: int,
    smoke: bool,
) -> None:
    required_columns = {
        "benchmark_point",
        "stage",
        "model_family",
        "nn_backbone_id",
        "strategy_id",
        "train_group_id",
        "machine",
        "gpu_slot",
        "target_day",
        "origin_day",
        "dayplus",
        "timestamp",
        "actual",
        "pred",
        "issue_id",
        "change_id",
        "line_id",
        "compare_to_run_id",
        "baseline_run_id",
        "decision_status",
        "record_md_relpath",
        "control_only",
    }
    missing_columns = sorted(required_columns - set(predictions.columns))
    if missing_columns:
        raise ValueError(f"predictions_long missing required columns: {missing_columns}")

    invalid_dayplus = predictions.loc[~predictions["dayplus"].astype(int).between(1, pred_days)]
    if not invalid_dayplus.empty:
        bad = invalid_dayplus[["benchmark_point", "target_day", "dayplus"]].head(10).to_dict(orient="records")
        raise ValueError(f"invalid dayplus detected in predictions_long: {bad}")

    combo_counts = (
        predictions.groupby(["benchmark_point", "target_day", "dayplus"], sort=True)
        .size()
        .reset_index(name="n_rows")
    )
    bad_counts = combo_counts.loc[combo_counts["n_rows"] != 96]
    if not bad_counts.empty:
        sample = bad_counts.head(10).to_dict(orient="records")
        raise ValueError(f"predictions_long has non-96 combo rows: {sample}")

    if smoke:
        return

    expected = _expected_target_horizon_frame(points, pred_days=pred_days)
    merged = expected.merge(
        combo_counts,
        how="left",
        on=["benchmark_point", "target_day", "dayplus"],
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] != "both"]
    if not missing.empty:
        sample = missing.head(10)[["benchmark_point", "target_day", "dayplus"]].to_dict(orient="records")
        raise ValueError(f"formal benchmark coverage missing target_day/dayplus combos: {sample}")


def _required_dayplus_range_for_point(
    *,
    point,
    origin_day,
    segment_range: tuple[int, int],
) -> tuple[int, int] | None:
    origin = pd.Timestamp(origin_day).date()
    point_start_offset = (pd.Timestamp(point.target_start).date() - origin).days
    point_end_offset = (pd.Timestamp(point.target_end).date() - origin).days
    start = max(int(segment_range[0]), int(point_start_offset))
    end = min(int(segment_range[1]), int(point_end_offset))
    if start > end:
        return None
    return (start, end)


def _history_days(profile: dict) -> int:
    history_points = int(profile["contract"].get("history_len_points", 672))
    step_points = max(1, int(profile["contract"].get("step_points_per_day", 96)))
    return max(1, history_points // step_points)


def _suggest_available_target_end(report: dict) -> str | None:
    first_missing = report.get("first_missing_target_day")
    if not first_missing:
        return None
    return (pd.Timestamp(first_missing).date() - timedelta(days=1)).isoformat()


def _build_data_preflight(
    *,
    merged,
    profile: dict,
    stage: str,
    points,
    train_groups,
) -> dict:
    history_days = _history_days(profile)
    segment_ranges = _stage_segment_ranges(profile, stage)
    pred_days = int(profile["contract"]["pred_days"])
    last_complete_load_day = merged.last_complete_load_day()
    failures: list[dict] = []
    point_summaries: list[dict] = []
    total_checks = 0

    for point in points:
        point_failures: list[dict] = []
        point_checks = 0
        origin_days = build_origin_days(point, pred_days=pred_days)
        for group in train_groups[point.point_id]:
            for segment_range in segment_ranges:
                for origin_day in group.assigned_origin_days:
                    required_range = _required_dayplus_range_for_point(
                        point=point,
                        origin_day=origin_day,
                        segment_range=segment_range,
                    )
                    if required_range is None:
                        continue
                    point_checks += 1
                    total_checks += 1
                    report = merged.origin_day_eligibility_report(
                        origin_day,
                        segment_range=segment_range,
                        required_target_load_segment=required_range,
                        history_days=history_days,
                        require_gap_weather=True,
                    )
                    if report["eligible"]:
                        continue
                    failure = {
                        "benchmark_point": point.point_id,
                        "train_group_id": group.group_id,
                        "origin_day": report["origin_day"],
                        "segment_range": report["segment_range"],
                        "required_target_load_segment": report["required_target_load_segment"],
                        "reason_code": report["reason_code"],
                        "reason_message": report["reason_message"],
                        "first_missing_target_day": report.get("first_missing_target_day"),
                        "last_complete_load_day": report.get("last_complete_load_day"),
                        "suggested_available_target_end": _suggest_available_target_end(report),
                    }
                    point_failures.append(failure)
                    failures.append(failure)

        summary = {
            "point_id": point.point_id,
            "target_start": point.target_start.isoformat(),
            "target_end": point.target_end.isoformat(),
            "origin_start": origin_days[0].isoformat(),
            "origin_end": origin_days[-1].isoformat(),
            "n_target_days": len(date_range_days(point.target_start, point.target_end)),
            "n_assigned_origin_days": len(origin_days),
            "n_segment_checks": point_checks,
            "n_failed_checks": len(point_failures),
            "status": "ok" if not point_failures else "failed",
        }
        if point_failures:
            first = point_failures[0]
            summary.update(
                {
                    "first_failed_origin_day": first["origin_day"],
                    "first_failure_reason_code": first["reason_code"],
                    "first_failure_message": first["reason_message"],
                    "first_missing_target_day": first.get("first_missing_target_day"),
                    "suggested_available_target_end": first.get("suggested_available_target_end"),
                    "last_complete_load_day": first.get("last_complete_load_day"),
                }
            )
        point_summaries.append(summary)

    return {
        "status": "ok" if not failures else "failed",
        "checked_at": datetime.now().isoformat(),
        "history_days": history_days,
        "segment_ranges": [[int(start), int(end)] for start, end in segment_ranges],
        "last_complete_load_day": last_complete_load_day.isoformat() if last_complete_load_day is not None else None,
        "n_total_checks": total_checks,
        "n_failed_checks": len(failures),
        "point_summaries": point_summaries,
        "first_failures": failures[:20],
    }


def _raise_preflight_failure(preflight: dict) -> None:
    details: list[str] = []
    for point in preflight.get("point_summaries", []):
        if point.get("status") == "ok":
            continue
        chunk = [str(point["point_id"])]
        if point.get("first_failed_origin_day"):
            chunk.append(f"first_failed_origin={point['first_failed_origin_day']}")
        if point.get("first_failure_reason_code"):
            chunk.append(f"reason={point['first_failure_reason_code']}")
        if point.get("first_missing_target_day"):
            chunk.append(f"first_missing_target_day={point['first_missing_target_day']}")
        if point.get("suggested_available_target_end"):
            chunk.append(f"suggested_available_target_end={point['suggested_available_target_end']}")
        if point.get("last_complete_load_day"):
            chunk.append(f"last_complete_load_day={point['last_complete_load_day']}")
        details.append(", ".join(chunk))
    detail_text = "; ".join(details[:8]) if details else "unknown data-boundary failure"
    raise RuntimeError(f"benchmark data preflight failed: {detail_text}")


def _resolve_fit_origin_days(
    *,
    merged,
    group,
    segment_range: tuple[int, int],
    stage: str,
    validation_config: dict | None = None,
) -> tuple[list, list]:
    assigned_start = group.assigned_origin_days[0]
    requested_valid_days = max(1, len(group.valid_origin_days))
    history_days = 7
    pool_start = min(merged.available_dates())
    pool_end = assigned_start - timedelta(days=1)
    eligible_pool = merged.eligible_origin_days(
        start_day=pool_start,
        end_day=pool_end,
        segment_range=segment_range,
        history_days=history_days,
        require_gap_weather=True,
    )
    if len(eligible_pool) < requested_valid_days + 1:
        raise ValueError(
            f"not enough eligible origin days before {assigned_start} for {group.group_id} "
            f"segment={segment_range}: eligible={len(eligible_pool)} required>={requested_valid_days + 1}"
        )
    validation_config = validation_config or {}
    validation_mode = str(validation_config.get("mode", "recent_contiguous"))
    if validation_mode == "recent_contiguous":
        valid_origin_days = eligible_pool[-requested_valid_days:]
    elif validation_mode == "recent_spaced_pool":
        pool_days = max(requested_valid_days, int(validation_config.get("pool_days", 42)))
        candidate_pool = eligible_pool[-pool_days:]
        if len(candidate_pool) < requested_valid_days:
            candidate_pool = eligible_pool
        idx = np.linspace(0, len(candidate_pool) - 1, requested_valid_days).astype(int)
        selected: list = []
        for raw_idx in idx:
            day = candidate_pool[int(raw_idx)]
            if day not in selected:
                selected.append(day)
        if len(selected) < requested_valid_days:
            for day in reversed(candidate_pool):
                if day not in selected:
                    selected.append(day)
                if len(selected) >= requested_valid_days:
                    break
        valid_origin_days = sorted(selected)
    else:
        raise ValueError(f"unsupported validation mode: {validation_mode}")
    train_origin_days = eligible_pool[:-requested_valid_days]
    train_origin_days = [day for day in train_origin_days if day not in set(valid_origin_days)]
    if not train_origin_days:
        raise ValueError(
            f"train origin days are empty for {group.group_id} segment={segment_range} "
            f"after reserving {requested_valid_days} validation days"
        )
    if stage in {"stage1", "stage3", "stage4"}:
        overlap_start = pd.Timestamp(group.assigned_origin_days[0]).date()
        if valid_origin_days[-1] >= overlap_start:
            raise ValueError(
                f"validation origin days overlap formal origin start for {group.group_id}: "
                f"{valid_origin_days[-1]} >= {overlap_start}"
            )
    return train_origin_days, valid_origin_days


def _load_reference_days_for_audit(
    *,
    origin_day,
    target_day,
    lag_days: list[int],
    load_reference_mode: str,
    control_load_gap_days: int,
) -> tuple:
    if load_reference_mode == "origin":
        gap_day = origin_day - timedelta(days=1)
        lag_ref_days = [origin_day - timedelta(days=int(lag)) for lag in lag_days]
        return gap_day, lag_ref_days
    if load_reference_mode == "target_recent_control":
        gap_day = target_day - timedelta(days=int(control_load_gap_days))
        lag_ref_days = [
            target_day - timedelta(days=int(control_load_gap_days) + int(lag) - 1)
            for lag in lag_days
        ]
        return gap_day, lag_ref_days
    raise ValueError(f"unsupported load_reference_mode: {load_reference_mode}")


def _weather_lead_for_audit(*, dayplus: int, weather_reference_mode: str) -> int:
    if weather_reference_mode == "dayplus":
        return int(dayplus)
    if weather_reference_mode == "fixed_d1_control":
        return 1
    raise ValueError(f"unsupported weather_reference_mode: {weather_reference_mode}")


def _build_lgb_reference_protocol_audit(*, profile: dict, model_config: dict, points) -> dict:
    feature_cfg = dict(model_config.get("features", {}))
    lag_days = [int(item) for item in feature_cfg.get("lag_days", [1, 2, 3, 4, 5, 6, 7])]
    load_reference_mode = str(feature_cfg.get("load_reference_mode", "origin"))
    weather_reference_mode = str(feature_cfg.get("weather_reference_mode", "dayplus"))
    control_load_gap_days = int(feature_cfg.get("control_load_gap_days", 2))
    pred_days = int(profile["contract"]["pred_days"])

    combo_count = 0
    combos_with_any_nonhistorical_load = 0
    combos_with_nonhistorical_gap_load = 0
    combos_with_nonhistorical_lag_load = 0
    combos_with_shorter_lead_weather = 0
    nonhistorical_lag_reference_count = 0
    unsafe_combo_count = 0
    max_load_days_beyond_cutoff = 0
    max_weather_lead_shortfall = 0
    examples: list[dict] = []
    dayplus_summary: dict[int, dict] = {}

    for point in points:
        for origin_day in build_origin_days(point, pred_days):
            for dayplus in range(1, pred_days + 1):
                target_day = origin_day + timedelta(days=dayplus)
                if target_day < point.target_start or target_day > point.target_end:
                    continue

                combo_count += 1
                strict_history_cutoff_day = origin_day - timedelta(days=1)
                gap_day, lag_ref_days = _load_reference_days_for_audit(
                    origin_day=origin_day,
                    target_day=target_day,
                    lag_days=lag_days,
                    load_reference_mode=load_reference_mode,
                    control_load_gap_days=control_load_gap_days,
                )
                late_gap = gap_day > strict_history_cutoff_day
                late_lag_days = [day for day in lag_ref_days if day > strict_history_cutoff_day]
                weather_lead_used = _weather_lead_for_audit(
                    dayplus=dayplus,
                    weather_reference_mode=weather_reference_mode,
                )
                weather_shortfall = weather_lead_used < int(dayplus)
                load_risk = late_gap or bool(late_lag_days)
                any_risk = load_risk or weather_shortfall

                summary = dayplus_summary.setdefault(
                    int(dayplus),
                    {
                        "dayplus": int(dayplus),
                        "combo_count": 0,
                        "combos_with_any_nonhistorical_load": 0,
                        "combos_with_nonhistorical_gap_load": 0,
                        "combos_with_nonhistorical_lag_load": 0,
                        "combos_with_shorter_lead_weather": 0,
                    },
                )
                summary["combo_count"] += 1

                if load_risk:
                    combos_with_any_nonhistorical_load += 1
                    summary["combos_with_any_nonhistorical_load"] += 1
                if late_gap:
                    combos_with_nonhistorical_gap_load += 1
                    summary["combos_with_nonhistorical_gap_load"] += 1
                    max_load_days_beyond_cutoff = max(
                        max_load_days_beyond_cutoff,
                        (gap_day - strict_history_cutoff_day).days,
                    )
                if late_lag_days:
                    combos_with_nonhistorical_lag_load += 1
                    summary["combos_with_nonhistorical_lag_load"] += 1
                    nonhistorical_lag_reference_count += len(late_lag_days)
                    max_load_days_beyond_cutoff = max(
                        max_load_days_beyond_cutoff,
                        max((day - strict_history_cutoff_day).days for day in late_lag_days),
                    )
                if weather_shortfall:
                    combos_with_shorter_lead_weather += 1
                    summary["combos_with_shorter_lead_weather"] += 1
                    max_weather_lead_shortfall = max(
                        max_weather_lead_shortfall,
                        int(dayplus) - int(weather_lead_used),
                    )
                if any_risk:
                    unsafe_combo_count += 1
                    if len(examples) < 12:
                        reasons = []
                        if late_gap:
                            reasons.append("gap_load_after_strict_cutoff")
                        if late_lag_days:
                            reasons.append("lag_load_after_strict_cutoff")
                        if weather_shortfall:
                            reasons.append("weather_uses_shorter_lead_than_dayplus")
                        examples.append(
                            {
                                "benchmark_point": point.point_id,
                                "origin_day": origin_day.isoformat(),
                                "target_day": target_day.isoformat(),
                                "dayplus": int(dayplus),
                                "strict_history_cutoff_day": strict_history_cutoff_day.isoformat(),
                                "gap_day": gap_day.isoformat(),
                                "lag_ref_days": [day.isoformat() for day in lag_ref_days],
                                "late_lag_days": [day.isoformat() for day in late_lag_days],
                                "weather_lead_used": int(weather_lead_used),
                                "required_weather_lead": int(dayplus),
                                "reasons": reasons,
                            }
                        )

    risk_reasons = []
    if combos_with_any_nonhistorical_load > 0:
        risk_reasons.append("nonhistorical_load_reference")
    if combos_with_shorter_lead_weather > 0:
        risk_reasons.append("shorter_lead_weather_reference")

    return {
        "profile_id": profile["profile_id"],
        "benchmark_type": profile.get("benchmark_type"),
        "model_family": "lgb",
        "load_reference_mode": load_reference_mode,
        "weather_reference_mode": weather_reference_mode,
        "control_load_gap_days": int(control_load_gap_days),
        "lag_days": lag_days,
        "strict_history_cutoff_rule": "all load references must be <= origin_day - 1",
        "strict_weather_lead_rule": "weather lead must equal dayplus under the target-day formal protocol",
        "targetday_combo_count": int(combo_count),
        "combos_with_any_nonhistorical_load": int(combos_with_any_nonhistorical_load),
        "combos_with_nonhistorical_gap_load": int(combos_with_nonhistorical_gap_load),
        "combos_with_nonhistorical_lag_load": int(combos_with_nonhistorical_lag_load),
        "nonhistorical_lag_reference_count": int(nonhistorical_lag_reference_count),
        "combos_with_shorter_lead_weather": int(combos_with_shorter_lead_weather),
        "unsafe_combo_count": int(unsafe_combo_count),
        "future_information_risk": bool(unsafe_combo_count > 0),
        "future_information_risk_reasons": risk_reasons,
        "max_load_days_beyond_cutoff": int(max_load_days_beyond_cutoff),
        "max_weather_lead_shortfall": int(max_weather_lead_shortfall),
        "dayplus_summary": [dayplus_summary[key] for key in sorted(dayplus_summary)],
        "examples": examples,
    }


def _plan_payload(
    *,
    args: argparse.Namespace,
    profile: dict,
    profile_path: Path,
    model_config: dict,
    model_config_path: Path,
    points,
    train_groups,
    result_dir: Path,
    archive_dir: Path,
    machine: str,
    data_config: dict,
    device: str | None,
    seed_info: dict,
    data_preflight: dict | None,
    audit_context: dict,
    nn_backbone_id: str | None,
    native_model_contract: dict | None,
    path_context: dict,
) -> dict:
    git_info = current_git_info()
    planned_machine_name = planned_machine(profile, args.stage, args.model_family)
    resolved_machine_root = str(profile.get("machine_roots", {}).get(machine, ""))
    resolved_machine_benchmark_root = machine_benchmark_root(profile, machine)
    resolved_machine_result = machine_result_dir(profile, machine, result_dir)
    planned_machine_root = str(profile.get("machine_roots", {}).get(planned_machine_name, ""))
    planned_machine_benchmark_root = machine_benchmark_root(profile, planned_machine_name)
    planned_machine_result = machine_result_dir(profile, planned_machine_name, result_dir)
    return {
        "profile_id": profile["profile_id"],
        "display_name": profile.get("display_name"),
        "source_profile_id": profile.get("source_profile_id"),
        "profile_status": profile.get("profile_status"),
        "availability_note": profile.get("availability_note"),
        "profile_path": str(profile_path),
        "stage": args.stage,
        "model_family": args.model_family,
        "nn_backbone_id": nn_backbone_id,
        "strategy_id": _stage_strategy_id(profile, args.stage),
        "segmented": bool(args.stage == "stage3"),
        "result_dir": str(result_dir),
        "execution_result_dir": str(result_dir),
        "machine_result_dir": str(resolved_machine_result) if resolved_machine_result is not None else None,
        "planned_machine_result_dir": str(planned_machine_result) if planned_machine_result is not None else None,
        "archive_dir": str(archive_dir),
        "desired_archive_dir": str(archive_dir),
        "planned_machine": planned_machine_name,
        "machine": machine,
        "machine_root": resolved_machine_root,
        "machine_benchmark_root": (
            str(resolved_machine_benchmark_root) if resolved_machine_benchmark_root is not None else None
        ),
        "planned_machine_root": planned_machine_root,
        "planned_machine_benchmark_root": (
            str(planned_machine_benchmark_root) if planned_machine_benchmark_root is not None else None
        ),
        "execution_cwd": str(PACKAGE_ROOT),
        "execution_repo_root": str(PACKAGE_ROOT.parent),
        "device": device,
        "gpu_slot": _gpu_slot(device),
        "gpu_ids": selected_gpu_ids(device),
        "smoke": bool(args.smoke),
        "seed_info": seed_info,
        "audit": audit_context,
        "data": data_config,
        "data_profile": profile["data"],
        "path_context": path_context,
        "resolved_paths": path_context.get("resolved_paths", {}),
        "repo_relative_paths": path_context.get("repo_relative_paths", {}),
        "path_sources": path_context.get("path_sources", {}),
        "path_existence": path_context.get("path_existence", {}),
        "paths_config_path": path_context.get("paths_config_path"),
        "data_fingerprint": path_context.get("data_fingerprint", {}),
        "contract": profile["contract"],
        "stage_config": profile["stage_defaults"][args.stage],
        "model_config_path": str(model_config_path),
        "model_config": model_config,
        "native_model_contract": native_model_contract,
        "points": [
            {
                "point_id": point.point_id,
                "target_start": point.target_start.isoformat(),
                "target_end": point.target_end.isoformat(),
                "label": point.label,
                "n_target_days": len(date_range_days(point.target_start, point.target_end)),
                "origin_days": [day.isoformat() for day in build_origin_days(point, int(profile["contract"]["pred_days"]))],
                "train_groups": [group.as_dict() for group in train_groups[point.point_id]],
            }
            for point in points
        ],
        "data_preflight": data_preflight,
        "git": git_info,
        "created_at": datetime.now().isoformat(),
        "de_dup_rule": "drop_duplicates_on(benchmark_point,target_day,dayplus,timestamp)_keep_last",
    }


def _write_run_outputs(
    *,
    result_dir: Path,
    profile: dict,
    args: argparse.Namespace,
    plan_payload: dict,
    predictions: pd.DataFrame,
    run_index_rows: list[dict],
    data_config: dict,
    model_config: dict,
    model_config_path: Path,
    merged,
    audit_context: dict,
    nn_backbone_id: str | None,
    native_model_contract: dict | None,
    reference_protocol_audit: dict | None,
) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    data_summary = {
        "rows": int(len(merged.frame)),
        "columns": int(len(merged.frame.columns)),
        "start": str(merged.frame.index.min()),
        "end": str(merged.frame.index.max()),
        "cache_path": merged.cache_path,
    }
    data_fingerprint = dict(plan_payload.get("data_fingerprint") or {})
    data_fingerprint["merged_data"] = data_summary
    plan_payload["data_fingerprint"] = data_fingerprint
    write_json(result_dir / "plan.json", plan_payload)
    if reference_protocol_audit is not None:
        write_json(result_dir / "reference_protocol_audit.json", reference_protocol_audit)
    pd.DataFrame(run_index_rows).to_csv(result_dir / "run_index.csv", index=False)

    predictions = predictions.sort_values(
        ["benchmark_point", "target_day", "dayplus", "timestamp", "train_group_id"]
    ).drop_duplicates(
        subset=["benchmark_point", "target_day", "dayplus", "timestamp"],
        keep="last",
    ).reset_index(drop=True)
    _validate_predictions_contract(
        predictions=predictions,
        points=parse_points(profile),
        pred_days=int(profile["contract"]["pred_days"]),
        smoke=bool(args.smoke),
    )
    predictions.to_csv(result_dir / "predictions_long.csv", index=False)

    horizon_metrics = build_target_day_horizon_metrics(predictions)
    horizon_metrics.to_csv(result_dir / "target_day_horizon_metrics.csv", index=False)

    horizon_summary = build_horizon_summary(horizon_metrics)
    horizon_summary.to_csv(result_dir / "horizon_summary.csv", index=False)

    target_day_summary = build_target_day_summary(horizon_metrics)
    target_day_summary.to_csv(result_dir / "target_day_summary.csv", index=False)

    per_point_summary = build_per_point_summary(target_day_summary)
    if args.stage == "stage4":
        per_point_summary.to_csv(result_dir / "per_point_summary.csv", index=False)

    summary = build_formal_summary(
        profile_id=profile["profile_id"],
        stage=args.stage,
        model_family=args.model_family,
        strategy_id=_stage_strategy_id(profile, args.stage),
        seed=int(profile.get("reproducibility", {}).get("seed", 3407)),
        target_horizon_df=horizon_metrics,
        horizon_summary=horizon_summary,
        target_day_summary=target_day_summary,
        nn_backbone_id=nn_backbone_id,
        backbone_type=None if native_model_contract is None else native_model_contract.get("backbone_type"),
        source_native_config_template=None if native_model_contract is None else native_model_contract.get("source_native_config_template"),
        native_model_contract=native_model_contract,
    )
    if not per_point_summary.empty:
        summary["per_point_summary"] = per_point_summary.to_dict(orient="records")
    summary["data_fingerprint"] = data_fingerprint
    write_yaml(result_dir / "formal_summary.yaml", summary)

    metadata = {
        "profile_id": profile["profile_id"],
        "stage": args.stage,
        "model_family": args.model_family,
        "nn_backbone_id": nn_backbone_id,
        "backbone_type": None if native_model_contract is None else native_model_contract.get("backbone_type"),
        "source_native_config_template": None if native_model_contract is None else native_model_contract.get("source_native_config_template"),
        "native_model_contract": native_model_contract,
        "normalization_method": None if native_model_contract is None else native_model_contract.get("normalization_method"),
        "time_embedding_type": None if native_model_contract is None else native_model_contract.get("time_embedding_type"),
        "only_output_after_gap": None if native_model_contract is None else native_model_contract.get("only_output_after_gap"),
        "gap_steps": None if native_model_contract is None else native_model_contract.get("gap_steps"),
        "day2_weight": None if native_model_contract is None else native_model_contract.get("day2_weight"),
        "midday_penalty": None if native_model_contract is None else native_model_contract.get("midday_penalty"),
        "issue_id": audit_context["issue_id"],
        "change_id": audit_context["change_id"],
        "line_id": audit_context["line_id"],
        "compare_to_run_id": audit_context["compare_to_run_id"],
        "baseline_run_id": audit_context["baseline_run_id"],
        "decision_status": audit_context["decision_status"],
        "record_md_relpath": audit_context["record_md_relpath"],
        "targeted_eval_set": audit_context["targeted_eval_set"],
        "control_only": audit_context["control_only"],
        "planned_machine": plan_payload["planned_machine"],
        "machine": plan_payload["machine"],
        "machine_root": plan_payload["machine_root"],
        "machine_benchmark_root": plan_payload["machine_benchmark_root"],
        "planned_machine_root": plan_payload["planned_machine_root"],
        "planned_machine_benchmark_root": plan_payload["planned_machine_benchmark_root"],
        "execution_cwd": plan_payload["execution_cwd"],
        "execution_repo_root": plan_payload["execution_repo_root"],
        "execution_result_dir": plan_payload["execution_result_dir"],
        "machine_result_dir": plan_payload["machine_result_dir"],
        "planned_machine_result_dir": plan_payload["planned_machine_result_dir"],
        "device": plan_payload["device"],
        "gpu_slot": plan_payload["gpu_slot"],
        "gpu_ids": plan_payload["gpu_ids"],
        "seed": int(profile.get("reproducibility", {}).get("seed", 3407)),
        "weather_source": data_config["weather_source"],
        "data_root": data_config["weather_root"],
        "load_path": data_config["load_path"],
        "resolved_paths": plan_payload.get("resolved_paths", {}),
        "repo_relative_paths": plan_payload.get("repo_relative_paths", {}),
        "path_sources": plan_payload.get("path_sources", {}),
        "path_existence": plan_payload.get("path_existence", {}),
        "paths_config_path": plan_payload.get("paths_config_path"),
        "data_fingerprint": data_fingerprint,
        "result_dir": str(result_dir),
        "archive_dir": plan_payload["archive_dir"],
        "desired_archive_dir": plan_payload["desired_archive_dir"],
        "smoke": bool(args.smoke),
        "official_files": profile["contract"].get("official_files", []),
        "data_summary": data_summary,
        "model_config_path": str(model_config_path),
        "model_config": model_config,
        "reference_protocol_audit_path": (
            str(result_dir / "reference_protocol_audit.json") if reference_protocol_audit is not None else None
        ),
        "reference_protocol_audit": reference_protocol_audit,
        "git": plan_payload["git"],
        "created_at": datetime.now().isoformat(),
    }
    write_json(result_dir / "metadata.json", metadata)
    return summary


def _archive_run_outputs(*, result_dir: Path, archive_dir: Path, run_name: str) -> tuple[Path | None, str]:
    archive_target = archive_dir / run_name
    archive_parent = archive_target.parent
    if not archive_parent.exists():
        return None, f"archive_root_unavailable:{archive_parent}"
    archive_target.parent.mkdir(parents=True, exist_ok=True)
    if archive_target.exists():
        shutil.rmtree(archive_target)
    shutil.copytree(result_dir, archive_target)
    return archive_target, "archived"


def _record_archive_status(*, result_dir: Path, archive_target: Path | None, archive_status: str) -> None:
    archive_effective_dir = str(archive_target) if archive_target is not None else None
    for name in ["plan.json", "metadata.json"]:
        path = result_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["archive_status"] = archive_status
        payload["archive_effective_dir"] = archive_effective_dir
        write_json(path, payload)


def main() -> None:
    args = parse_args()
    audit_context = _audit_context_from_args(args)
    profile, profile_path = load_profile(args.profile, stage=args.stage)

    model_config, model_config_path, resolved_nn_backbone_id, backbone_entry = _load_model_config(
        profile,
        args.model_family,
        args.model_config_path,
        args.nn_backbone_id,
    )
    native_model_contract = None if backbone_entry is None else dict(backbone_entry.get("native_model_contract", {}))
    if args.model_family == "nn":
        model_config.setdefault("model", {})["benchmark_backbone_id"] = resolved_nn_backbone_id
        model_config["model"]["benchmark_backbone_type"] = (
            native_model_contract.get("backbone_type", "legacy_targetday_dlinear")
            if native_model_contract is not None
            else "legacy_targetday_dlinear"
        )
    if args.device is not None:
        model_config.setdefault("model", {})["device"] = args.device
    if args.smoke:
        model_config.setdefault("model", {})["epochs"] = min(
            int(model_config["model"].get("epochs", 1)),
            int(args.smoke_epochs),
        )
        if args.model_family == "lgb":
            model_config["model"]["n_estimators"] = min(
                int(model_config["model"].get("n_estimators", 50)),
                50,
            )
            model_config["model"]["early_stopping_rounds"] = min(
                int(model_config["model"].get("early_stopping_rounds", 10)),
                10,
            )
        if args.max_train_samples is None:
            args.max_train_samples = int(args.smoke_max_train_samples)

    seed = int(profile.get("reproducibility", {}).get("seed", model_config.get("model", {}).get("seed", 3407)))
    model_config.setdefault("model", {})["seed"] = seed
    seed_info = set_global_seed(seed)
    points = parse_points(profile)
    if args.smoke:
        points = points[:1]

    strategy_id = _stage_strategy_id(profile, args.stage)
    segmented = bool(args.stage == "stage3")
    pred_days = int(profile["contract"]["pred_days"])
    valid_days = int(profile["contract"]["validation_origin_days"])
    rolling_interval_days = int(profile["contract"]["rolling_interval_days"])
    validation_config = dict(model_config.get("validation", {}))

    train_groups = {
        point.point_id: build_train_groups(
            stage=args.stage,
            point=point,
            pred_days=pred_days,
            valid_days=valid_days,
            rolling_interval_days=rolling_interval_days,
            strategy_id=strategy_id,
            segmented=segmented,
        )
        for point in points
    }
    if not args.smoke:
        _validate_train_group_assignments(points=points, train_groups=train_groups, pred_days=pred_days)
    if args.smoke:
        original_groups = train_groups
        train_groups = {point.point_id: groups[:1] for point, groups in zip(points, original_groups.values())}
        keep_assigned = pred_days if args.stage in {"stage1", "stage3", "stage4"} else rolling_interval_days
        for point in points:
            base_group = train_groups[point.point_id][0]
            train_groups[point.point_id][0] = type(base_group)(
                group_id=base_group.group_id,
                assigned_origin_days=base_group.assigned_origin_days[:keep_assigned],
                valid_origin_days=base_group.valid_origin_days[: min(2, len(base_group.valid_origin_days))],
                train_end_day=base_group.train_end_day,
                strategy_id=base_group.strategy_id,
                segmented=base_group.segmented,
            )

    planned_machine_name = planned_machine(profile, args.stage, args.model_family)
    machine = selected_machine(profile, args.stage, args.model_family, args.machine)
    data_config, path_context = resolve_runtime_paths(
        profile=profile,
        machine=machine,
        paths_config_path=args.paths_config,
        cli_overrides={
            "load_path": args.load_path,
            "weather_root": args.weather_root,
            "weather_source": args.weather_source,
            "merged_cache_root": args.cache_root,
            "result_root": args.result_root or args.output_root,
        },
    )
    run_tag = args.run_tag if args.run_tag is not None else ("dryrun" if args.dry_run else None)
    result_dir = build_result_dir(
        profile=profile,
        stage=args.stage,
        model_family=args.model_family,
        run_tag=run_tag,
        smoke=args.smoke,
    )
    archive_dir = resolve_path(profile["contract"]["archive_root"], PACKAGE_ROOT) / args.stage / args.model_family / profile["profile_id"]

    if args.dry_run:
        print(
            dry_run_lines(
                profile=profile,
                data_config=data_config,
                profile_path=profile_path,
                stage=args.stage,
                model_family=args.model_family,
                points=points,
                train_groups=train_groups,
                result_dir=result_dir,
                archive_dir=archive_dir,
                planned_machine_name=planned_machine_name,
                machine=machine,
                device=model_config.get("model", {}).get("device"),
                gpu_limit=2,
                audit_context=audit_context,
                nn_backbone_id=resolved_nn_backbone_id,
                default_backbone_id=(
                    profile.get("model_defaults", {}).get("nn", {}).get("default_backbone_id")
                    if args.model_family == "nn"
                    else None
                ),
                native_model_contract=native_model_contract,
                path_context=path_context,
            )
        )
        return

    adapter = ShandongD0D14Adapter(
        load_path=data_config["load_path"],
        weather_root=data_config["weather_root"],
        weather_source=data_config["weather_source"],
        cache_root=data_config["merged_cache_root"],
        weather_base_cols=model_config.get("features", {}).get("weather_base_cols", []),
    )
    merged = adapter.load_or_build(force_rebuild=args.force_rebuild_cache)
    data_preflight = _build_data_preflight(
        merged=merged,
        profile=profile,
        stage=args.stage,
        points=points,
        train_groups=train_groups,
    )

    plan_payload = _plan_payload(
        args=args,
        profile=profile,
        profile_path=profile_path,
        model_config=model_config,
        model_config_path=model_config_path,
        points=points,
        train_groups=train_groups,
        result_dir=result_dir,
        archive_dir=archive_dir,
        machine=machine,
        data_config=data_config,
        device=model_config.get("model", {}).get("device"),
        seed_info=seed_info,
        data_preflight=data_preflight,
        audit_context=audit_context,
        nn_backbone_id=resolved_nn_backbone_id,
        native_model_contract=native_model_contract,
        path_context=path_context,
    )
    reference_protocol_audit = None
    if args.model_family == "lgb":
        reference_protocol_audit = _build_lgb_reference_protocol_audit(
            profile=profile,
            model_config=model_config,
            points=points,
        )
        plan_payload["reference_protocol_audit"] = reference_protocol_audit
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "plan.json", plan_payload)
    if reference_protocol_audit is not None:
        write_json(result_dir / "reference_protocol_audit.json", reference_protocol_audit)
        if reference_protocol_audit["future_information_risk"] and not audit_context["control_only"]:
            raise ValueError(
                "unsafe LGB reference protocol detected; rerun with --control-only for diagnostics "
                "or revert to strict origin/dayplus references"
            )
    if data_preflight["status"] != "ok":
        _raise_preflight_failure(data_preflight)

    run_index_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for point in points:
        point_dir = result_dir / "point_runs" / point.point_id
        point_dir.mkdir(parents=True, exist_ok=True)
        for group in train_groups[point.point_id]:
            group_dir = point_dir / group.group_id
            group_dir.mkdir(parents=True, exist_ok=True)
            group_segments = _stage_segment_ranges(profile, args.stage)
            model_rows = []
            status = "success"
            error_message = ""
            train_log = []
            try:
                trained_models = []
                for segment_range in group_segments:
                    train_origin_days, valid_origin_days = _resolve_fit_origin_days(
                        merged=merged,
                        group=group,
                        segment_range=segment_range,
                        stage=args.stage,
                        validation_config=validation_config,
                    )
                    executor = _build_run_executor(args.model_family, model_config)
                    fit_kwargs = {
                        "merged": merged,
                        "train_origin_days": list(train_origin_days),
                        "valid_origin_days": list(valid_origin_days),
                        "segment_range": segment_range,
                    }
                    if args.model_family == "lgb":
                        fit_kwargs["max_train_rows"] = args.max_train_samples
                    else:
                        fit_kwargs["max_train_examples"] = args.max_train_samples
                    train_meta = executor.fit(**fit_kwargs)
                    artifact_path = group_dir / f"{args.model_family}_{segment_range[0]}_{segment_range[1]}.bin"
                    executor.save(artifact_path)
                    trained_models.append((segment_range, executor, artifact_path))
                    train_log.append(
                        {
                            "segment_range": [segment_range[0], segment_range[1]],
                            "artifact_path": str(artifact_path),
                            "train_origin_start": train_origin_days[0].isoformat(),
                            "train_origin_end": train_origin_days[-1].isoformat(),
                            "n_train_origin_days": len(train_origin_days),
                            "valid_origin_start": valid_origin_days[0].isoformat(),
                            "valid_origin_end": valid_origin_days[-1].isoformat(),
                            "n_valid_origin_days": len(valid_origin_days),
                            **train_meta,
                        }
                    )

                for origin_day in group.assigned_origin_days:
                    segment_predictions = []
                    for segment_range, executor, artifact_path in trained_models:
                        required_range = _required_dayplus_range_for_point(
                            point=point,
                            origin_day=origin_day,
                            segment_range=segment_range,
                        )
                        if required_range is None:
                            continue
                        segment_pred = executor.predict_origin(
                            merged=merged,
                            origin_day=origin_day,
                            segment_range=segment_range,
                            required_dayplus_range=required_range,
                        )
                        segment_pred["segment_id"] = f"D{segment_range[0]}_D{segment_range[1]}"
                        segment_predictions.append(segment_pred)
                    if not segment_predictions:
                        continue
                    group_pred = pd.concat(segment_predictions, ignore_index=True)
                    group_pred["benchmark_point"] = point.point_id
                    group_pred["stage"] = args.stage
                    group_pred["model_family"] = args.model_family
                    group_pred["nn_backbone_id"] = resolved_nn_backbone_id
                    group_pred["strategy_id"] = strategy_id
                    group_pred["train_group_id"] = group.group_id
                    group_pred["machine"] = machine
                    group_pred["gpu_slot"] = _gpu_slot(model_config.get("model", {}).get("device"))
                    group_pred["gpu_ids"] = ",".join(selected_gpu_ids(model_config.get("model", {}).get("device")))
                    group_pred["issue_id"] = audit_context["issue_id"]
                    group_pred["change_id"] = audit_context["change_id"]
                    group_pred["line_id"] = audit_context["line_id"]
                    group_pred["compare_to_run_id"] = audit_context["compare_to_run_id"]
                    group_pred["baseline_run_id"] = audit_context["baseline_run_id"]
                    group_pred["decision_status"] = audit_context["decision_status"]
                    group_pred["record_md_relpath"] = audit_context["record_md_relpath"]
                    group_pred["control_only"] = audit_context["control_only"]
                    group_pred = group_pred[
                        (pd.to_datetime(group_pred["target_day"]).dt.date >= point.target_start)
                        & (pd.to_datetime(group_pred["target_day"]).dt.date <= point.target_end)
                    ].reset_index(drop=True)
                    prediction_frames.append(group_pred)
            except Exception as exc:  # pragma: no cover - runtime failure path
                status = "failed"
                error_message = str(exc)

            run_index_rows.append(
                {
                    "benchmark_point": point.point_id,
                    "stage": args.stage,
                    "model_family": args.model_family,
                    "nn_backbone_id": resolved_nn_backbone_id,
                    "train_group_id": group.group_id,
                    "strategy_id": strategy_id,
                    "machine": machine,
                    "gpu_slot": _gpu_slot(model_config.get("model", {}).get("device")),
                    "gpu_ids": ",".join(selected_gpu_ids(model_config.get("model", {}).get("device"))),
                    "issue_id": audit_context["issue_id"],
                    "change_id": audit_context["change_id"],
                    "line_id": audit_context["line_id"],
                    "compare_to_run_id": audit_context["compare_to_run_id"],
                    "baseline_run_id": audit_context["baseline_run_id"],
                    "decision_status": audit_context["decision_status"],
                    "record_md_relpath": audit_context["record_md_relpath"],
                    "targeted_eval_set": ",".join(audit_context["targeted_eval_set"]),
                    "control_only": audit_context["control_only"],
                    "branch": plan_payload["git"].get("branch"),
                    "commit": plan_payload["git"].get("commit"),
                    "assigned_origin_start": group.assigned_origin_days[0].isoformat(),
                    "assigned_origin_end": group.assigned_origin_days[-1].isoformat(),
                    "n_assigned_origins": len(group.assigned_origin_days),
                    "status": status,
                    "group_dir": str(group_dir),
                    "error_message": error_message,
                    "train_log_json": json.dumps(train_log, ensure_ascii=False),
                }
            )

    pd.DataFrame(run_index_rows).to_csv(result_dir / "run_index.csv", index=False)
    failed_groups = [row for row in run_index_rows if row["status"] != "success"]
    if failed_groups:
        raise RuntimeError(f"benchmark groups failed: {failed_groups}")
    if not prediction_frames:
        raise RuntimeError("no prediction frames were produced")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = _write_run_outputs(
        result_dir=result_dir,
        profile=profile,
        args=args,
        plan_payload=plan_payload,
        predictions=predictions,
        run_index_rows=run_index_rows,
        data_config=data_config,
        model_config=model_config,
        model_config_path=model_config_path,
        merged=merged,
        audit_context=audit_context,
        nn_backbone_id=resolved_nn_backbone_id,
        native_model_contract=native_model_contract,
        reference_protocol_audit=reference_protocol_audit,
    )
    archive_target, archive_status = _archive_run_outputs(
        result_dir=result_dir,
        archive_dir=archive_dir,
        run_name=result_dir.name,
    )
    _record_archive_status(
        result_dir=result_dir,
        archive_target=archive_target,
        archive_status=archive_status,
    )
    print(f"result_dir: {result_dir}")
    if archive_target is not None:
        print(f"archive_dir: {archive_target}")
    else:
        print(f"archive_dir: unavailable ({archive_status})")
    print(f"formal_summary: {result_dir / 'formal_summary.yaml'}")
    print(
        json.dumps(
            {
                "overall_acc": summary["overall_acc"],
                "mid_acc": summary["mid_acc"],
                "night_acc": summary["night_acc"],
                "n_target_days": summary["n_target_days"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
