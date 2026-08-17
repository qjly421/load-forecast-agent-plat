from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _to_series(values, index=None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return pd.Series(dtype=float)
        return values.iloc[:, 0]
    arr = np.asarray(values).reshape(-1)
    return pd.Series(arr, index=index)


def accuracy_from_arrays(actual, pred) -> float | None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if actual.size == 0 or pred.size == 0:
        return None
    n = min(actual.size, pred.size)
    actual = actual[:n]
    pred = pred[:n]
    actual = np.where(np.abs(actual) < 1e-8, 1e-8, actual)
    mape = float(np.mean(np.abs((actual - pred) / actual)))
    return float((1.0 - mape) * 100.0)


def mae_from_arrays(actual, pred) -> float | None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if actual.size == 0 or pred.size == 0:
        return None
    n = min(actual.size, pred.size)
    return float(np.mean(np.abs(actual[:n] - pred[:n])))


def mse_from_arrays(actual, pred) -> float | None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if actual.size == 0 or pred.size == 0:
        return None
    n = min(actual.size, pred.size)
    return float(np.mean((actual[:n] - pred[:n]) ** 2))


def rmse_from_arrays(actual, pred) -> float | None:
    mse = mse_from_arrays(actual, pred)
    if mse is None:
        return None
    return float(np.sqrt(mse))


def metric_row(group: pd.DataFrame) -> dict:
    actual = group["actual"].to_numpy(dtype=float)
    pred = group["pred"].to_numpy(dtype=float)
    ts = pd.to_datetime(group["timestamp"])
    midday_mask = ts.dt.hour.between(10, 15)
    night_mask = ts.dt.hour.between(18, 21)
    return {
        "overall_acc": accuracy_from_arrays(actual, pred),
        "mid_acc": accuracy_from_arrays(group.loc[midday_mask, "actual"], group.loc[midday_mask, "pred"]),
        "night_acc": accuracy_from_arrays(group.loc[night_mask, "actual"], group.loc[night_mask, "pred"]),
        "mae": mae_from_arrays(actual, pred),
        "mse": mse_from_arrays(actual, pred),
        "rmse": rmse_from_arrays(actual, pred),
    }


def build_target_day_horizon_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["benchmark_point", "target_day", "dayplus"]
    for keys, group in pred_df.groupby(group_cols, sort=True):
        benchmark_point, target_day, dayplus = keys
        metrics = metric_row(group)
        rows.append(
            {
                "benchmark_point": benchmark_point,
                "target_day": target_day,
                "dayplus": int(dayplus),
                "label": f"D{int(dayplus)}",
                "n_points": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_horizon_summary(target_horizon_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dayplus, group in target_horizon_df.groupby("dayplus", sort=True):
        rows.append(
            {
                "dayplus": int(dayplus),
                "label": f"D{int(dayplus)}",
                "n_target_days": int(group["target_day"].nunique()),
                "overall_acc": None if group["overall_acc"].dropna().empty else float(group["overall_acc"].mean()),
                "mid_acc": None if group["mid_acc"].dropna().empty else float(group["mid_acc"].mean()),
                "night_acc": None if group["night_acc"].dropna().empty else float(group["night_acc"].mean()),
                "mae": None if group["mae"].dropna().empty else float(group["mae"].mean()),
                "mse": None if group["mse"].dropna().empty else float(group["mse"].mean()),
                "rmse": None if group["rmse"].dropna().empty else float(group["rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("dayplus").reset_index(drop=True)


def build_target_day_summary(target_horizon_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in target_horizon_df.groupby(["benchmark_point", "target_day"], sort=True):
        benchmark_point, target_day = keys
        rows.append(
            {
                "benchmark_point": benchmark_point,
                "target_day": target_day,
                "n_horizons": int(group["dayplus"].nunique()),
                "overall_acc": None if group["overall_acc"].dropna().empty else float(group["overall_acc"].mean()),
                "mid_acc": None if group["mid_acc"].dropna().empty else float(group["mid_acc"].mean()),
                "night_acc": None if group["night_acc"].dropna().empty else float(group["night_acc"].mean()),
                "mae": None if group["mae"].dropna().empty else float(group["mae"].mean()),
                "mse": None if group["mse"].dropna().empty else float(group["mse"].mean()),
                "rmse": None if group["rmse"].dropna().empty else float(group["rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark_point", "target_day"]).reset_index(drop=True)


def _segment_summary(horizon_summary: pd.DataFrame, start_day: int, end_day: int) -> dict:
    part = horizon_summary.loc[horizon_summary["dayplus"].between(start_day, end_day)].copy()
    if part.empty:
        return {
            "label": f"D{start_day}-D{end_day}",
            "overall_acc": None,
            "mid_acc": None,
            "night_acc": None,
            "mae": None,
            "mse": None,
            "rmse": None,
        }
    return {
        "label": f"D{start_day}-D{end_day}",
        "overall_acc": None if part["overall_acc"].dropna().empty else float(part["overall_acc"].mean()),
        "mid_acc": None if part["mid_acc"].dropna().empty else float(part["mid_acc"].mean()),
        "night_acc": None if part["night_acc"].dropna().empty else float(part["night_acc"].mean()),
        "mae": None if part["mae"].dropna().empty else float(part["mae"].mean()),
        "mse": None if part["mse"].dropna().empty else float(part["mse"].mean()),
        "rmse": None if part["rmse"].dropna().empty else float(part["rmse"].mean()),
    }


def build_per_point_summary(target_day_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark_point, group in target_day_summary.groupby("benchmark_point", sort=True):
        rows.append(
            {
                "benchmark_point": benchmark_point,
                "n_target_days": int(len(group)),
                "overall_acc": None if group["overall_acc"].dropna().empty else float(group["overall_acc"].mean()),
                "mid_acc": None if group["mid_acc"].dropna().empty else float(group["mid_acc"].mean()),
                "night_acc": None if group["night_acc"].dropna().empty else float(group["night_acc"].mean()),
                "mae": None if group["mae"].dropna().empty else float(group["mae"].mean()),
                "mse": None if group["mse"].dropna().empty else float(group["mse"].mean()),
                "rmse": None if group["rmse"].dropna().empty else float(group["rmse"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("benchmark_point").reset_index(drop=True)


def build_formal_summary(
    *,
    profile_id: str,
    stage: str,
    model_family: str,
    strategy_id: str,
    seed: int,
    target_horizon_df: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    target_day_summary: pd.DataFrame,
    nn_backbone_id: str | None = None,
    backbone_type: str | None = None,
    source_native_config_template: str | None = None,
    native_model_contract: dict | None = None,
) -> dict:
    worst_target_day = None
    if not target_day_summary.empty:
        worst_row = target_day_summary.sort_values("overall_acc", ascending=True).iloc[0]
        worst_target_day = {
            "benchmark_point": worst_row["benchmark_point"],
            "target_day": worst_row["target_day"],
            "overall_acc": None if pd.isna(worst_row["overall_acc"]) else float(worst_row["overall_acc"]),
        }

    worst_horizon = None
    if not horizon_summary.empty:
        worst_row = horizon_summary.sort_values("overall_acc", ascending=True).iloc[0]
        worst_horizon = {
            "dayplus": int(worst_row["dayplus"]),
            "label": worst_row["label"],
            "overall_acc": None if pd.isna(worst_row["overall_acc"]) else float(worst_row["overall_acc"]),
        }

    worst_target_horizon = None
    if not target_horizon_df.empty:
        worst_row = target_horizon_df.sort_values("overall_acc", ascending=True).iloc[0]
        worst_target_horizon = {
            "benchmark_point": worst_row["benchmark_point"],
            "target_day": worst_row["target_day"],
            "dayplus": int(worst_row["dayplus"]),
            "label": worst_row["label"],
            "overall_acc": None if pd.isna(worst_row["overall_acc"]) else float(worst_row["overall_acc"]),
        }

    return {
        "profile_id": profile_id,
        "stage": stage,
        "model_family": model_family,
        "nn_backbone_id": nn_backbone_id,
        "backbone_type": backbone_type,
        "source_native_config_template": source_native_config_template,
        "normalization_method": None if native_model_contract is None else native_model_contract.get("normalization_method"),
        "time_embedding_type": None if native_model_contract is None else native_model_contract.get("time_embedding_type"),
        "only_output_after_gap": None if native_model_contract is None else native_model_contract.get("only_output_after_gap"),
        "gap_steps": None if native_model_contract is None else native_model_contract.get("gap_steps"),
        "day2_weight": None if native_model_contract is None else native_model_contract.get("day2_weight"),
        "midday_penalty": None if native_model_contract is None else native_model_contract.get("midday_penalty"),
        "strategy_id": strategy_id,
        "seed": int(seed),
        "n_target_days": int(len(target_day_summary)),
        "overall_acc": None if target_day_summary["overall_acc"].dropna().empty else float(target_day_summary["overall_acc"].mean()),
        "mid_acc": None if target_day_summary["mid_acc"].dropna().empty else float(target_day_summary["mid_acc"].mean()),
        "night_acc": None if target_day_summary["night_acc"].dropna().empty else float(target_day_summary["night_acc"].mean()),
        "mae": None if target_day_summary["mae"].dropna().empty else float(target_day_summary["mae"].mean()),
        "mse": None if target_day_summary["mse"].dropna().empty else float(target_day_summary["mse"].mean()),
        "rmse": None if target_day_summary["rmse"].dropna().empty else float(target_day_summary["rmse"].mean()),
        "segment_summaries": [
            _segment_summary(horizon_summary, 1, 5),
            _segment_summary(horizon_summary, 6, 10),
            _segment_summary(horizon_summary, 11, 14),
        ],
        "worst_target_day": worst_target_day,
        "worst_horizon": worst_horizon,
        "worst_target_day_horizon": worst_target_horizon,
        "horizon_rows": [
            {
                "dayplus": int(row.dayplus),
                "label": row.label,
                "overall_acc": None if pd.isna(row.overall_acc) else float(row.overall_acc),
                "mid_acc": None if pd.isna(row.mid_acc) else float(row.mid_acc),
                "night_acc": None if pd.isna(row.night_acc) else float(row.night_acc),
            }
            for row in horizon_summary.itertuples(index=False)
        ],
    }
