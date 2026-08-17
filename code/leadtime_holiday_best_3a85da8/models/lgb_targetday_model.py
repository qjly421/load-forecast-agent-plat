from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from models.future_dpv import FutureDPVFeatureStore, FutureDPVAlignmentError


def _mape_style_eval(y_true, y_pred):
    """iter-15 IDEA-040: MAPE-style eval metric for early stopping.

    Mirrors the benchmark's scored accuracy form (mean relative absolute error
    as a percentage, lower is better) so `early_stopping` selects the iteration
    that minimises the *scored* error rather than absolute-MW error.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, y_true)
    return "mape_style", float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0), False


# Chinese public-holiday date ranges (inclusive). Covers the train window
# (2024-01 .. 2025-08) and the formal eval window (2025-08 .. 2026-06).
# Multi-day ranges follow State-Council official / best-estimate schedules.
# Mechanism: Shandong is heavily industrial, so public holidays cause abrupt
# industrial-demand drops -- but (per iter-15) ONLY where PV does NOT mask them,
# i.e. in the demand-dominated evening/night regime. The flag is therefore
# hour-gated at feature-build time (see _row_for_point): it fires only outside
# the solar window so it never corrupts the PV-saturated midday net-load.
# Each range carries a type tag: "major" holidays (Spring Festival, Labour
# Day, National Day) cause deep provincial-wide industrial cessation
# (train-window daily-mean drops -18..-32%), while "minor" holidays
# (New Year, Qingming, Dragon Boat, Mid-Autumn) dip only -3..-8%. A single
# binary flag forces the tree to fit both magnitudes in the same leaves and
# over-corrects the minor days -- hence the type split (holiday v2).
_CN_PUBLIC_HOLIDAY_RANGES: list[tuple[date, date, str]] = [
    (date(2024, 1, 1), date(2024, 1, 1), "minor"),
    (date(2024, 2, 10), date(2024, 2, 17), "major"),   # Spring Festival
    (date(2024, 4, 4), date(2024, 4, 6), "minor"),     # Qingming
    (date(2024, 5, 1), date(2024, 5, 5), "major"),     # Labour Day
    (date(2024, 6, 8), date(2024, 6, 10), "minor"),    # Dragon Boat
    (date(2024, 9, 15), date(2024, 9, 17), "minor"),   # Mid-Autumn
    (date(2024, 10, 1), date(2024, 10, 7), "major"),   # National Day
    (date(2025, 1, 1), date(2025, 1, 1), "minor"),
    (date(2025, 1, 28), date(2025, 2, 4), "major"),    # Spring Festival
    (date(2025, 4, 4), date(2025, 4, 6), "minor"),     # Qingming
    (date(2025, 5, 1), date(2025, 5, 5), "major"),     # Labour Day
    (date(2025, 5, 31), date(2025, 6, 2), "minor"),    # Dragon Boat
    (date(2025, 10, 1), date(2025, 10, 8), "major"),   # National Day + Mid-Autumn
    (date(2026, 1, 1), date(2026, 1, 1), "minor"),
    (date(2026, 2, 16), date(2026, 2, 22), "major"),   # Spring Festival (CNY Feb 17)
    (date(2026, 4, 4), date(2026, 4, 6), "minor"),     # Qingming
    (date(2026, 5, 1), date(2026, 5, 5), "major"),     # Labour Day
    (date(2026, 6, 19), date(2026, 6, 21), "minor"),   # Dragon Boat (in medium-eval month)
    (date(2026, 10, 1), date(2026, 10, 8), "major"),   # National Day + Mid-Autumn
]


def _cn_public_holiday_days() -> frozenset[date]:
    """Expand the inclusive ranges into a set of holiday dates (memoised)."""
    days: set[date] = set()
    for start, end, _tag in _CN_PUBLIC_HOLIDAY_RANGES:
        cur = start
        while cur <= end:
            days.add(cur)
            cur += timedelta(days=1)
    return frozenset(days)


def _cn_public_holiday_day_types() -> dict[date, str]:
    """Expand the inclusive ranges into a day -> type map (memoised)."""
    day_types: dict[date, str] = {}
    for start, end, tag in _CN_PUBLIC_HOLIDAY_RANGES:
        cur = start
        while cur <= end:
            day_types[cur] = tag
            cur += timedelta(days=1)
    return day_types


_CN_HOLIDAY_DAYS: frozenset[date] = _cn_public_holiday_days()
_CN_HOLIDAY_DAY_TYPES: dict[date, str] = _cn_public_holiday_day_types()


def _cn_holiday_pos_norm() -> dict[date, float]:
    """Day -> position within its holiday range, normalised to [0, 1]
    (0 = first day, 1 = last day; single-day ranges -> 0.5 = own peak).

    IDEA-068 mechanism: industrial cessation RAMPS -- day 1 of a range is a
    partial shutdown (plants close through the day), mid-range is deepest, the
    final day sees early restart. A position-blind binary averages over the
    ramp, so leaves over-discount day 1 and under-discount the last day."""
    pos: dict[date, float] = {}
    for start, end, _tag in _CN_PUBLIC_HOLIDAY_RANGES:
        length = (end - start).days + 1
        denom = max(1, length - 1)
        cur = start
        idx = 0
        while cur <= end:
            pos[cur] = idx / denom if length > 1 else 0.5
            cur += timedelta(days=1)
            idx += 1
    return pos


def _cn_days_after_holiday_run() -> dict[date, int]:
    """Day -> days since the end of the most recent >=2-day holiday range,
    within a 7-day lookback (0 = the first day after the range = the industrial
    RESTART day); days not covered are absent (sentinel 7 at feature build).

    IDEA-068 mechanism: after a multi-day cessation, production restart
    OVERSHOOTS the weekday norm for 1-3 days (make-up orders, lines brought
    back online) -- the mirror image of the cessation depression that the
    iter-39 anchor flags discount. Without a marker the tree reads the
    elevated anchor/target as a regime shift and over-predicts."""
    after: dict[date, int] = {}
    for start, end, _tag in _CN_PUBLIC_HOLIDAY_RANGES:
        if (end - start).days + 1 < 2:
            continue  # single-day holidays restart as an ordinary weekday
        for k in range(1, 8):
            d = end + timedelta(days=k)
            val = k - 1
            if d not in after or val < after[d]:
                after[d] = val
    return after


_CN_HOLIDAY_POS_NORM: dict[date, float] = _cn_holiday_pos_norm()
_CN_DAYS_AFTER_HOLIDAY_RUN: dict[date, int] = _cn_days_after_holiday_run()


@dataclass
class PointPredictionFrame:
    frame: pd.DataFrame
    n_rows: int


class ShandongTargetDayLGBModel:
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = dict(config["model"])
        self.feature_cfg = dict(config.get("features", {}))
        self.validation_cfg = dict(config.get("validation", {}))
        self.model = None
        self.feature_names_: list[str] | None = None
        self.weather_base_cols = list(self.feature_cfg.get("weather_base_cols", []))
        self.lag_days = list(self.feature_cfg.get("lag_days", [1, 2, 3, 4, 5, 6, 7]))
        self.rolling_windows = list(self.feature_cfg.get("rolling_windows", [96, 672]))
        self.segment_range: tuple[int, int] | None = None
        self.early_stopping_rounds = int(self.model_cfg.get("early_stopping_rounds", 30))
        self.enable_rolling_stats = bool(self.feature_cfg.get("enable_rolling_stats", False))
        # Cache for the contiguous trailing-window rolling values, keyed by
        # (origin_day, window). Because the rewritten rolling stats are day-aligned
        # (full trailing day(s), independent of slot_index), the value list is shared
        # across all 96 slots of an origin -> avoids 96x redundant recompute.
        self._rolling_values_cache: dict[tuple[date, int], list[float] | None] = {}
        self.enable_dayplus_calendar_interactions = bool(
            self.feature_cfg.get("enable_dayplus_calendar_interactions", False)
        )
        self.enable_dayplus_lag_interactions = bool(
            self.feature_cfg.get("enable_dayplus_lag_interactions", False)
        )
        self.enable_weather_delta_features = bool(
            self.feature_cfg.get("enable_weather_delta_features", False)
        )
        self.enable_horizon_bucket_features = bool(
            self.feature_cfg.get("enable_horizon_bucket_features", False)
        )
        self.enable_monthstart_regime_features = bool(
            self.feature_cfg.get("enable_monthstart_regime_features", False)
        )
        # IDEA-053 (round 4): transition-month-gated monthstart family -- the
        # month-boundary flags re-introduced as SEPARATE sparse columns that fire
        # only when the target month is NOT a summer steady-state month.
        self.enable_monthstart_gated_features = bool(
            self.feature_cfg.get("enable_monthstart_gated_features", False)
        )
        self.enable_lag_regime_features = bool(
            self.feature_cfg.get("enable_lag_regime_features", False)
        )
        self.enable_temperature_group_features = bool(
            self.feature_cfg.get("enable_temperature_group_features", False)
        )
        self.enable_day_night_regime_features = bool(
            self.feature_cfg.get("enable_day_night_regime_features", False)
        )
        self.enable_holiday_features = bool(
            self.feature_cfg.get("enable_holiday_features", False)
        )
        # Holiday v2: split the single binary into major/minor type flags so
        # leaves can specialise on cessation magnitude (see table comment above).
        self.enable_holiday_v2_features = bool(
            self.feature_cfg.get("enable_holiday_v2_features", False)
        )
        # IDEA-067 (round 5): anchor-holiday context, HOUR-GATED to the demand
        # regime. iter-38 proved the family's night-band value (medium night
        # +0.264) but its full validation showed the un-gated variant misfires
        # at midday, where the anchor slot value is PV-dominated: a demand-
        # interpreted discount then corrupts a solar-noise anchor. The gate
        # mirrors iter-15's proven target-side hour gate.
        self.enable_anchor_holiday_features = bool(
            self.feature_cfg.get("enable_anchor_holiday_features", False)
        )
        # IDEA-068 (round 5): holiday-axis completion -- intra-range ramp
        # position + restart-snap days-after markers (target and gap sides),
        # all hour-gated to the demand regime like iters 15/39.
        self.enable_holiday_position_features = bool(
            self.feature_cfg.get("enable_holiday_position_features", False)
        )
        self.enable_temperature_fix_features = bool(
            self.feature_cfg.get("enable_temperature_fix_features", False)
        )
        self.enable_temperature_fix_daily_summary = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_summary", True)
        )
        self.enable_temperature_fix_daily_core_stats = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_core_stats", True)
        )
        self.enable_temperature_fix_daily_slot_delta = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_slot_delta", True)
        )
        self.enable_temperature_fix_daily_gap_delta = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_gap_delta", True)
        )
        self.enable_temperature_fix_daily_flags = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_flags", True)
        )
        self.enable_temperature_fix_daily_bins = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_bins", True)
        )
        self.enable_temperature_fix_streak_adjust = bool(
            self.feature_cfg.get("enable_temperature_fix_streak_adjust", True)
        )
        self.temperature_group_base_cols = [
            str(item) for item in self.feature_cfg.get("temperature_group_base_cols", [])
        ]
        self.temperature_fix_base_cols = [
            str(item) for item in self.feature_cfg.get("temperature_fix_base_cols", [])
        ]
        self.temperature_fix_cold_threshold_c = float(
            self.feature_cfg.get("temperature_fix_cold_threshold_c", 0.0)
        )
        self.temperature_fix_hot_threshold_c = float(
            self.feature_cfg.get("temperature_fix_hot_threshold_c", 30.0)
        )
        self.temperature_fix_cold_ref_c = float(self.feature_cfg.get("temperature_fix_cold_ref_c", 3.0))
        self.temperature_fix_hot_ref_c = float(self.feature_cfg.get("temperature_fix_hot_ref_c", 27.0))
        self.temperature_fix_decay = float(self.feature_cfg.get("temperature_fix_decay", 0.6))
        self.temperature_fix_lookback_days = int(self.feature_cfg.get("temperature_fix_lookback_days", 14))
        self.enable_temperature_fix_anomaly = bool(
            self.feature_cfg.get("enable_temperature_fix_anomaly", False)
        )
        # HDD/CDD V-split degree-day abstraction (holiday-v2 sibling family):
        # below/above a comfort threshold heating/AC uptake switches on with a
        # non-linear asymmetric ramp that a raw linear temperature column cannot
        # express. Local U-curve minimum sits at 12-16C -> default ref 14C.
        self.enable_hdd_cdd_features = bool(
            self.feature_cfg.get("enable_hdd_cdd_features", False)
        )
        self.hdd_cdd_reference_temp = float(
            self.feature_cfg.get("hdd_cdd_reference_temp", 14.0)
        )
        self.hdd_cdd_week_days = int(self.feature_cfg.get("hdd_cdd_week_days", 7))
        self.temperature_fix_anomaly_days = int(
            self.feature_cfg.get("temperature_fix_anomaly_days", 7)
        )
        self._temperature_group_index_cache: dict[tuple[str, str], list[int]] = {}
        self._temperature_group_stats_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._temperature_daily_summary_cache: dict[tuple[str, str, str], dict[str, np.ndarray | float] | None] = {}
        self._temperature_streak_cache: dict[tuple[str, str, str], int | None] = {}
        self._temperature_trailing_mean_cache: dict[tuple[str, str, int], float | None] = {}
        self.feature_selection_cfg = dict(self.config.get("feature_selection", {}))
        self.sample_weight_cfg = dict(self.config.get("sample_weight", {}))
        self.calibration_cfg = dict(self.config.get("calibration", {}))
        self.calibration_bias_: dict[tuple[int, int], float] = {}
        self.load_reference_mode = str(self.feature_cfg.get("load_reference_mode", "origin"))
        self.control_load_gap_days = int(self.feature_cfg.get("control_load_gap_days", 2))
        self.weather_reference_mode = str(self.feature_cfg.get("weather_reference_mode", "dayplus"))
        # Optional strict-horizon future-DPV feature (VAE-inferred non-direct/PV series).
        # Contract: (origin_day, target_day, dayplus, timestamp) exact join; hourly input
        # expanded repeat_4 -> 15-min. Off by default; enabling requires a finite
        # validation.train_origin_start inside the parquet coverage.
        self.enable_future_dpv = bool(self.feature_cfg.get("enable_future_dpv", False))
        self.future_dpv_store: FutureDPVFeatureStore | None = None
        if self.enable_future_dpv:
            future_dpv_path = self.feature_cfg.get("future_dpv_path")
            if not future_dpv_path:
                raise ValueError(
                    "features.enable_future_dpv=true requires features.future_dpv_path"
                )
            self.future_dpv_store = FutureDPVFeatureStore(
                future_dpv_path,
                value_col=str(self.feature_cfg.get("future_dpv_value_col", "future_dpv_mw")),
                hourly_expansion=str(
                    self.feature_cfg.get("future_dpv_hourly_expansion", "repeat_4")
                ),
            )
        self.enable_apparent_temp_hour_interaction = bool(
            self.feature_cfg.get("enable_apparent_temp_hour_interaction", False)
        )
        self._apparent_temp_idx_cache: dict[str, list[int] | None] = {}

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        return float(np.std(values)) if values else 0.0

    @staticmethod
    def _safe_min(values: list[float]) -> float:
        return float(np.min(values)) if values else 0.0

    @staticmethod
    def _safe_max(values: list[float]) -> float:
        return float(np.max(values)) if values else 0.0

    def _rolling_feature_values(self, merged, origin_day: date, slot_index: int, window: int) -> list[float] | None:
        """Genuine contiguous trailing-day level window (day-aligned, origin-side).

        Mechanism: load = (operational base level) x (calendar shape) + (weather
        deviation). The d1..d7 same-slot lags encode the calendar SHAPE at this
        hour; they cannot represent the operational BASE LEVEL (the average demand
        across all slots of recent days). A contiguous trailing-day mean is a
        low-variance level anchor that lets the tree attribute a target residual to
        a level shift (growth, weekday<->weekend base change) rather than confusing
        it with a weather signal -- most valuable at D4-D5 where the model
        extrapolates from noisy point-lags. Window=96 -> previous full day; window=672
        -> trailing 7 days. All reads are day_load(origin_day - k), i.e. strictly
        origin-side, preserving the protocol audit. Values are independent of
        slot_index, so they are cached per (origin_day, window).
        """
        if window <= 0:
            return []
        key = (origin_day, int(window))
        if key in self._rolling_values_cache:
            return self._rolling_values_cache[key]
        history_days = max(1, int(round(int(window) / 96)))
        values: list[float] = []
        for offset in range(1, history_days + 1):
            lag_day = origin_day - timedelta(days=offset)
            lag_load = merged.day_load(lag_day)
            if lag_load is None:
                self._rolling_values_cache[key] = None
                return None
            values.extend(float(x) for x in lag_load)
        max_points = max(1, int(window))
        values = values[:max_points]
        self._rolling_values_cache[key] = values
        return values

    @staticmethod
    def future_dpv_alignment_audit(self) -> dict:
        audit = {
            "enabled": bool(self.enable_future_dpv),
            "feature_name": "future_dpv_mw" if self.enable_future_dpv else None,
            "train_origin_start": self.validation_cfg.get("train_origin_start"),
        }
        if self.future_dpv_store is not None:
            audit.update(self.future_dpv_store.audit_snapshot())
        return audit

    def _horizon_bucket(dayplus: int) -> int:
        if dayplus <= 5:
            return 1
        if dayplus <= 10:
            return 2
        return 3

    @staticmethod
    def _temperature_bin_5c(value: float, *, lower: float = -40.0, upper: float = 60.0) -> int:
        clipped = min(max(float(value), lower), upper - 1e-6)
        return int(np.floor((clipped - lower) / 5.0))

    def _target_weather_matrix(self, merged, target_day: date, dayplus: int) -> np.ndarray | None:
        if self.weather_reference_mode == "dayplus":
            return merged.day_feature_matrix(target_day, f"D_{dayplus}__")
        if self.weather_reference_mode == "fixed_d1_control":
            return merged.day_feature_matrix(target_day, "D_1__")
        raise ValueError(f"unsupported weather_reference_mode: {self.weather_reference_mode}")

    def _apparent_temp_indices(self, merged, prefix: str) -> list[int] | None:
        """Indices of apparent_temperature columns (across all stations) under prefix."""
        if prefix in self._apparent_temp_idx_cache:
            return self._apparent_temp_idx_cache[prefix]
        cols = merged.prefix_feature_columns(prefix)
        if not cols:
            self._apparent_temp_idx_cache[prefix] = None
            return None
        idx = [i for i, c in enumerate(cols) if c.endswith("__apparent_temperature")]
        self._apparent_temp_idx_cache[prefix] = idx if idx else None
        return self._apparent_temp_idx_cache[prefix]

    def _temperature_group_features(
        self,
        *,
        merged,
        origin_day: date,
        target_day: date,
        dayplus: int,
        slot_index: int,
        target_weather: np.ndarray,
        gap_weather: np.ndarray,
    ) -> dict[str, float] | None:
        base_cols = self.temperature_group_base_cols
        if not base_cols:
            return {}

        if self.weather_reference_mode == "fixed_d1_control":
            target_prefix = "D_1__"
        else:
            target_prefix = f"D_{dayplus}__"
        target_cols = merged.prefix_feature_columns(target_prefix)
        gap_cols = merged.prefix_feature_columns("D_0__")
        if not target_cols or not gap_cols:
            return None

        stats: dict[str, float] = {}
        for base_col in base_cols:
            cache_key = (
                origin_day.isoformat(),
                target_day.isoformat(),
                int(dayplus),
                base_col,
            )
            cached = self._temperature_group_stats_cache.get(cache_key)
            if cached is None:
                target_key = (target_prefix, base_col)
                gap_key = ("D_0__", base_col)
                target_indices = self._temperature_group_index_cache.get(target_key)
                if target_indices is None:
                    target_indices = [idx for idx, col in enumerate(target_cols) if col.endswith(f"__{base_col}")]
                    self._temperature_group_index_cache[target_key] = target_indices
                gap_indices = self._temperature_group_index_cache.get(gap_key)
                if gap_indices is None:
                    gap_indices = [idx for idx, col in enumerate(gap_cols) if col.endswith(f"__{base_col}")]
                    self._temperature_group_index_cache[gap_key] = gap_indices
                if not target_indices or not gap_indices:
                    return None
                target_values = target_weather[:, target_indices].astype(float, copy=False)
                gap_values = gap_weather[:, gap_indices].astype(float, copy=False)
                if np.isnan(target_values).any() or np.isnan(gap_values).any():
                    return None
                cached = {
                    "target_mean": np.mean(target_values, axis=1),
                    "target_std": np.std(target_values, axis=1),
                    "target_min": np.min(target_values, axis=1),
                    "target_max": np.max(target_values, axis=1),
                    "gap_mean": np.mean(gap_values, axis=1),
                    "gap_std": np.std(gap_values, axis=1),
                    "gap_min": np.min(gap_values, axis=1),
                    "gap_max": np.max(gap_values, axis=1),
                }
                self._temperature_group_stats_cache[cache_key] = cached
            if cached is None:
                return None

            stats[f"{base_col}_target_station_mean"] = float(cached["target_mean"][slot_index])
            stats[f"{base_col}_target_station_std"] = float(cached["target_std"][slot_index])
            stats[f"{base_col}_target_station_min"] = float(cached["target_min"][slot_index])
            stats[f"{base_col}_target_station_max"] = float(cached["target_max"][slot_index])
            stats[f"{base_col}_gap_station_mean"] = float(cached["gap_mean"][slot_index])
            stats[f"{base_col}_gap_station_std"] = float(cached["gap_std"][slot_index])
            stats[f"{base_col}_gap_station_min"] = float(cached["gap_min"][slot_index])
            stats[f"{base_col}_gap_station_max"] = float(cached["gap_max"][slot_index])
            stats[f"{base_col}_target_gap_station_mean_delta"] = float(
                stats[f"{base_col}_target_station_mean"] - stats[f"{base_col}_gap_station_mean"]
            )
            stats[f"{base_col}_target_gap_station_std_delta"] = float(
                stats[f"{base_col}_target_station_std"] - stats[f"{base_col}_gap_station_std"]
            )
        return stats

    def _temperature_daily_summary(
        self,
        *,
        merged,
        target_day: date,
        prefix: str,
        base_col: str,
    ) -> dict[str, np.ndarray | float] | None:
        cache_key = (target_day.isoformat(), prefix, base_col)
        cached = self._temperature_daily_summary_cache.get(cache_key)
        if cache_key in self._temperature_daily_summary_cache:
            return cached

        matrix = merged.day_feature_matrix(target_day, prefix)
        cols = merged.prefix_feature_columns(prefix)
        if matrix is None or not cols:
            self._temperature_daily_summary_cache[cache_key] = None
            return None

        index_key = (prefix, base_col)
        indices = self._temperature_group_index_cache.get(index_key)
        if indices is None:
            indices = [idx for idx, col in enumerate(cols) if col.endswith(f"__{base_col}")]
            self._temperature_group_index_cache[index_key] = indices
        if not indices:
            self._temperature_daily_summary_cache[cache_key] = None
            return None

        values = matrix[:, indices].astype(float, copy=False)
        if np.isnan(values).any():
            self._temperature_daily_summary_cache[cache_key] = None
            return None

        slot_mean = np.mean(values, axis=1)
        summary: dict[str, np.ndarray | float] = {
            "slot_mean": slot_mean,
            "daily_mean": float(np.mean(slot_mean)),
            "daily_std": float(np.std(slot_mean)),
            "daily_min": float(np.min(slot_mean)),
            "daily_max": float(np.max(slot_mean)),
            "daily_range": float(np.max(slot_mean) - np.min(slot_mean)),
        }
        self._temperature_daily_summary_cache[cache_key] = summary
        return summary

    def _temperature_streak_days(
        self,
        *,
        merged,
        origin_day: date,
        base_col: str,
        streak_kind: str,
    ) -> int | None:
        cache_key = (origin_day.isoformat(), base_col, streak_kind)
        if cache_key in self._temperature_streak_cache:
            return self._temperature_streak_cache[cache_key]

        streak = 0
        for offset in range(self.temperature_fix_lookback_days):
            current_day = origin_day - timedelta(days=offset)
            summary = self._temperature_daily_summary(
                merged=merged,
                target_day=current_day,
                prefix="D_0__",
                base_col=base_col,
            )
            if summary is None:
                if offset == 0:
                    self._temperature_streak_cache[cache_key] = None
                    return None
                break
            if streak_kind == "cold":
                if float(summary["daily_min"]) < self.temperature_fix_cold_threshold_c:
                    streak += 1
                else:
                    break
            elif streak_kind == "hot":
                if float(summary["daily_max"]) > self.temperature_fix_hot_threshold_c:
                    streak += 1
                else:
                    break
            else:
                raise ValueError(f"unsupported streak_kind: {streak_kind}")

        self._temperature_streak_cache[cache_key] = streak
        return streak

    def _temperature_fix_adjust(self, temperature_c: float, *, streak_days: int, kind: str) -> float:
        temperature_c = float(temperature_c)
        if streak_days <= 0:
            return temperature_c
        if kind == "cold":
            if temperature_c > self.temperature_fix_cold_ref_c:
                return temperature_c
            return temperature_c - (self.temperature_fix_cold_ref_c - temperature_c) * (
                self.temperature_fix_decay**streak_days
            )
        if kind == "hot":
            if temperature_c < self.temperature_fix_hot_ref_c:
                return temperature_c
            return temperature_c + (temperature_c - self.temperature_fix_hot_ref_c) * (
                self.temperature_fix_decay**streak_days
            )
        raise ValueError(f"unsupported temperature fix kind: {kind}")

    def _temperature_fix_trailing_mean(
        self,
        *,
        merged,
        origin_day: date,
        base_col: str,
        days: int,
    ) -> float | None:
        """Mean daily-mean temperature over the trailing `days` origin-side days.

        Genuinely-new info vs the existing tempfix features: those give the
        target day's ABSOLUTE level, the 1-day gap delta, and the cold/hot
        STREAK (persistence). None encodes *how far today's temperature departs
        from the recent multi-day thermal baseline* — the behavioural-adaptation
        signal (AC/heating uptake lags the change vs the recent norm). Uses only
        origin-side D_0__ history (protocol-safe), memoised per (origin, col, days).
        """
        cache_key = (origin_day.isoformat(), base_col, days)
        if cache_key in self._temperature_trailing_mean_cache:
            return self._temperature_trailing_mean_cache[cache_key]
        means: list[float] = []
        for offset in range(1, days + 1):
            past_day = origin_day - timedelta(days=offset)
            summary = self._temperature_daily_summary(
                merged=merged,
                target_day=past_day,
                prefix="D_0__",
                base_col=base_col,
            )
            if summary is None:
                # Need at least the immediately-prior day for a non-trivial baseline.
                if offset == 1:
                    self._temperature_trailing_mean_cache[cache_key] = None
                    return None
                break
            means.append(float(summary["daily_mean"]))
        if not means:
            self._temperature_trailing_mean_cache[cache_key] = None
            return None
        val = float(np.mean(means))
        self._temperature_trailing_mean_cache[cache_key] = val
        return val

    def _temperature_fix_features(
        self,
        *,
        merged,
        origin_day: date,
        target_day: date,
        dayplus: int,
        slot_index: int,
        feature_row: dict,
    ) -> dict[str, float] | None:
        base_cols = self.temperature_fix_base_cols or self.temperature_group_base_cols
        if not base_cols:
            return {}

        if self.weather_reference_mode == "fixed_d1_control":
            target_prefix = "D_1__"
        else:
            target_prefix = f"D_{dayplus}__"

        stats: dict[str, float] = {}
        for base_col in base_cols:
            target_summary = self._temperature_daily_summary(
                merged=merged,
                target_day=target_day,
                prefix=target_prefix,
                base_col=base_col,
            )
            gap_summary = self._temperature_daily_summary(
                merged=merged,
                target_day=origin_day,
                prefix="D_0__",
                base_col=base_col,
            )
            cold_streak_days = self._temperature_streak_days(
                merged=merged,
                origin_day=origin_day,
                base_col=base_col,
                streak_kind="cold",
            )
            hot_streak_days = self._temperature_streak_days(
                merged=merged,
                origin_day=origin_day,
                base_col=base_col,
                streak_kind="hot",
            )
            if (
                target_summary is None
                or gap_summary is None
                or cold_streak_days is None
                or hot_streak_days is None
            ):
                return None

            slot_temp = float(
                feature_row.get(f"{base_col}_target_station_mean", target_summary["slot_mean"][slot_index])
            )
            gap_slot_temp = float(
                feature_row.get(f"{base_col}_gap_station_mean", gap_summary["slot_mean"][slot_index])
            )
            cold_adjusted = self._temperature_fix_adjust(slot_temp, streak_days=cold_streak_days, kind="cold")
            hot_adjusted = self._temperature_fix_adjust(slot_temp, streak_days=hot_streak_days, kind="hot")
            prefix_name = f"tempfix_{base_col}"

            if self.enable_temperature_fix_anomaly:
                # IDEA-044: target-day thermal DEPARTURE from the recent trailing
                # baseline (genuinely-new — not the absolute level, not the 1-day
                # gap delta, not the streak persistence). Captures behavioural
                # load adaptation to *change vs the recent norm* (heating/cooling
                # uptake lags the anomaly). Origin-side baseline -> protocol-safe.
                trailing_mean = self._temperature_fix_trailing_mean(
                    merged=merged,
                    origin_day=origin_day,
                    base_col=base_col,
                    days=self.temperature_fix_anomaly_days,
                )
                if trailing_mean is None:
                    return None
                stats[f"{prefix_name}_target_anom_{self.temperature_fix_anomaly_days}d"] = (
                    float(target_summary["daily_mean"]) - trailing_mean
                )

            if self.enable_hdd_cdd_features:
                # Degree-day V-split: heating/cooling uptake ramps from a comfort
                # threshold (local U-curve min 12-16C), so hdd/cdd carry the
                # thermostat-switch signal a linear temperature column cannot.
                # Level AND 7-day delta arrive as a pair: the delta isolates the
                # behavioural adaptation (AC switched on this week vs last week)
                # from the seasonal level the model already knows via doy.
                t_ref = self.hdd_cdd_reference_temp
                target_daily_mean = float(target_summary["daily_mean"])
                gap_daily_mean = float(gap_summary["daily_mean"])
                target_hdd = max(0.0, t_ref - target_daily_mean)
                target_cdd = max(0.0, target_daily_mean - t_ref)
                gap_hdd = max(0.0, t_ref - gap_daily_mean)
                gap_cdd = max(0.0, gap_daily_mean - t_ref)
                stats[f"{prefix_name}_target_hdd"] = target_hdd
                stats[f"{prefix_name}_target_cdd"] = target_cdd
                stats[f"{prefix_name}_gap_hdd"] = gap_hdd
                stats[f"{prefix_name}_gap_cdd"] = gap_cdd
                stats[f"{prefix_name}_target_hdd_gap_delta"] = target_hdd - gap_hdd
                stats[f"{prefix_name}_target_cdd_gap_delta"] = target_cdd - gap_cdd
                stats[f"{prefix_name}_slot_hdd"] = max(0.0, t_ref - slot_temp)
                stats[f"{prefix_name}_slot_cdd"] = max(0.0, slot_temp - t_ref)
                # Hour-regime gates: iter-2 showed the V-split signal lands at
                # night; AC/cooling MW concentrates at specific hours, so gate
                # the slot degree-days by the regime where they bite (afternoon
                # cooling, evening-peak cooling, morning-shoulder heating).
                slot_hour = slot_index // 4  # interval-ending 15-min grid: slot i -> hour i//4
                slot_cdd_val = max(0.0, slot_temp - t_ref)
                slot_hdd_val = max(0.0, t_ref - slot_temp)
                stats[f"{prefix_name}_cdd_x_afternoon"] = slot_cdd_val if 12 <= slot_hour < 18 else 0.0
                stats[f"{prefix_name}_cdd_x_evening_peak"] = slot_cdd_val if 18 <= slot_hour < 22 else 0.0
                stats[f"{prefix_name}_hdd_x_morning_shoulder"] = slot_hdd_val if 6 <= slot_hour < 10 else 0.0
                hdd_trailing_mean = self._temperature_fix_trailing_mean(
                    merged=merged,
                    origin_day=origin_day,
                    base_col=base_col,
                    days=self.hdd_cdd_week_days,
                )
                if hdd_trailing_mean is None:
                    return None
                trailing_hdd = max(0.0, t_ref - hdd_trailing_mean)
                trailing_cdd = max(0.0, hdd_trailing_mean - t_ref)
                stats[f"{prefix_name}_target_hdd_week_delta"] = target_hdd - trailing_hdd
                stats[f"{prefix_name}_target_cdd_week_delta"] = target_cdd - trailing_cdd

            if self.enable_temperature_fix_daily_summary:
                if self.enable_temperature_fix_daily_core_stats:
                    stats[f"{prefix_name}_target_daily_mean"] = float(target_summary["daily_mean"])
                    stats[f"{prefix_name}_target_daily_std"] = float(target_summary["daily_std"])
                    stats[f"{prefix_name}_target_daily_min"] = float(target_summary["daily_min"])
                    stats[f"{prefix_name}_target_daily_max"] = float(target_summary["daily_max"])
                    stats[f"{prefix_name}_target_daily_range"] = float(target_summary["daily_range"])
                    stats[f"{prefix_name}_gap_daily_mean"] = float(gap_summary["daily_mean"])
                    stats[f"{prefix_name}_gap_daily_std"] = float(gap_summary["daily_std"])
                    stats[f"{prefix_name}_gap_daily_min"] = float(gap_summary["daily_min"])
                    stats[f"{prefix_name}_gap_daily_max"] = float(gap_summary["daily_max"])
                    stats[f"{prefix_name}_gap_daily_range"] = float(gap_summary["daily_range"])
                if self.enable_temperature_fix_daily_slot_delta:
                    stats[f"{prefix_name}_slot_minus_target_daily_mean"] = slot_temp - float(target_summary["daily_mean"])
                    stats[f"{prefix_name}_slot_minus_target_daily_min"] = slot_temp - float(target_summary["daily_min"])
                    stats[f"{prefix_name}_slot_minus_target_daily_max"] = slot_temp - float(target_summary["daily_max"])
                    stats[f"{prefix_name}_gap_slot_minus_gap_daily_mean"] = gap_slot_temp - float(gap_summary["daily_mean"])
                if self.enable_temperature_fix_daily_gap_delta:
                    stats[f"{prefix_name}_target_gap_daily_mean_delta"] = float(target_summary["daily_mean"]) - float(
                        gap_summary["daily_mean"]
                    )
                    stats[f"{prefix_name}_target_gap_daily_min_delta"] = float(target_summary["daily_min"]) - float(
                        gap_summary["daily_min"]
                    )
                    stats[f"{prefix_name}_target_gap_daily_max_delta"] = float(target_summary["daily_max"]) - float(
                        gap_summary["daily_max"]
                    )
                if self.enable_temperature_fix_daily_flags:
                    stats[f"{prefix_name}_target_is_cold_day"] = float(
                        float(target_summary["daily_min"]) < self.temperature_fix_cold_threshold_c
                    )
                    stats[f"{prefix_name}_target_is_hot_day"] = float(
                        float(target_summary["daily_max"]) > self.temperature_fix_hot_threshold_c
                    )
                if self.enable_temperature_fix_daily_bins:
                    stats[f"{prefix_name}_target_slot_bin5"] = float(self._temperature_bin_5c(slot_temp))
                    stats[f"{prefix_name}_target_daily_min_bin5"] = float(
                        self._temperature_bin_5c(float(target_summary["daily_min"]))
                    )
                    stats[f"{prefix_name}_target_daily_max_bin5"] = float(
                        self._temperature_bin_5c(float(target_summary["daily_max"]))
                    )

            if self.enable_temperature_fix_streak_adjust:
                stats[f"{prefix_name}_cold_streak_days"] = float(cold_streak_days)
                stats[f"{prefix_name}_hot_streak_days"] = float(hot_streak_days)
                stats[f"{prefix_name}_cold_adjusted_slot"] = cold_adjusted
                stats[f"{prefix_name}_hot_adjusted_slot"] = hot_adjusted
                stats[f"{prefix_name}_cold_adjust_delta"] = cold_adjusted - slot_temp
                stats[f"{prefix_name}_hot_adjust_delta"] = hot_adjusted - slot_temp

        return stats

    def _load_reference_days(self, origin_day: date, target_day: date, lag: int) -> tuple[date, date]:
        if self.load_reference_mode == "origin":
            gap_day = origin_day - timedelta(days=1)
            lag_day = origin_day - timedelta(days=lag)
            return gap_day, lag_day
        if self.load_reference_mode == "target_recent_control":
            gap_day = target_day - timedelta(days=self.control_load_gap_days)
            lag_day = target_day - timedelta(days=self.control_load_gap_days + lag - 1)
            return gap_day, lag_day
        raise ValueError(f"unsupported load_reference_mode: {self.load_reference_mode}")

    @staticmethod
    def _limit_origin_days(origin_days: list[date], segment_range: tuple[int, int], max_train_rows: int | None) -> list[date]:
        if max_train_rows is None or not origin_days:
            return origin_days
        rows_per_origin = max(1, (segment_range[1] - segment_range[0] + 1) * 96)
        max_origin_days = max(1, int(np.ceil(max_train_rows / rows_per_origin)))
        if len(origin_days) <= max_origin_days:
            return origin_days
        idx = np.linspace(0, len(origin_days) - 1, max_origin_days).astype(int)
        return [origin_days[i] for i in idx]

    def _row_for_point(self, merged, origin_day: date, dayplus: int, slot_index: int) -> dict | None:
        target_day = origin_day + timedelta(days=dayplus)
        actual_load = merged.day_load(target_day)
        target_weather = self._target_weather_matrix(merged, target_day, dayplus)
        gap_weather = merged.day_feature_matrix(origin_day, "D_0__")
        if actual_load is None or target_weather is None or gap_weather is None:
            return None

        feature_row = {
            "origin_ordinal": pd.Timestamp(origin_day).toordinal(),
            "target_ordinal": pd.Timestamp(target_day).toordinal(),
            "dayplus": int(dayplus),
            "slot_index": int(slot_index),
            "hour": int(slot_index // 4),
            "quarter": int(slot_index % 4),
            "dow": int(pd.Timestamp(target_day).dayofweek),
            "month": int(pd.Timestamp(target_day).month),
            "is_weekend": int(pd.Timestamp(target_day).dayofweek >= 5),
            "origin_month": int(pd.Timestamp(origin_day).month),
            "origin_dow": int(pd.Timestamp(origin_day).dayofweek),
        }
        hour_val = feature_row["hour"]
        doy_val = int(pd.Timestamp(target_day).timetuple().tm_yday)
        feature_row["hour_sin"] = float(np.sin(2.0 * np.pi * hour_val / 24.0))
        feature_row["hour_cos"] = float(np.cos(2.0 * np.pi * hour_val / 24.0))
        feature_row["doy_sin"] = float(np.sin(2.0 * np.pi * doy_val / 365.0))
        feature_row["doy_cos"] = float(np.cos(2.0 * np.pi * doy_val / 365.0))
        dow_val = feature_row["dow"]
        feature_row["dow_sin"] = float(np.sin(2.0 * np.pi * dow_val / 7.0))
        feature_row["dow_cos"] = float(np.cos(2.0 * np.pi * dow_val / 7.0))
        if self.enable_holiday_features:
            # Hour-gated holiday flag (IDEA-042). iter-15 proved a *global* holiday
            # flag is RIGHT at night (industrial demand-drop) but WRONG at midday
            # (PV-saturated net load: solar swamps the demand-drop). So the flag
            # only fires outside the solar window [8, 18) -- the demand-dominated
            # evening/night/pre-dawn regime where the cessation signal is clean.
            is_demand_regime = hour_val < 8 or hour_val >= 18
            feature_row["is_holiday"] = 1 if (target_day in _CN_HOLIDAY_DAYS and is_demand_regime) else 0
            if self.enable_holiday_v2_features:
                # Holiday v2 type split: major (SF/Labour/ND, -18..-32% daily mean)
                # vs minor (Qingming/Dragon-Boat/Mid-Autumn, -3..-8%). Same hour
                # gate: the flag only carries information in the demand regime.
                holiday_type = _CN_HOLIDAY_DAY_TYPES.get(target_day)
                feature_row["is_major_holiday"] = 1 if (holiday_type == "major" and is_demand_regime) else 0
                feature_row["is_minor_holiday"] = 1 if (holiday_type == "minor" and is_demand_regime) else 0

        lag_values = []
        for lag in self.lag_days:
            _, lag_day = self._load_reference_days(origin_day, target_day, lag)
            lag_load = merged.day_load(lag_day)
            if lag_load is None:
                return None
            value = float(lag_load[slot_index])
            feature_row[f"lag_load_d{lag}"] = value
            lag_values.append(value)

        gap_day, _ = self._load_reference_days(origin_day, target_day, 1)
        gap_load = merged.day_load(gap_day)
        if gap_load is None:
            return None
        feature_row["lag_mean"] = float(np.mean(lag_values))
        feature_row["lag_std"] = float(np.std(lag_values))
        feature_row["lag_min"] = float(np.min(lag_values))
        feature_row["lag_max"] = float(np.max(lag_values))
        feature_row["gap_load_same_slot"] = float(gap_load[slot_index])
        feature_row["lag_last_delta"] = float(lag_values[0] - lag_values[-1])
        feature_row["gap_vs_lag_mean"] = float(feature_row["gap_load_same_slot"] - feature_row["lag_mean"])
        if self.enable_anchor_holiday_features:
            # IDEA-067 (round 5): holiday contamination of the load anchors,
            # HOUR-GATED to the demand regime (row hour < 8 or >= 18).
            # Mechanism: gap/lag anchor levels sit -3..-8% (minor) /
            # -18..-32% (major) below the weekday norm on holiday days, and the
            # tree needs the attribution flag to discount the anchor instead of
            # reading the drop as a regime shift. The gate is the iter-15
            # lesson applied to the anchor side: the depression is a DEMAND
            # effect, while midday anchor slots are PV-dominated net load where
            # a demand-interpreted discount corrupts a solar-noise anchor
            # (iter-38's un-gated variant paid medium midday -0.065 and full
            # -0.022 exactly there).
            anchor_demand_regime = hour_val < 8 or hour_val >= 18
            gap_hol_type = _CN_HOLIDAY_DAY_TYPES.get(gap_day)
            feature_row["gap_is_major_holiday"] = int(
                gap_hol_type == "major" and anchor_demand_regime
            )
            feature_row["gap_is_minor_holiday"] = int(
                gap_hol_type == "minor" and anchor_demand_regime
            )
            feature_row["lag_holiday_days"] = int(
                sum(
                    1
                    for k in range(1, len(self.lag_days) + 1)
                    if (origin_day - timedelta(days=k)) in _CN_HOLIDAY_DAYS
                )
                * anchor_demand_regime
            )
        if self.enable_holiday_position_features:
            # IDEA-068 (round 5): the holiday axis beyond a binary --
            # (a) intra-range ramp position (target side), (b) restart-snap
            # days-after markers (target AND gap sides). Both are DEMAND-side
            # effects, so both carry the iter-15/39 demand-regime gate:
            # outside [8, 18) they carry information; on PV-saturated midday
            # rows they take their no-information sentinel values.
            pos_regime = hour_val < 8 or hour_val >= 18
            feature_row["target_holiday_pos_norm"] = (
                _CN_HOLIDAY_POS_NORM.get(target_day, 0.0) if pos_regime else 0.0
            )
            feature_row["target_days_after_holiday"] = float(
                _CN_DAYS_AFTER_HOLIDAY_RUN.get(target_day, 7) if pos_regime else 7
            )
            feature_row["gap_days_after_holiday"] = float(
                _CN_DAYS_AFTER_HOLIDAY_RUN.get(gap_day, 7) if pos_regime else 7
            )
        hour_sin_val = float(np.sin(2.0 * np.pi * (slot_index // 4) / 24.0))
        hour_cos_val = float(np.cos(2.0 * np.pi * (slot_index // 4) / 24.0))
        feature_row["lag_load_d1_x_hour_sin"] = float(lag_values[0]) * hour_sin_val
        feature_row["lag_load_d1_x_hour_cos"] = float(lag_values[0]) * hour_cos_val
        feature_row["gap_load_x_hour_sin"] = float(gap_load[slot_index]) * hour_sin_val
        feature_row["gap_load_x_hour_cos"] = float(gap_load[slot_index]) * hour_cos_val

        if self.enable_rolling_stats:
            for window in self.rolling_windows:
                roll_values = self._rolling_feature_values(merged, origin_day, slot_index, int(window))
                if roll_values is None:
                    return None
                feature_row[f"roll_mean_{int(window)}"] = self._safe_mean(roll_values)
                feature_row[f"roll_std_{int(window)}"] = self._safe_std(roll_values)
                feature_row[f"gap_vs_roll_mean_{int(window)}"] = float(
                    feature_row["gap_load_same_slot"] - feature_row[f"roll_mean_{int(window)}"]
                )

        if self.enable_dayplus_calendar_interactions:
            feature_row["dayplus_x_month"] = int(dayplus) * int(feature_row["month"])
            feature_row["dayplus_x_dow"] = int(dayplus) * int(feature_row["dow"])
            feature_row["dayplus_x_is_weekend"] = int(dayplus) * int(feature_row["is_weekend"])
            feature_row["dayplus_x_hour"] = int(dayplus) * int(feature_row["hour"])

        if self.enable_dayplus_lag_interactions:
            for lag in self.lag_days:
                feature_row[f"dayplus_x_lag_load_d{lag}"] = int(dayplus) * float(feature_row[f"lag_load_d{lag}"])
            feature_row["dayplus_x_gap_load_same_slot"] = int(dayplus) * float(feature_row["gap_load_same_slot"])
            feature_row["dayplus_x_lag_mean"] = int(dayplus) * float(feature_row["lag_mean"])

        for idx, value in enumerate(gap_weather[slot_index], start=1):
            feature_row[f"gap_w_{idx:03d}"] = float(value)
        for idx, value in enumerate(target_weather[slot_index], start=1):
            feature_row[f"fut_w_{idx:03d}"] = float(value)
            if self.enable_weather_delta_features:
                gap_value = float(gap_weather[slot_index][idx - 1])
                fut_value = float(value)
                feature_row[f"delta_w_{idx:03d}"] = fut_value - gap_value
                feature_row[f"ratio_w_{idx:03d}"] = fut_value / gap_value if abs(gap_value) > 1e-8 else 0.0

        if self.enable_apparent_temp_hour_interaction:
            target_prefix_app = f"D_{dayplus}__" if self.weather_reference_mode == "dayplus" else "D_1__"
            apparent_idx = self._apparent_temp_indices(merged, target_prefix_app)
            if apparent_idx:
                apparent_slot = float(np.mean([target_weather[slot_index][i] for i in apparent_idx]))
                feature_row["apparent_temp_target_x_hour_sin"] = apparent_slot * hour_sin_val
                feature_row["apparent_temp_target_x_hour_cos"] = apparent_slot * hour_cos_val
                is_midday_scored = 1 if 10 <= hour_val < 15 else 0
                feature_row["apparent_temp_target_x_midday_scored"] = apparent_slot * is_midday_scored

        if self.enable_temperature_group_features:
            grouped = self._temperature_group_features(
                merged=merged,
                origin_day=origin_day,
                target_day=target_day,
                dayplus=dayplus,
                slot_index=slot_index,
                target_weather=target_weather,
                gap_weather=gap_weather,
            )
            if grouped is None:
                return None
            feature_row.update(grouped)

        if self.enable_horizon_bucket_features:
            horizon_bucket = self._horizon_bucket(int(dayplus))
            feature_row["horizon_bucket"] = int(horizon_bucket)
            feature_row["is_head_horizon"] = int(horizon_bucket == 1)
            feature_row["is_mid_horizon"] = int(horizon_bucket == 2)
            feature_row["is_tail_horizon"] = int(horizon_bucket == 3)

        if self.enable_monthstart_regime_features:
            day_of_month = int(pd.Timestamp(target_day).day)
            feature_row["target_day_of_month"] = day_of_month
            feature_row["is_month_start_window"] = int(day_of_month <= 3)
            feature_row["origin_to_target_cross_month"] = int(
                pd.Timestamp(origin_day).month != pd.Timestamp(target_day).month
            )
            feature_row["origin_day_of_month"] = int(pd.Timestamp(origin_day).day)
            feature_row["target_week_of_month"] = int((day_of_month - 1) // 7 + 1)
            is_first_3 = int(day_of_month <= 3)
            hour_now = int(slot_index // 4)
            is_morning_ramp = int(6 <= hour_now < 10)
            is_midday_scored = int(10 <= hour_now < 15)
            is_evening_peak = int(18 <= hour_now < 21)
            feature_row["is_first_3_x_morning_ramp"] = is_first_3 * is_morning_ramp
            feature_row["is_first_3_x_midday_scored"] = is_first_3 * is_midday_scored
            feature_row["is_first_3_x_evening_peak"] = is_first_3 * is_evening_peak

        if self.enable_monthstart_gated_features:
            # IDEA-053 (round 4): iter-16 decomposition showed the month-boundary
            # family is REGIME-ASYMMETRIC -- removing it gained the June/summer
            # steady-state frame (night +0.30 medium) but cost the transition
            # months on the full profile (midday -0.28): industrial
            # production/billing-cycle resets at month boundaries genuinely shift
            # load in Sep/Dec/Mar-type months, while mid-summer regimes run
            # steady. Re-introduce the flags gated OFF for summer targets
            # (month in 6/7/8) as separate sparse columns, so the summer frame
            # keeps its monthstart-OFF specialisation and transition months
            # recover the scheduling markers at zero summer-row cost.
            tgt_month = pd.Timestamp(target_day).month
            ms_gate = int(tgt_month not in (6, 7, 8))
            is_first_3_g = int(int(pd.Timestamp(target_day).day) <= 3) * ms_gate
            hour_now = int(slot_index // 4)
            feature_row["ms_trans_is_first_3"] = is_first_3_g
            feature_row["ms_trans_x_morning_ramp"] = is_first_3_g * int(6 <= hour_now < 10)
            feature_row["ms_trans_x_midday_scored"] = is_first_3_g * int(10 <= hour_now < 15)
            feature_row["ms_trans_x_evening_peak"] = is_first_3_g * int(18 <= hour_now < 21)
            # IDEA-059 (iter-34): complete the family's hour-axis with the night
            # member (the removed ungated family had monthstart_x_is_night_regime;
            # the gated re-introduction shipped morning/midday/evening only).
            # Same iter-29 mechanism: sparse, constant-0 on summer eval rows.
            feature_row["ms_trans_x_is_night_regime"] = is_first_3_g * int(hour_now < 7)
            feature_row["ms_trans_cross_month"] = int(
                pd.Timestamp(origin_day).month != tgt_month
            ) * ms_gate

        if self.enable_lag_regime_features:
            feature_row["lag_cv"] = float(feature_row["lag_std"] / (abs(feature_row["lag_mean"]) + 1e-8))
            feature_row["lag_range"] = float(feature_row["lag_max"] - feature_row["lag_min"])
            feature_row["origin_is_monday"] = int(pd.Timestamp(origin_day).dayofweek == 0)

        if self.enable_day_night_regime_features:
            hour = int(feature_row["hour"])
            is_midday = int(10 <= hour < 16)
            is_evening = int(18 <= hour < 22)
            is_night = int(hour < 7)
            feature_row["is_midday_regime"] = is_midday
            feature_row["is_evening_regime"] = is_evening
            feature_row["is_night_regime"] = is_night
            feature_row["dayplus_x_is_night_regime"] = int(dayplus) * is_night
            feature_row["dayplus_x_is_evening_regime"] = int(dayplus) * is_evening
            feature_row["cross_month_x_is_night_regime"] = int(
                feature_row.get("origin_to_target_cross_month", 0)
            ) * is_night
            feature_row["monthstart_x_is_night_regime"] = int(
                feature_row.get("is_month_start_window", 0)
            ) * is_night

        if self.enable_temperature_fix_features:
            tempfix = self._temperature_fix_features(
                merged=merged,
                origin_day=origin_day,
                target_day=target_day,
                dayplus=dayplus,
                slot_index=slot_index,
                feature_row=feature_row,
            )
            if tempfix is None:
                return None
            feature_row.update(tempfix)

        feature_row["target"] = float(actual_load[slot_index])
        feature_row["timestamp"] = pd.Timestamp(target_day) + pd.Timedelta(minutes=15 * slot_index)

        if self.future_dpv_store is not None:
            feature_row["future_dpv_mw"] = self.future_dpv_store.value(
                origin_day=origin_day,
                target_day=target_day,
                dayplus=dayplus,
                timestamp=feature_row["timestamp"],
            )

        return feature_row

    def _frame_from_origins(
        self,
        merged,
        origin_days: list[date],
        segment_range: tuple[int, int],
        required_dayplus_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        required_range = required_dayplus_range or segment_range
        if self.future_dpv_store is not None:
            self.future_dpv_store.assert_coverage(
                origin_days=origin_days,
                required_dayplus_range=required_range,
                scope="frame_build",
            )
        rows = []
        for origin_day in origin_days:
            for dayplus in range(required_range[0], required_range[1] + 1):
                for slot_index in range(96):
                    row = self._row_for_point(merged, origin_day, dayplus, slot_index)
                    if row is not None:
                        rows.append(row)
        if not rows:
            raise ValueError("no valid rows built for LGB model")
        frame = pd.DataFrame(rows)
        frame = frame.sort_values(["timestamp", "dayplus", "slot_index"]).reset_index(drop=True)
        return frame

    def _selected_feature_names(self, X_train: pd.DataFrame, y_train: pd.Series) -> list[str]:
        enabled = bool(self.feature_selection_cfg.get("enabled", False))
        if not enabled:
            return list(X_train.columns)

        method = str(self.feature_selection_cfg.get("method", "lgb_importance"))
        top_k = int(self.feature_selection_cfg.get("top_k", 0))
        sample_rows = int(self.feature_selection_cfg.get("sample_rows", 0))
        protect_prefixes = [str(item) for item in self.feature_selection_cfg.get("protect_prefixes", [])]

        selector_X = X_train
        selector_y = y_train
        if sample_rows > 0 and len(selector_X) > sample_rows:
            selector_X = selector_X.iloc[:sample_rows].copy()
            selector_y = selector_y.iloc[:sample_rows].copy()

        protected = [
            col
            for col in X_train.columns
            if any(col == prefix or col.startswith(prefix) for prefix in protect_prefixes)
        ]

        if method == "lgb_importance":
            import lightgbm as lgb

            selector = lgb.LGBMRegressor(
                n_estimators=int(self.feature_selection_cfg.get("n_estimators", 80)),
                learning_rate=float(self.model_cfg.get("learning_rate", 0.05)),
                num_leaves=int(min(int(self.model_cfg.get("num_leaves", 63)), 31)),
                max_depth=int(self.model_cfg.get("max_depth", -1)),
                min_child_samples=int(self.model_cfg.get("min_child_samples", 40)),
                subsample=float(self.model_cfg.get("subsample", 0.9)),
                colsample_bytree=float(self.model_cfg.get("colsample_bytree", 0.8)),
                reg_alpha=float(self.model_cfg.get("reg_alpha", 0.0)),
                reg_lambda=float(self.model_cfg.get("reg_lambda", 0.1)),
                random_state=int(self.model_cfg.get("random_state", 3407)),
                n_jobs=int(self.model_cfg.get("n_jobs", 8)),
            )
            selector.fit(selector_X, selector_y)
            importances = pd.Series(selector.feature_importances_, index=selector_X.columns, dtype=float)
            ranked = list(importances.sort_values(ascending=False).index)
        else:
            raise ValueError(f"unsupported feature_selection method: {method}")

        if top_k <= 0 or top_k >= len(ranked):
            chosen = ranked
        else:
            chosen = ranked[:top_k]

        final = list(dict.fromkeys(protected + chosen))
        return [col for col in X_train.columns if col in final]

    def _sample_weights(self, frame: pd.DataFrame) -> np.ndarray | None:
        if not bool(self.sample_weight_cfg.get("enabled", False)):
            return None
        weights = np.ones(len(frame), dtype=np.float32)
        hour_values = frame["hour"].to_numpy(dtype=int)
        dayplus_values = frame["dayplus"].to_numpy(dtype=int)

        hour_rules = self.sample_weight_cfg.get("hour_rules", []) or []
        for rule in hour_rules:
            start_hour = int(rule.get("start_hour", 0))
            end_hour = int(rule.get("end_hour", 24))
            weight = float(rule.get("weight", 1.0))
            mask = (hour_values >= start_hour) & (hour_values < end_hour)
            weights[mask] *= weight

        dayplus_weights = self.sample_weight_cfg.get("dayplus_weights")
        if dayplus_weights:
            for idx, weight in enumerate(dayplus_weights, start=1):
                mask = dayplus_values == idx
                weights[mask] *= float(weight)

        horizon_buckets = self.sample_weight_cfg.get("horizon_bucket_weights", {}) or {}
        for bucket_name, weight in horizon_buckets.items():
            if str(bucket_name) == "head":
                mask = dayplus_values <= 5
            elif str(bucket_name) == "mid":
                mask = (dayplus_values >= 6) & (dayplus_values <= 10)
            elif str(bucket_name) == "tail":
                mask = dayplus_values >= 11
            else:
                continue
            weights[mask] *= float(weight)

        column_rules = self.sample_weight_cfg.get("column_rules", []) or []
        for rule in column_rules:
            weight = float(rule.get("weight", 1.0))
            mask = self._sample_weight_rule_mask(frame, rule)
            weights[mask] *= weight

        return weights

    def _sample_weight_rule_mask(self, frame: pd.DataFrame, rule: dict) -> np.ndarray:
        if "all" in rule:
            masks = [self._sample_weight_rule_mask(frame, sub_rule) for sub_rule in rule.get("all", [])]
            if not masks:
                return np.ones(len(frame), dtype=bool)
            mask = masks[0].copy()
            for sub_mask in masks[1:]:
                mask &= sub_mask
            return mask

        if "any" in rule:
            masks = [self._sample_weight_rule_mask(frame, sub_rule) for sub_rule in rule.get("any", [])]
            if not masks:
                return np.zeros(len(frame), dtype=bool)
            mask = masks[0].copy()
            for sub_mask in masks[1:]:
                mask |= sub_mask
            return mask

        column = str(rule.get("column", "")).strip()
        if not column:
            raise ValueError(f"sample_weight column rule is missing column: {rule}")
        if column not in frame.columns:
            raise KeyError(f"sample_weight column rule references missing column: {column}")

        series = frame[column]
        mask = np.ones(len(frame), dtype=bool)

        if "equals" in rule:
            mask &= (series == rule["equals"]).to_numpy(dtype=bool)
        if "not_equals" in rule:
            mask &= (series != rule["not_equals"]).to_numpy(dtype=bool)
        if "in" in rule:
            mask &= series.isin(rule["in"]).to_numpy(dtype=bool)
        if "not_in" in rule:
            mask &= (~series.isin(rule["not_in"])).to_numpy(dtype=bool)
        if "gte" in rule:
            mask &= (series >= rule["gte"]).to_numpy(dtype=bool)
        if "gt" in rule:
            mask &= (series > rule["gt"]).to_numpy(dtype=bool)
        if "lte" in rule:
            mask &= (series <= rule["lte"]).to_numpy(dtype=bool)
        if "lt" in rule:
            mask &= (series < rule["lt"]).to_numpy(dtype=bool)

        return mask

    def fit(
        self,
        *,
        merged,
        train_origin_days: list[date],
        valid_origin_days: list[date],
        segment_range: tuple[int, int],
        max_train_rows: int | None = None,
    ) -> dict:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency varies by environment
            raise ModuleNotFoundError(
                "lightgbm is required for LGB target-day benchmark runs"
            ) from exc
        self.segment_range = segment_range
        if self.future_dpv_store is not None:
            start_raw = self.validation_cfg.get("train_origin_start")
            start_ts = (
                pd.Timestamp(start_raw).date()
                if start_raw
                else self.future_dpv_store.frame["origin_day"].min().date()
            )
            train_origin_days = [d for d in train_origin_days if d >= start_ts]
            valid_origin_days = [d for d in valid_origin_days if d >= start_ts]
            if not train_origin_days or not valid_origin_days:
                raise FutureDPVAlignmentError(
                    "future DPV enabled but train/valid origin pool empty after "
                    f"train_origin_start={start_ts}"
                )
            audit_path = Path("future_dpv_alignment_audit.json")
            audit_path.write_text(
                json.dumps(self.future_dpv_alignment_audit(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        limited_train_origin_days = self._limit_origin_days(train_origin_days, segment_range, max_train_rows)
        train_frame = self._frame_from_origins(merged, limited_train_origin_days, segment_range)
        valid_frame = self._frame_from_origins(merged, valid_origin_days, segment_range)

        if max_train_rows is not None and len(train_frame) > max_train_rows:
            train_frame = train_frame.sample(n=max_train_rows, random_state=int(self.model_cfg.get("seed", 3407))).sort_index()

        drop_cols = ["target", "timestamp"]
        X_train = train_frame.drop(columns=drop_cols)
        y_train = train_frame["target"]
        X_valid = valid_frame.drop(columns=drop_cols)
        y_valid = valid_frame["target"]
        train_weights = self._sample_weights(train_frame)
        valid_weights = self._sample_weights(valid_frame)

        selected_feature_names = self._selected_feature_names(X_train, y_train)
        X_train = X_train[selected_feature_names]
        X_valid = X_valid[selected_feature_names]

        self.feature_names_ = list(X_train.columns)
        seed_bag_cfg = dict(self.model_cfg.get("seed_bag", {}) or {})
        seed_bag_count = int(seed_bag_cfg.get("count", 1))
        # R5 combined-hypothesis seed: retain the horizon island's seed/ET
        # structure while adding the calendar island's origin-level subbagging.
        # Members k>=1 see deterministic subsets of whole origin-day blocks;
        # member 0 remains the full-data anchor.
        bootstrap_origins = bool(seed_bag_cfg.get("bootstrap_origins", False))
        bootstrap_fraction = min(
            1.0,
            max(0.1, float(seed_bag_cfg.get("bootstrap_fraction", 0.7))),
        )
        base_seed = int(self.model_cfg.get("random_state", 3407))
        self.models_: list = []
        common_params = dict(
            n_estimators=int(self.model_cfg.get("n_estimators", 300)),
            learning_rate=float(self.model_cfg.get("learning_rate", 0.05)),
            num_leaves=int(self.model_cfg.get("num_leaves", 63)),
            max_depth=int(self.model_cfg.get("max_depth", -1)),
            min_child_samples=int(self.model_cfg.get("min_child_samples", 40)),
            subsample=float(self.model_cfg.get("subsample", 0.9)),
            subsample_freq=int(self.model_cfg.get("subsample_freq", 0)),
            colsample_bytree=float(self.model_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(self.model_cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(self.model_cfg.get("reg_lambda", 0.1)),
            n_jobs=int(self.model_cfg.get("n_jobs", 8)),
        )
        diversity_mode = str(seed_bag_cfg.get("diversity", "none")).lower()
        train_origin_ids = (
            X_train["origin_ordinal"].to_numpy(dtype=np.int64)
            if bootstrap_origins and "origin_ordinal" in X_train.columns
            else None
        )
        for k in range(max(1, seed_bag_count)):
            params_k = dict(common_params)
            params_k["random_state"] = base_seed + k * 101
            # IDEA-048: per-seed ensemble decorrelation. Alternate extra_trees
            # (extremely-randomised splits with random thresholds) across the bag so
            # the seed models form genuinely independent tree structures -> the bagged
            # mean achieves a stronger 1/sqrt(n) variance reduction than correlated
            # same-param seeds (which only differ by random_state draws). Crucially
            # this operates PER-SEED before bagging+calibration, so it survives the
            # calibration-dominance filter that neutralised iter-7's point-shift.
            if diversity_mode in ("extra_trees", "extra-trees", "et"):
                params_k["extra_trees"] = bool(k % 2 == 1)
            X_fit, y_fit, w_fit = X_train, y_train, train_weights
            if bootstrap_origins and k > 0 and train_origin_ids is not None:
                rng = np.random.default_rng(base_seed + k * 101)
                unique_origins = np.unique(train_origin_ids)
                n_keep = max(1, int(round(len(unique_origins) * bootstrap_fraction)))
                kept_origins = rng.choice(unique_origins, size=n_keep, replace=False)
                fit_idx = np.flatnonzero(np.isin(train_origin_ids, kept_origins))
                X_fit = X_train.iloc[fit_idx]
                y_fit = y_train.iloc[fit_idx]
                if train_weights is not None:
                    w_fit = np.asarray(train_weights)[fit_idx]
            model_k = lgb.LGBMRegressor(**params_k)
            model_k.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit,
                eval_set=[(X_valid, y_valid)],
                eval_sample_weight=[valid_weights] if valid_weights is not None else None,
                # iter-15 IDEA-040: early-stop on the SCORED metric form (MAPE-style
                # relative error) instead of l1. MAE is dominated by the 60-75GW
                # evening/night peaks; the benchmark scores relative error, where a
                # midday-trough MW costs 1.3-1.5x more. Objective and weights are
                # untouched -- only each member's stopping iteration moves.
                eval_metric=_mape_style_eval,
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.models_.append(model_k)
        self.model = self.models_[0]
        valid_dayplus = valid_frame["dayplus"].to_numpy(dtype=int)
        pred_valid = self._bagged_prediction(X_valid, valid_dayplus)
        if bool(self.calibration_cfg.get("enabled", False)):
            self._fit_calibration(pred_valid, y_valid.to_numpy(), valid_frame)
        valid_acc = 100.0 - np.mean(np.abs((y_valid.to_numpy() - pred_valid) / np.where(np.abs(y_valid.to_numpy()) < 1e-8, 1e-8, y_valid.to_numpy()))) * 100.0
        return {
            "n_train_origin_days": int(len(limited_train_origin_days)),
            "n_train_rows": int(len(train_frame)),
            "n_valid_rows": int(len(valid_frame)),
            "sample_weight_enabled": bool(self.sample_weight_cfg.get("enabled", False)),
            "best_iteration": int(getattr(self.model, "best_iteration_", 0) or 0),
            "valid_acc": float(valid_acc),
            "segment_range": [segment_range[0], segment_range[1]],
        }

    def _calibration_hour_bin(self, hour: int) -> int:
        if hour < 6:
            return 0
        if hour < 10:
            return 1
        if hour < 16:
            return 2
        if hour < 18:
            return 3
        if hour < 22:
            return 4
        return 5

    def _fit_calibration(self, pred_valid: np.ndarray, y_valid: np.ndarray, valid_frame: pd.DataFrame) -> None:
        """Fit per-(hour_bin, dayplus) mean-residual bias correction.

        When `scored_only=true`, calibration is restricted to hour_bins 2 (midday
        10-15) and 4 (night 18-21) — the scored bands. Non-scored hours are left
        untouched, avoiding the iter-7 failure mode where over-correction of
        unscored hours dragged overall accuracy down.
        """
        min_samples = int(self.calibration_cfg.get("min_samples", 50))
        shrink = float(self.calibration_cfg.get("shrinkage", 1.0))
        scored_only = bool(self.calibration_cfg.get("scored_only", True))
        # Calendar island's paired full-positive mechanism: shrink each
        # (hour_bin, dayplus) residual mean toward the same-hour-bin mean across
        # dayplus. pool_lambda=0 preserves the horizon island's legacy behavior.
        pool_lambda = min(
            1.0,
            max(0.0, float(self.calibration_cfg.get("pool_lambda", 0.0))),
        )
        shrink_by_hour_bin_raw = self.calibration_cfg.get("shrinkage_by_hour_bin")
        shrink_by_hour_bin: dict[int, float] = {}
        if shrink_by_hour_bin_raw is not None:
            for k, v in dict(shrink_by_hour_bin_raw).items():
                shrink_by_hour_bin[int(k)] = float(v)
        residuals = y_valid - pred_valid
        hours = valid_frame["hour"].to_numpy(dtype=int)
        dayplus = valid_frame["dayplus"].to_numpy(dtype=int)
        hour_bins = np.array([self._calibration_hour_bin(h) for h in hours], dtype=int)
        self.calibration_bias_ = {}
        active_bins = {2, 4} if scored_only else {0, 1, 2, 3, 4, 5}
        hour_bin_prior: dict[int, float] = {}
        if pool_lambda > 0.0:
            for hb in active_bins:
                hb_mask = hour_bins == hb
                if int(hb_mask.sum()) >= min_samples:
                    hour_bin_prior[int(hb)] = float(np.mean(residuals[hb_mask]))
        for hb in np.unique(hour_bins):
            if int(hb) not in active_bins:
                continue
            bucket_shrink = shrink_by_hour_bin.get(int(hb), shrink)
            prior = hour_bin_prior.get(int(hb))
            for dp in np.unique(dayplus):
                mask = (hour_bins == hb) & (dayplus == dp)
                if mask.sum() < min_samples:
                    continue
                bucket_mean = float(np.mean(residuals[mask]))
                if pool_lambda > 0.0 and prior is not None:
                    bucket_mean = pool_lambda * prior + (1.0 - pool_lambda) * bucket_mean
                self.calibration_bias_[(int(hb), int(dp))] = bucket_mean * bucket_shrink

    def _apply_calibration(self, pred: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        if not self.calibration_bias_:
            return pred
        hours = frame["hour"].to_numpy(dtype=int)
        dayplus = frame["dayplus"].to_numpy(dtype=int)
        hour_bins = np.array([self._calibration_hour_bin(h) for h in hours], dtype=int)
        corrected = pred.astype(np.float64, copy=True)
        for (hb, dp), bias in self.calibration_bias_.items():
            mask = (hour_bins == hb) & (dayplus == dp)
            if mask.any():
                corrected[mask] += bias
        return corrected

    def _bagged_prediction(self, X, dayplus_values: np.ndarray) -> np.ndarray:
        """Average seed-bag predictions with optional per-horizon blend weights.

        Default (no `seed_bag.horizon_blend`): uniform mean across all seed models.
        With `horizon_blend` (a list whose i-th entry is the weight vector for
        dayplus=i+1): per-row weighted average across seed models. Row weights
        are normalized so each row's prediction is a convex combination of seeds.
        """
        models = getattr(self, "models_", []) or [self.model]
        if len(models) == 1:
            return models[0].predict(X)
        preds_per_seed = np.stack([m.predict(X) for m in models], axis=0)  # (n_seeds, n_rows)
        seed_bag_cfg = dict(self.model_cfg.get("seed_bag", {}) or {})
        horizon_blend = seed_bag_cfg.get("horizon_blend")
        if not horizon_blend:
            return np.mean(preds_per_seed, axis=0)
        n_seeds = len(models)
        dp = np.asarray(dayplus_values, dtype=int)
        max_dp = int(dp.max()) if dp.size else 0
        weight_table = np.ones((max_dp + 1, n_seeds), dtype=np.float64)
        for idx, weights in enumerate(horizon_blend, start=1):
            if idx > max_dp:
                break
            w_list = list(weights)[:n_seeds]
            if len(w_list) < n_seeds:
                w_list = w_list + [0.0] * (n_seeds - len(w_list))
            weight_table[idx] = np.asarray(w_list, dtype=np.float64)
        per_row = weight_table[dp]  # (n_rows, n_seeds)
        row_sums = per_row.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        per_row = per_row / row_sums
        return (per_row * preds_per_seed.T).sum(axis=1)

    def predict_origin(
        self,
        *,
        merged,
        origin_day: date,
        segment_range: tuple[int, int] | None = None,
        required_dayplus_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if self.model is None or self.feature_names_ is None:
            raise RuntimeError("LGB model has not been trained")
        segment = segment_range or self.segment_range
        if segment is None:
            raise RuntimeError("segment_range is missing")
        required_range = required_dayplus_range or segment
        frame = self._frame_from_origins(merged, [origin_day], segment, required_range)
        X = frame[self.feature_names_]
        pred_dayplus = frame["dayplus"].to_numpy(dtype=int)
        pred = self._bagged_prediction(X, pred_dayplus)
        if bool(self.calibration_cfg.get("enabled", False)) and self.calibration_bias_:
            pred = self._apply_calibration(pred, frame)
        result = pd.DataFrame(
            {
                "origin_day": origin_day.isoformat(),
                "target_day": pd.to_datetime(frame["timestamp"]).dt.date.astype(str),
                "dayplus": frame["dayplus"].astype(int),
                "timestamp": pd.to_datetime(frame["timestamp"]).astype(str),
                "actual": frame["target"].astype(float),
                "pred": pred.astype(float),
                "point_index": np.arange(len(frame), dtype=int),
            }
        )
        return result

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("LGB model has not been trained")
        payload = {
            "config": self.config,
            "feature_names": self.feature_names_,
            "segment_range": self.segment_range,
            "model": self.model,
            "models": getattr(self, "models_", [self.model]),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
