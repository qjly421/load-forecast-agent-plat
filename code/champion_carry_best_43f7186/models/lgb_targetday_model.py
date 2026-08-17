from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


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
        # iter-12/13 (IDEA-031/032): origin-visible rolling forecast-error
        # feedback, gated to short horizons. Mechanism (measured iter-12):
        # regime bias persists on 1-2 day lags (D1 +0.08) but decorrelates by
        # D3-D5 where weather-forecast error dominates (stale correction adds
        # noise there).
        self.error_feedback_cfg = dict(config.get("error_feedback", {}))
        self.model = None
        self.feature_names_: list[str] | None = None
        self.weather_base_cols = list(self.feature_cfg.get("weather_base_cols", []))
        self.lag_days = list(self.feature_cfg.get("lag_days", [1, 2, 3, 4, 5, 6, 7]))
        self.rolling_windows = list(self.feature_cfg.get("rolling_windows", [96, 672]))
        self.segment_range: tuple[int, int] | None = None
        self.early_stopping_rounds = int(self.model_cfg.get("early_stopping_rounds", 30))
        self.enable_rolling_stats = bool(self.feature_cfg.get("enable_rolling_stats", False))
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
        self.enable_lag_regime_features = bool(
            self.feature_cfg.get("enable_lag_regime_features", False)
        )
        self.enable_temperature_group_features = bool(
            self.feature_cfg.get("enable_temperature_group_features", False)
        )
        self.enable_day_night_regime_features = bool(
            self.feature_cfg.get("enable_day_night_regime_features", False)
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
        # iter-22: HDD/CDD degree-day features. Mechanism: data_analysis
        # showed corr(t2m, load) flips sign between Dec-Mar (-0.53) and
        # Jul-Sep (+0.76); trees spend splits rediscovering thresholds.
        # HDD = max(0, T_ref - daily_mean) activates in winter (heating load);
        # CDD = max(0, daily_mean - T_ref) activates in summer (cooling load).
        # Two monotonic features linearize the non-monotonic relationship.
        self.enable_temperature_fix_daily_hdd_cdd = bool(
            self.feature_cfg.get("enable_temperature_fix_daily_hdd_cdd", False)
        )
        self.enable_temperature_fix_hdd_cdd_week_delta = bool(
            self.feature_cfg.get("enable_temperature_fix_hdd_cdd_week_delta", False)
        )
        # iter-25: 3-day trailing mean HDD/CDD. Mechanism: building thermal
        # mass saturates over sustained heat/cold spells, so HVAC compressor
        # duty cycle on day N depends on HDD/CDD history over days N, N-1, N-2
        # (not just N alone). Complements iter-22 (instant) and iter-24
        # (week-over-week change) by capturing the cumulative dimension.
        self.enable_temperature_fix_hdd_cdd_3d_mean = bool(
            self.feature_cfg.get("enable_temperature_fix_hdd_cdd_3d_mean", False)
        )
        self.temperature_fix_hdd_cdd_ref_c = float(
            self.feature_cfg.get("temperature_fix_hdd_cdd_ref_c", 18.0)
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
        self._temperature_group_index_cache: dict[tuple[str, str], list[int]] = {}
        self._temperature_group_stats_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self._temperature_daily_summary_cache: dict[tuple[str, str, str], dict[str, np.ndarray | float] | None] = {}
        self._temperature_streak_cache: dict[tuple[str, str, str], int | None] = {}
        self.feature_selection_cfg = dict(self.config.get("feature_selection", {}))
        self.sample_weight_cfg = dict(self.config.get("sample_weight", {}))
        self.calibration_cfg = dict(self.config.get("calibration", {}))
        self.calibration_bias_: dict[tuple[int, int], float] = {}
        # IDEA-012: hybrid two-model stack. Trains a single LGB on the ratio
        # target y/gap_load_same_slot ALONGSIDE the existing raw-MW seed-bag.
        # At predict, blend: pred = α*pred_raw + (1-α)*(pred_ratio*gap_load).
        # Mechanism being tested: does the ratio model capture *shape* signal
        # (per-slot relative deviation from gap_load) that the raw model —
        # which learns absolute MW — misses? The ratio model is a single LGB
        # (not a seed-bag) to limit training cost; its sole job is to provide
        # shape info, with the raw model providing level calibration. The
        # blend weight α is set by `hybrid_blend.alpha` (default 0.7).
        self.hybrid_blend_cfg = dict(self.config.get("hybrid_blend", {}) or {})
        self.hybrid_blend_enabled = bool(self.hybrid_blend_cfg.get("enabled", False))
        self.hybrid_blend_alpha = float(self.hybrid_blend_cfg.get("alpha", 0.7))
        # iter-14: alpha_exclude_segment — blend is suppressed (alpha=alpha_in)
        # inside the segment, full default alpha elsewhere. Used to skip blend
        # at scored-night window (h18-21) where iter-11 showed -0.054 medium /
        # -0.96 full regression; raw model is well-calibrated there.
        self.hybrid_blend_alpha_exclude_segment = (
            self.hybrid_blend_cfg.get("alpha_exclude_segment") or None
        )
        self.ratio_model_ = None  # single LGB on y/gap_load_same_slot
        # iter-19: holiday encoding with TYPE split. iter-16 lesson: a single
        # is_holiday feature conflates CNY (-30% deviation) with minor holidays
        # like Dragon Boat (-5%), causing LGB to over-predict minor-holiday
        # drops. Split into is_major_holiday (CNY/National/May Day, -10%+ dev)
        # and is_minor_holiday (Qingming/Dragon Boat/Mid-Autumn, -3% to -8%).
        self.enable_holiday_features_v2 = bool(
            self.feature_cfg.get("enable_holiday_features_v2", False)
        )
        self._holiday_calendar_v2 = self._build_holiday_calendar_v2()
        self.load_reference_mode = str(self.feature_cfg.get("load_reference_mode", "origin"))
        self.control_load_gap_days = int(self.feature_cfg.get("control_load_gap_days", 2))
        self.weather_reference_mode = str(self.feature_cfg.get("weather_reference_mode", "dayplus"))
        self.enable_apparent_temp_hour_interaction = bool(
            self.feature_cfg.get("enable_apparent_temp_hour_interaction", False)
        )
        # iter-20: midday-gated solar radiation interaction. Mechanism:
        # behind-the-meter PV is the dominant midday load driver in Shandong
        # (data_analysis: shortwave corr -0.40 at midday vs temp ~0). iter-18
        # lesson: hour-interaction features are hour-segment-specific (helpful
        # midday, noisy night). Use midday_scored gating (h10-15) so the
        # feature is zero outside midday, eliminating night pollution.
        self.enable_solar_hour_interaction = bool(
            self.feature_cfg.get("enable_solar_hour_interaction", False)
        )
        self._apparent_temp_idx_cache: dict[str, list[int] | None] = {}
        self._solar_idx_cache: dict[str, list[int] | None] = {}

    @staticmethod
    def _build_holiday_calendar_v2() -> dict[str, str]:
        """Returns mapping {YYYY-MM-DD: 'major' | 'minor'} for 2024-2026.

        Major (>=10% load deviation in Shandong training data):
        - CNY (Feb): -22% to -34%
        - National Day (Oct 1-7): -14%
        - May Day (May 1-5): -8% to -14%
        Minor (-3% to -8% deviation):
        - Qingming (Apr 4-6)
        - Dragon Boat (variable, May/Jun)
        - Mid-Autumn (variable, Sep/Oct)
        """
        cal: dict[str, str] = {}
        # CNY (most important)
        for d in range(9, 18):
            cal[f"2024-02-{d:02d}"] = "major"
        for d in range(26, 32):
            cal[f"2025-01-{d:02d}"] = "major"
        for d in range(1, 5):
            cal[f"2025-02-{d:02d}"] = "major"
        for d in range(15, 25):
            cal[f"2026-02-{d:02d}"] = "major"
        # National Day (Oct 1-7, 2025: Oct 1-8 with Mid-Autumn)
        for d in range(1, 8):
            cal[f"2024-10-{d:02d}"] = "major"
            cal[f"2026-10-{d:02d}"] = "major"
        for d in range(1, 9):
            cal[f"2025-10-{d:02d}"] = "major"
        # May Day (May 1-5)
        for d in range(1, 6):
            cal[f"2024-05-{d:02d}"] = "major"
            cal[f"2025-05-{d:02d}"] = "major"
            cal[f"2026-05-{d:02d}"] = "major"
        # Minor: Qingming (Apr 4-6)
        for d in range(4, 7):
            cal[f"2024-04-{d:02d}"] = "minor"
            cal[f"2025-04-{d:02d}"] = "minor"
            cal[f"2026-04-{d:02d}"] = "minor"
        # Minor: Dragon Boat (Jun 10 2024, May 31-Jun 2 2025, Jun 19-21 2026)
        cal["2024-06-10"] = "minor"
        cal["2025-05-31"] = "minor"
        for d in range(1, 3):
            cal[f"2025-06-{d:02d}"] = "minor"
        for d in range(19, 22):
            cal[f"2026-06-{d:02d}"] = "minor"
        # Minor: Mid-Autumn (Sep 15-17 2024, Oct 6 2025, Sep 25-27 2026)
        for d in range(15, 18):
            cal[f"2024-09-{d:02d}"] = "minor"
        for d in range(25, 28):
            cal[f"2026-09-{d:02d}"] = "minor"
        # Note: 2025 Mid-Autumn already covered by National Day extension
        return cal

    def _holiday_features_v2(self, target_day: date) -> dict:
        """Two-binary holiday encoding split by type (major vs minor)."""
        cal = self._holiday_calendar_v2
        target_str = target_day.strftime("%Y-%m-%d")
        htype = cal.get(target_str)
        return {
            "is_major_holiday": 1 if htype == "major" else 0,
            "is_minor_holiday": 1 if htype == "minor" else 0,
        }

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
        if window <= 0:
            return []
        history_days = max(1, int(np.ceil(window / 96)))
        values: list[float] = []
        for offset in range(1, history_days + 1):
            lag_day = origin_day - timedelta(days=offset)
            lag_load = merged.day_load(lag_day)
            if lag_load is None:
                return None
            values.append(float(lag_load[slot_index]))
        max_points = max(1, int(window))
        return values[:max_points]

    @staticmethod
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

    def _solar_indices(self, merged, prefix: str) -> list[int] | None:
        """Indices of shortwave_radiation columns (across all stations) under prefix."""
        if prefix in self._solar_idx_cache:
            return self._solar_idx_cache[prefix]
        cols = merged.prefix_feature_columns(prefix)
        if not cols:
            self._solar_idx_cache[prefix] = None
            return None
        idx = [i for i, c in enumerate(cols) if c.endswith("__shortwave_radiation")]
        self._solar_idx_cache[prefix] = idx if idx else None
        return self._solar_idx_cache[prefix]

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
                if self.enable_temperature_fix_daily_hdd_cdd:
                    t_ref = self.temperature_fix_hdd_cdd_ref_c
                    target_daily_mean = float(target_summary["daily_mean"])
                    gap_daily_mean = float(gap_summary["daily_mean"])
                    target_hdd = max(0.0, t_ref - target_daily_mean)
                    target_cdd = max(0.0, target_daily_mean - t_ref)
                    gap_hdd = max(0.0, t_ref - gap_daily_mean)
                    gap_cdd = max(0.0, gap_daily_mean - t_ref)
                    stats[f"{prefix_name}_target_daily_hdd"] = target_hdd
                    stats[f"{prefix_name}_target_daily_cdd"] = target_cdd
                    stats[f"{prefix_name}_gap_daily_hdd"] = gap_hdd
                    stats[f"{prefix_name}_gap_daily_cdd"] = gap_cdd
                    stats[f"{prefix_name}_target_gap_hdd_delta"] = target_hdd - gap_hdd
                    stats[f"{prefix_name}_target_gap_cdd_delta"] = target_cdd - gap_cdd
                    if self.enable_temperature_fix_hdd_cdd_week_delta:
                        # iter-24: 7-day HDD/CDD delta (cold-snap / heat-wave signal).
                        # Mechanism: HVAC responds to *changes* in heating/cooling
                        # demand, not absolute levels. A target day with HDD=8 after
                        # a lag7 day with HDD=2 represents a cold-snap (ramp-up);
                        # the inverse represents a warm spell (ramp-down). Trees
                        # splitting on this delta capture the dynamic regime
                        # direction directly. Uses existing temp lookback infra
                        # (no load lookback, no train-set shrinkage).
                        lag7_day = origin_day - timedelta(days=7)
                        lag7_summary = self._temperature_daily_summary(
                            merged=merged,
                            target_day=lag7_day,
                            prefix="D_0__",
                            base_col=base_col,
                        )
                        if lag7_summary is not None:
                            lag7_daily_mean = float(lag7_summary["daily_mean"])
                            lag7_hdd = max(0.0, t_ref - lag7_daily_mean)
                            lag7_cdd = max(0.0, lag7_daily_mean - t_ref)
                            stats[f"{prefix_name}_target_lag7_hdd_delta"] = target_hdd - lag7_hdd
                            stats[f"{prefix_name}_target_lag7_cdd_delta"] = target_cdd - lag7_cdd
                            stats[f"{prefix_name}_gap_lag7_hdd_delta"] = gap_hdd - lag7_hdd
                            stats[f"{prefix_name}_gap_lag7_cdd_delta"] = gap_cdd - lag7_cdd
                    if self.enable_temperature_fix_hdd_cdd_3d_mean:
                        # iter-25: 3-day trailing mean HDD/CDD ending at
                        # target_day (and at origin_day for the gap analog).
                        # Mechanism: thermal mass of buildings means HVAC
                        # duty cycle on day N reflects days N, N-1, N-2.
                        # Uses the existing temperature lookback (already
                        # 14 days) — no train-set shrinkage.
                        target_window = [target_day]
                        gap_window = [origin_day]
                        for back in (1, 2):
                            target_window.append(target_day - timedelta(days=back))
                            gap_window.append(origin_day - timedelta(days=back))
                        target_hdds, target_cdds = [], []
                        gap_hdds, gap_cdds = [], []
                        ok = True
                        for d, gd in zip(target_window, gap_window):
                            ts = self._temperature_daily_summary(
                                merged=merged, target_day=d, prefix=target_prefix, base_col=base_col,
                            )
                            gs = self._temperature_daily_summary(
                                merged=merged, target_day=gd, prefix="D_0__", base_col=base_col,
                            )
                            if ts is None or gs is None:
                                ok = False
                                break
                            tm = float(ts["daily_mean"])
                            gm = float(gs["daily_mean"])
                            target_hdds.append(max(0.0, t_ref - tm))
                            target_cdds.append(max(0.0, tm - t_ref))
                            gap_hdds.append(max(0.0, t_ref - gm))
                            gap_cdds.append(max(0.0, gm - t_ref))
                        if ok:
                            stats[f"{prefix_name}_target_hdd_3d_mean"] = sum(target_hdds) / 3.0
                            stats[f"{prefix_name}_target_cdd_3d_mean"] = sum(target_cdds) / 3.0
                            stats[f"{prefix_name}_gap_hdd_3d_mean"] = sum(gap_hdds) / 3.0
                            stats[f"{prefix_name}_gap_cdd_3d_mean"] = sum(gap_cdds) / 3.0

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
                feature_row[f"roll_min_{int(window)}"] = self._safe_min(roll_values)
                feature_row[f"roll_max_{int(window)}"] = self._safe_max(roll_values)
                feature_row[f"gap_vs_roll_mean_{int(window)}"] = float(
                    feature_row["gap_load_same_slot"] - feature_row[f"roll_mean_{int(window)}"]
                )

        if self.enable_dayplus_calendar_interactions:
            feature_row["dayplus_x_month"] = int(dayplus) * int(feature_row["month"])
            feature_row["dayplus_x_dow"] = int(dayplus) * int(feature_row["dow"])
            feature_row["dayplus_x_is_weekend"] = int(dayplus) * int(feature_row["is_weekend"])
            feature_row["dayplus_x_hour"] = int(dayplus) * int(feature_row["hour"])

        if self.enable_holiday_features_v2:
            hf = self._holiday_features_v2(target_day)
            feature_row["is_major_holiday"] = hf["is_major_holiday"]
            feature_row["is_minor_holiday"] = hf["is_minor_holiday"]

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

        if self.enable_solar_hour_interaction:
            target_prefix_sol = f"D_{dayplus}__" if self.weather_reference_mode == "dayplus" else "D_1__"
            solar_idx = self._solar_indices(merged, target_prefix_sol)
            if solar_idx:
                solar_slot = float(np.mean([target_weather[slot_index][i] for i in solar_idx]))
                is_midday_scored = 1 if 10 <= hour_val < 15 else 0
                # iter-20: only emit the midday-gated form (per iter-18 lesson:
                # hour_sin/cos versions pollute night). PV signal is meaningful
                # only during daytime; gating to scored-midday window eliminates
                # night variance injection.
                feature_row["shortwave_target_x_midday_scored"] = solar_slot * is_midday_scored

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
        return feature_row

    def _frame_from_origins(
        self,
        merged,
        origin_days: list[date],
        segment_range: tuple[int, int],
        required_dayplus_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        required_range = required_dayplus_range or segment_range
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
        base_seed = int(self.model_cfg.get("random_state", 3407))
        self.models_: list = []
        common_params = dict(
            n_estimators=int(self.model_cfg.get("n_estimators", 300)),
            learning_rate=float(self.model_cfg.get("learning_rate", 0.05)),
            num_leaves=int(self.model_cfg.get("num_leaves", 63)),
            max_depth=int(self.model_cfg.get("max_depth", -1)),
            min_child_samples=int(self.model_cfg.get("min_child_samples", 40)),
            subsample=float(self.model_cfg.get("subsample", 0.9)),
            colsample_bytree=float(self.model_cfg.get("colsample_bytree", 0.8)),
            reg_alpha=float(self.model_cfg.get("reg_alpha", 0.0)),
            reg_lambda=float(self.model_cfg.get("reg_lambda", 0.1)),
            n_jobs=int(self.model_cfg.get("n_jobs", 8)),
        )
        # iter-75 IDEA-091: path_smoothing for leaf-value regularization.
        # Default 0 (off); 0.1 = gentle smoothing of leaf values toward
        # ancestor node means. Complementary to early_stopping + monotone.
        path_smoothing = float(self.model_cfg.get("path_smoothing", 0.0))
        if path_smoothing > 0.0:
            common_params["path_smoothing"] = path_smoothing
        # iter-79 IDEA-095: min_gain_to_split filters noise splits. At saturation
        # with 300 boosted trees, late-round splits often gain <0.001 MSE — pure
        # noise overfit. This threshold prevents them without affecting useful
        # splits (which gain >>0.001). Unlike iter-78 ffb (which randomly removed
        # useful feature combinations), this targets only the noise regime.
        min_gain = float(self.model_cfg.get("min_gain_to_split", 0.0))
        if min_gain > 0.0:
            common_params["min_gain_to_split"] = min_gain
        # iter-72 IDEA-083: monotone constraints on HDD/CDD level features.
        # Mechanism: HDD (heating degree-days) captures winter heating load
        # (monotonically increasing in load); CDD (cooling degree-days) captures
        # summer AC load (also monotonically increasing). Encoding this physical
        # prior as a hard constraint frees leaf budget for non-monotonic
        # interactions at saturation. Apply only to LEVEL features (daily_hdd,
        # daily_cdd, hdd_3d_mean, cdd_3d_mean) — NOT delta features (ambiguous
        # direction depending on baseline).
        monotone_targets = str(self.model_cfg.get("monotone_constraints_targets", "")).strip()
        if monotone_targets:
            target_suffixes = [s.strip() for s in monotone_targets.split(",") if s.strip()]
            constraint_vec = []
            for fname in self.feature_names_:
                # only match LEVEL features (hdd/cdd as word-boundary suffix)
                matched = any(fname.endswith(s) for s in target_suffixes)
                constraint_vec.append(1 if matched else 0)
            n_constrained = sum(1 for c in constraint_vec if c != 0)
            if n_constrained > 0:
                common_params["monotone_constraints"] = constraint_vec
        for k in range(max(1, seed_bag_count)):
            params_k = dict(common_params)
            params_k["random_state"] = base_seed + k * 101
            model_k = lgb.LGBMRegressor(**params_k)
            model_k.fit(
                X_train,
                y_train,
                sample_weight=train_weights,
                eval_set=[(X_valid, y_valid)],
                eval_sample_weight=[valid_weights] if valid_weights is not None else None,
                eval_metric="l1",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.models_.append(model_k)
        self.model = self.models_[0]
        # IDEA-012: train a parallel ratio model (single LGB, no seed-bag) on
        # y/gap_load_same_slot. Re-uses the same X_train. The ratio model
        # captures shape signal; raw model captures level. Blend at predict.
        if self.hybrid_blend_enabled and "gap_load_same_slot" in train_frame.columns:
            ratio_denom_train = train_frame["gap_load_same_slot"].to_numpy(dtype=np.float64)
            ratio_denom_valid = valid_frame["gap_load_same_slot"].to_numpy(dtype=np.float64)
            eps_ratio = max(1.0, float(self.hybrid_blend_cfg.get("eps", 1.0)))
            ratio_denom_train = np.maximum(ratio_denom_train, eps_ratio)
            ratio_denom_valid = np.maximum(ratio_denom_valid, eps_ratio)
            y_ratio_train = y_train.to_numpy(dtype=np.float64) / ratio_denom_train
            y_ratio_valid = y_valid.to_numpy(dtype=np.float64) / ratio_denom_valid
            ratio_params = dict(common_params)
            # iter-72: monotone_constraints incompatible with quantile objective (ratio model).
            # Strip from ratio_params if present (only seed-bag models use it).
            ratio_params.pop("monotone_constraints", None)
            ratio_params["random_state"] = base_seed + 991
            # iter-48: ratio model uses quantile=0.5 (median) objective for
            # robustness to outlier days (CNY -30%, National Day -14%) that
            # skew MSE-learned shape. Median preserves the per-slot pattern.
            ratio_objective = str(self.hybrid_blend_cfg.get("ratio_objective", "regression"))
            if ratio_objective == "quantile":
                ratio_params["objective"] = "quantile"
                ratio_params["alpha"] = float(self.hybrid_blend_cfg.get("ratio_quantile_alpha", 0.5))
            self.ratio_model_ = lgb.LGBMRegressor(**ratio_params)
            self.ratio_model_.fit(
                X_train,
                y_ratio_train,
                sample_weight=train_weights,
                eval_set=[(X_valid, y_ratio_valid)],
                eval_sample_weight=[valid_weights] if valid_weights is not None else None,
                eval_metric="l1",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
        valid_dayplus = valid_frame["dayplus"].to_numpy(dtype=int)
        pred_valid = self._bagged_prediction(X_valid, valid_dayplus)
        # IDEA-012: blend raw model with ratio model on validation set BEFORE
        # fitting calibration (so calibration captures blend's residual bias).
        if self.hybrid_blend_enabled and self.ratio_model_ is not None and "gap_load_same_slot" in valid_frame.columns:
            ratio_pred_valid = self.ratio_model_.predict(X_valid)
            gap_load_valid = valid_frame["gap_load_same_slot"].to_numpy(dtype=np.float64)
            ratio_inverse_valid = ratio_pred_valid * np.maximum(gap_load_valid, 1.0)
            alpha_v = self._hybrid_blend_alpha_per_row(valid_frame)
            pred_valid = alpha_v * pred_valid + (1.0 - alpha_v) * ratio_inverse_valid
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
        if hour < 20:
            return 4
        if hour < 22:
            return 6
        return 5

    def _fit_calibration(self, pred_valid: np.ndarray, y_valid: np.ndarray, valid_frame: pd.DataFrame) -> None:
        """Fit per-(hour_bin, dayplus) mean-residual bias correction.

        When `scored_only=true`, calibration is restricted to hour_bins 2 (midday
        10-15) and 4 (night 18-21) — the scored bands. Non-scored hours are left
        untouched, avoiding the iter-7 failure mode where over-correction of
        unscored hours dragged overall accuracy down.

        IDEA-001 (iter-1): optional partial pooling across dayplus. When
        `calibration.pool_lambda` is set (a {dayplus: λ} map), each cell-mean bias
        is blended toward the hour-bin pooled mean:
            bias = λ_dp * cell_mean + (1 - λ_dp) * mean_hourbin
        BEFORE the per-hour-bin shrinkage is applied. λ shrinks more at longer
        horizons → empirical-Bayes variance reduction for the noisier long-horizon
        cells (fewer samples, larger weather-forecast-error residuals) by borrowing
        strength from the pooled hour-bin estimate. λ=1.0 (or unset) = no pooling.
        """
        min_samples = int(self.calibration_cfg.get("min_samples", 50))
        shrink = float(self.calibration_cfg.get("shrinkage", 1.0))
        scored_only = bool(self.calibration_cfg.get("scored_only", True))
        shrink_by_hour_bin_raw = self.calibration_cfg.get("shrinkage_by_hour_bin")
        shrink_by_hour_bin: dict[int, float] = {}
        if shrink_by_hour_bin_raw is not None:
            for k, v in dict(shrink_by_hour_bin_raw).items():
                shrink_by_hour_bin[int(k)] = float(v)
        pool_lambda_raw = self.calibration_cfg.get("pool_lambda")
        pool_lambda: dict[int, float] = {}
        if pool_lambda_raw is not None:
            for k, v in dict(pool_lambda_raw).items():
                pool_lambda[int(k)] = float(v)
        residuals = y_valid - pred_valid
        hours = valid_frame["hour"].to_numpy(dtype=int)
        dayplus = valid_frame["dayplus"].to_numpy(dtype=int)
        hour_bins = np.array([self._calibration_hour_bin(h) for h in hours], dtype=int)
        self.calibration_bias_ = {}
        active_bins = {2, 4, 6} if scored_only else {0, 1, 2, 3, 4, 5, 6}
        for hb in np.unique(hour_bins):
            if int(hb) not in active_bins:
                continue
            bucket_shrink = shrink_by_hour_bin.get(int(hb), shrink)
            # IDEA-001: hour-bin pooled mean = empirical-Bayes anchor across dayplus
            if pool_lambda:
                hb_mask = hour_bins == hb
                mean_hourbin = float(np.mean(residuals[hb_mask])) if hb_mask.any() else 0.0
            for dp in np.unique(dayplus):
                mask = (hour_bins == hb) & (dayplus == dp)
                if mask.sum() < min_samples:
                    continue
                cell_mean = float(np.mean(residuals[mask]))
                if pool_lambda:
                    lam = pool_lambda.get(int(dp), 1.0)
                    bias_value = lam * cell_mean + (1.0 - lam) * mean_hourbin
                else:
                    bias_value = cell_mean
                self.calibration_bias_[(int(hb), int(dp))] = bias_value * bucket_shrink

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

    def _hybrid_blend_alpha_per_row(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-row blend alpha. Modes (apply in order, last wins):
        - `alpha_by_hour_bin`: alpha override per calibration hour_bin
          (e.g. {2: 0.75} — tilt toward raw at midday where median ratio
          under-predicts the PV dip).
        - `alpha_by_hour_bin_by_dayplus`: nested alpha override per
          (calibration hour_bin, dayplus). Highest priority; e.g.
          {2: {1: 0.7, 2: 0.7, 3: 0.75, 4: 0.8, 5: 0.8}} lets short
          horizons use more median ratio (D1/D2) while long horizons
          tilt toward raw (D4/D5). Wins over `alpha_by_hour_bin`.
        - `alpha_exclude_segment`: alpha = exclude_alpha inside segment,
          default alpha elsewhere. Used to skip blend at scored-night window.
        - default: uniform alpha across all rows.
        """
        n = len(frame)
        out = np.full(n, self.hybrid_blend_alpha, dtype=np.float64)
        hours = frame["hour"].to_numpy(dtype=int)
        hour_bins = np.array([self._calibration_hour_bin(int(h)) for h in hours], dtype=int)
        alpha_by_hb = self.hybrid_blend_cfg.get("alpha_by_hour_bin") or None
        if alpha_by_hb:
            for hb_str, alpha_val in alpha_by_hb.items():
                hb = int(hb_str)
                a = float(alpha_val)
                out[hour_bins == hb] = a
        alpha_by_hb_dp = self.hybrid_blend_cfg.get("alpha_by_hour_bin_by_dayplus") or None
        if alpha_by_hb_dp:
            dayplus = frame["dayplus"].to_numpy(dtype=int)
            for hb_str, dp_dict in alpha_by_hb_dp.items():
                hb = int(hb_str)
                for dp_str, alpha_val in dp_dict.items():
                    dp = int(dp_str)
                    a = float(alpha_val)
                    mask = (hour_bins == hb) & (dayplus == dp)
                    out[mask] = a
        if self.hybrid_blend_alpha_exclude_segment is not None:
            seg = self.hybrid_blend_alpha_exclude_segment
            start_h = int(seg.get("start_hour", 0))
            end_h = int(seg.get("end_hour", 24))
            alpha_in = float(seg.get("alpha", 1.0))
            mask = (hours >= start_h) & (hours < end_h)
            out[mask] = alpha_in
        return out

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
        # iter-8 IDEA-029: post-hoc seed-bag variance reduction via trimmed
        # mean. The seed models are exchangeable (identical config, differ only
        # by random_state), so the single prediction farthest from the active-
        # seed median is statistical noise -> dropping it is an unbiased robust
        # mean (RF/ensemble variance reduction, zero training-side capacity —
        # the only lever class that ever helped, per iters 1-7). Gated on
        # `seed_bag.agg: "trim1"` and applied ONLY to rows with >=
        # `trim_min_active` active seeds (default 5 => D3-D5 only), preserving
        # the champion-tuned D1/D2 mean-of-3 (more reduction there "over-dilutes"
        # per the horizon_blend comment).
        agg = str(seed_bag_cfg.get("agg", "mean"))
        if agg == "trim1":
            trim_min_active = int(seed_bag_cfg.get("trim_min_active", 5))
            active = per_row > 0.0  # (n_rows, n_seeds)
            n_active = active.sum(axis=1)
            eligible = n_active >= trim_min_active
            if eligible.any():
                preds_T = preds_per_seed.T  # (n_rows, n_seeds)
                masked = np.where(active, preds_T, np.nan)
                with np.errstate(invalid="ignore"):
                    row_med = np.nanmedian(masked, axis=1)  # (n_rows,)
                dist = np.abs(preds_T - row_med[:, None])
                # inactive seeds can never be the trimmed one
                dist = np.where(active, dist, -1.0)
                chosen = np.argmax(dist, axis=1)  # (n_rows,)
                rows = np.where(eligible)[0]
                cols = chosen[eligible]
                per_row[rows, cols] = 0.0
        row_sums = per_row.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        per_row = per_row / row_sums
        return (per_row * preds_per_seed.T).sum(axis=1)

    def _apply_error_feedback(
        self, merged, origin_day: date, segment: tuple[int, int], required_range: tuple[int, int], pred: np.ndarray, frame: pd.DataFrame
    ) -> np.ndarray:
        """Subtract lambda * recent mean realized residual per calibration hour bin.

        Residuals come from forecasts issued at origins origin_day-1..-n_origins
        for target days that are already realized (target_day <= last complete
        load day, never beyond origin_day itself). Strictly origin-visible.
        Applied only to rows with dayplus <= max_dayplus (default 2).
        """
        lam = float(self.error_feedback_cfg.get("lambda", 0.3))
        # IDEA-062/063: per-horizon and per-(band,D1) correction strength.
        # Residual freshness relative to the TARGET day decays with horizon
        # (D1's freshest realized residual is 1 day stale, D2's ~2), so the
        # admissible lambda is higher where the estimate is fresher; and the
        # evening/night bands at D1 specifically tolerate more correction
        # (iter-33/34: D1-night +0.030 at bin lambda 0.4) because their base
        # load bias is the most persistent. Falls back to flat lambda.
        lam_by_dp_raw = self.error_feedback_cfg.get("lambda_by_dayplus") or {}
        lam_by_dp = {int(k): float(v) for k, v in lam_by_dp_raw.items()}
        lam_by_bin_d1_raw = self.error_feedback_cfg.get("lambda_by_bin_d1") or {}
        lam_by_bin_d1 = {int(k): float(v) for k, v in lam_by_bin_d1_raw.items()}
        n_back = int(self.error_feedback_cfg.get("n_origins", 5))
        max_dp = int(self.error_feedback_cfg.get("max_dayplus", 2))
        last_realized = merged.last_complete_load_day()
        if lam <= 0.0 or n_back <= 0 or max_dp < 1 or last_realized is None:
            return pred
        last_realized = min(last_realized, origin_day)
        fb_origins = [origin_day - timedelta(days=b) for b in range(1, n_back + 1)]
        fb_frame = self._frame_from_origins(merged, fb_origins, segment, required_range)
        if fb_frame is None or getattr(fb_frame, "empty", True):
            return pred
        ts_dates = pd.to_datetime(fb_frame["timestamp"]).dt.date
        fb_frame = fb_frame[ts_dates.apply(lambda d: d <= last_realized)]
        if fb_frame.empty:
            return pred
        X_fb = fb_frame[self.feature_names_]
        dp_fb = fb_frame["dayplus"].to_numpy(dtype=int)
        p_fb = self._bagged_prediction(X_fb, dp_fb)
        if self.hybrid_blend_enabled and self.ratio_model_ is not None and "gap_load_same_slot" in fb_frame.columns:
            ratio_pred_fb = self.ratio_model_.predict(X_fb)
            gap_fb = fb_frame["gap_load_same_slot"].to_numpy(dtype=np.float64)
            alpha_fb = self._hybrid_blend_alpha_per_row(fb_frame)
            p_fb = alpha_fb * p_fb + (1.0 - alpha_fb) * ratio_pred_fb * np.maximum(gap_fb, 1.0)
        if bool(self.calibration_cfg.get("enabled", False)) and self.calibration_bias_:
            p_fb = self._apply_calibration(p_fb, fb_frame)
        y_fb = fb_frame["target"].to_numpy(dtype=np.float64)
        resid_fb = p_fb - y_fb
        bins_fb = np.array(
            [self._calibration_hour_bin(int(h)) for h in fb_frame["hour"].to_numpy(dtype=int)], dtype=int
        )
        bins_cur = np.array(
            [self._calibration_hour_bin(int(h)) for h in frame["hour"].to_numpy(dtype=int)], dtype=int
        )
        adjusted = pred.astype(np.float64, copy=True)
        dp_cur = frame["dayplus"].to_numpy(dtype=int)
        gated = dp_cur <= max_dp
        # IDEA-033: recency-weight the residuals — regime bias decays with lag
        # (iter-12 mechanism), so yesterday's realized error should count more
        # than one from 5 days back. decay=1.0 reduces to the flat mean.
        decay = float(self.error_feedback_cfg.get("recency_decay", 1.0))
        if decay <= 0.0 or decay >= 1.0:
            w_fb = np.ones(len(resid_fb), dtype=np.float64)
        else:
            fb_dates = pd.to_datetime(fb_frame["timestamp"]).dt.date
            lags = np.array([(origin_day - d).days for d in fb_dates], dtype=float)
            w_fb = np.power(decay, np.maximum(lags, 0.0))
        for hb in np.unique(bins_cur[gated]):
            m = bins_fb == hb
            if m.any():
                sel = gated & (bins_cur == hb)
                w = w_fb[m]
                wsum = float(w.sum())
                if wsum > 0.0:
                    corr = float(np.dot(w, resid_fb[m]) / wsum)
                else:
                    corr = float(np.mean(resid_fb[m]))
                for dp in np.unique(dp_cur[sel]):
                    lam_dp = float(lam_by_dp.get(int(dp), lam))
                    if int(dp) == 1 and int(hb) in lam_by_bin_d1:
                        lam_dp = float(lam_by_bin_d1[int(hb)])
                    adjusted[sel & (dp_cur == dp)] -= lam_dp * corr
        return adjusted

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
        # IDEA-012: blend with ratio model's inverse-transformed prediction.
        # alpha=1.0 reduces to baseline (raw only); alpha=0.0 would be pure
        # ratio (IDEA-001, known bad). Default alpha=0.7 trusts the raw model
        # mostly but allows ratio model's shape signal to contribute.
        if self.hybrid_blend_enabled and self.ratio_model_ is not None and "gap_load_same_slot" in frame.columns:
            ratio_pred = self.ratio_model_.predict(X)
            gap_load_predict = frame["gap_load_same_slot"].to_numpy(dtype=np.float64)
            ratio_inverse = ratio_pred * np.maximum(gap_load_predict, 1.0)
            alpha_p = self._hybrid_blend_alpha_per_row(frame)
            pred = alpha_p * pred + (1.0 - alpha_p) * ratio_inverse
        if bool(self.calibration_cfg.get("enabled", False)) and self.calibration_bias_:
            pred = self._apply_calibration(pred, frame)
        if bool(self.error_feedback_cfg.get("enabled", False)):
            pred = self._apply_error_feedback(merged, origin_day, segment, required_range, pred, frame)
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
