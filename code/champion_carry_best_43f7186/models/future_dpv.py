from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STRICT_KEY_COLUMNS = ("origin_day", "target_day", "dayplus", "timestamp")


class FutureDPVAlignmentError(ValueError):
    """Raised when a future-DPV feature cannot be aligned without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_records(frame: pd.DataFrame, columns: Iterable[str], limit: int = 5) -> list[dict]:
    return frame.loc[:, list(columns)].head(limit).astype(str).to_dict(orient="records")


class FutureDPVFeatureStore:
    """Strict, horizon-aware reader for future distributed-PV predictions.

    The only accepted join contract is
    ``(origin_day, target_day, dayplus, timestamp)``.  Hourly inputs are allowed
    only when every horizon group contains the complete 00:00..23:00 grid; each
    hourly prediction is then repeated into the four corresponding 15-minute
    slots.  Missing rows, duplicate rows, inconsistent horizons, and partial
    days are fatal rather than silently imputed.
    """

    def __init__(
        self,
        source_path: str | Path,
        *,
        value_col: str = "future_dpv_mw",
        hourly_expansion: str = "repeat_4",
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.value_col = str(value_col)
        self.hourly_expansion = str(hourly_expansion)
        self._coverage_checks: list[dict] = []

        if not self.source_path.is_file():
            raise FileNotFoundError(f"future DPV parquet does not exist: {self.source_path}")
        if self.source_path.suffix.lower() not in {".parquet", ".pq"}:
            raise FutureDPVAlignmentError(
                f"future DPV input must be parquet (.parquet or .pq): {self.source_path}"
            )
        if self.hourly_expansion != "repeat_4":
            raise FutureDPVAlignmentError(
                "unsupported future DPV hourly_expansion="
                f"{self.hourly_expansion!r}; only 'repeat_4' is allowed"
            )

        raw = pd.read_parquet(self.source_path)
        self.frame, self._source_audit = self._normalize_frame(raw)
        self._series = self.frame.set_index(list(STRICT_KEY_COLUMNS))[self.value_col]
        if not self._series.index.is_unique:
            raise FutureDPVAlignmentError(
                "future DPV strict key is not unique after normalization; "
                f"key={STRICT_KEY_COLUMNS}"
            )

    @staticmethod
    def _parse_day_column(frame: pd.DataFrame, column: str) -> pd.Series:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed.isna().any():
            bad = frame.loc[parsed.isna(), [column]]
            raise FutureDPVAlignmentError(
                f"future DPV {column} contains unparseable values: "
                f"{_sample_records(bad, [column])}"
            )
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise FutureDPVAlignmentError(
                f"future DPV {column} must be timezone-naive; normalize upstream explicitly"
            )
        non_midnight = parsed != parsed.dt.normalize()
        if non_midnight.any():
            bad = frame.loc[non_midnight, [column]]
            raise FutureDPVAlignmentError(
                f"future DPV {column} must contain day values at midnight: "
                f"{_sample_records(bad, [column])}"
            )
        return parsed.dt.normalize()

    @staticmethod
    def _parse_timestamp_column(frame: pd.DataFrame) -> pd.Series:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
        if parsed.isna().any():
            bad = frame.loc[parsed.isna(), ["timestamp"]]
            raise FutureDPVAlignmentError(
                "future DPV timestamp contains unparseable values: "
                f"{_sample_records(bad, ['timestamp'])}"
            )
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise FutureDPVAlignmentError(
                "future DPV timestamp must be timezone-naive; normalize upstream explicitly"
            )
        return parsed

    def _normalize_frame(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        required = [*STRICT_KEY_COLUMNS, self.value_col]
        missing_columns = [column for column in required if column not in raw.columns]
        if missing_columns:
            raise FutureDPVAlignmentError(
                f"future DPV parquet is missing required columns {missing_columns}; "
                f"required={required}"
            )
        if raw.empty:
            raise FutureDPVAlignmentError("future DPV parquet is empty")

        frame = raw.loc[:, required].copy()
        frame["origin_day"] = self._parse_day_column(frame, "origin_day")
        frame["target_day"] = self._parse_day_column(frame, "target_day")
        frame["timestamp"] = self._parse_timestamp_column(frame)

        dayplus_numeric = pd.to_numeric(frame["dayplus"], errors="coerce")
        integral_dayplus = dayplus_numeric.notna() & np.isclose(
            dayplus_numeric.to_numpy(dtype=float),
            np.round(dayplus_numeric.to_numpy(dtype=float)),
        )
        if not integral_dayplus.all():
            bad = frame.loc[~integral_dayplus, ["dayplus"]]
            raise FutureDPVAlignmentError(
                "future DPV dayplus must contain finite integers: "
                f"{_sample_records(bad, ['dayplus'])}"
            )
        frame["dayplus"] = dayplus_numeric.astype(int)
        if (frame["dayplus"] <= 0).any():
            bad = frame.loc[frame["dayplus"] <= 0, list(STRICT_KEY_COLUMNS)]
            raise FutureDPVAlignmentError(
                "future DPV dayplus must be positive: "
                f"{_sample_records(bad, STRICT_KEY_COLUMNS)}"
            )

        expected_dayplus = (frame["target_day"] - frame["origin_day"]).dt.days
        wrong_horizon = expected_dayplus != frame["dayplus"]
        if wrong_horizon.any():
            bad = frame.loc[wrong_horizon, list(STRICT_KEY_COLUMNS)].copy()
            bad["expected_dayplus"] = expected_dayplus.loc[wrong_horizon]
            raise FutureDPVAlignmentError(
                "future DPV horizon mismatch: target_day-origin_day must equal dayplus; "
                f"examples={_sample_records(bad, [*STRICT_KEY_COLUMNS, 'expected_dayplus'])}"
            )

        wrong_target_day = frame["timestamp"].dt.normalize() != frame["target_day"]
        if wrong_target_day.any():
            bad = frame.loc[wrong_target_day, list(STRICT_KEY_COLUMNS)]
            raise FutureDPVAlignmentError(
                "future DPV timestamp falls outside target_day: "
                f"examples={_sample_records(bad, STRICT_KEY_COLUMNS)}"
            )

        values = pd.to_numeric(frame[self.value_col], errors="coerce")
        finite_values = values.notna() & np.isfinite(values.to_numpy(dtype=float))
        if not finite_values.all():
            bad = frame.loc[~finite_values, [*STRICT_KEY_COLUMNS, self.value_col]]
            raise FutureDPVAlignmentError(
                f"future DPV {self.value_col} must be finite numeric values: "
                f"{_sample_records(bad, [*STRICT_KEY_COLUMNS, self.value_col])}"
            )
        frame[self.value_col] = values.astype(float)

        duplicate_mask = frame.duplicated(subset=list(STRICT_KEY_COLUMNS), keep=False)
        if duplicate_mask.any():
            bad = frame.loc[duplicate_mask, list(STRICT_KEY_COLUMNS)]
            raise FutureDPVAlignmentError(
                "future DPV contains duplicate strict keys before cadence expansion: "
                f"examples={_sample_records(bad, STRICT_KEY_COLUMNS)}"
            )

        normalized_groups: list[pd.DataFrame] = []
        hourly_groups = 0
        quarter_hour_groups = 0
        group_columns = ["origin_day", "target_day", "dayplus"]
        for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            target_day = pd.Timestamp(group_key[1]).normalize()
            expected_hourly = pd.date_range(target_day, periods=24, freq="1h")
            expected_quarter_hour = pd.date_range(target_day, periods=96, freq="15min")
            observed = pd.DatetimeIndex(group["timestamp"])

            if len(group) == 24 and observed.equals(expected_hourly):
                hourly_groups += 1
                repeated = group.loc[group.index.repeat(4)].reset_index(drop=True)
                offsets = np.tile(pd.to_timedelta([0, 15, 30, 45], unit="min"), len(group))
                repeated["timestamp"] = repeated["timestamp"] + offsets
                normalized_groups.append(repeated)
            elif len(group) == 96 and observed.equals(expected_quarter_hour):
                quarter_hour_groups += 1
                normalized_groups.append(group)
            else:
                missing_15m = expected_quarter_hour.difference(observed)
                extra = observed.difference(expected_quarter_hour)
                raise FutureDPVAlignmentError(
                    "future DPV group must be a complete 24-hour or 96-quarter-hour day; "
                    f"group=(origin_day={pd.Timestamp(group_key[0]).date()}, "
                    f"target_day={target_day.date()}, dayplus={int(group_key[2])}), "
                    f"rows={len(group)}, first_missing_15m={list(missing_15m[:4].astype(str))}, "
                    f"first_extra={list(extra[:4].astype(str))}"
                )

        normalized = pd.concat(normalized_groups, ignore_index=True)
        normalized = normalized.sort_values(list(STRICT_KEY_COLUMNS)).reset_index(drop=True)
        duplicate_mask = normalized.duplicated(subset=list(STRICT_KEY_COLUMNS), keep=False)
        if duplicate_mask.any():
            bad = normalized.loc[duplicate_mask, list(STRICT_KEY_COLUMNS)]
            raise FutureDPVAlignmentError(
                "future DPV contains duplicate strict keys after cadence expansion: "
                f"examples={_sample_records(bad, STRICT_KEY_COLUMNS)}"
            )

        audit = {
            "source_path": str(self.source_path),
            "source_sha256": _sha256_file(self.source_path),
            "strict_key_columns": list(STRICT_KEY_COLUMNS),
            "value_column": self.value_col,
            "hourly_expansion": self.hourly_expansion,
            "source_rows": int(len(raw)),
            "normalized_15m_rows": int(len(normalized)),
            "horizon_group_count": int(hourly_groups + quarter_hour_groups),
            "hourly_groups_expanded": int(hourly_groups),
            "quarter_hour_groups": int(quarter_hour_groups),
            "origin_day_start": str(normalized["origin_day"].min().date()),
            "origin_day_end": str(normalized["origin_day"].max().date()),
            "target_day_start": str(normalized["target_day"].min().date()),
            "target_day_end": str(normalized["target_day"].max().date()),
            "dayplus_values": sorted(int(value) for value in normalized["dayplus"].unique()),
        }
        return normalized, audit

    @staticmethod
    def _normalize_request_day(value: date | str | pd.Timestamp, field: str) -> pd.Timestamp:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is not None:
            raise FutureDPVAlignmentError(f"requested {field} must be timezone-naive: {value!r}")
        if parsed != parsed.normalize():
            raise FutureDPVAlignmentError(f"requested {field} must be a day value: {value!r}")
        return parsed.normalize()

    def value(
        self,
        *,
        origin_day: date | str | pd.Timestamp,
        target_day: date | str | pd.Timestamp,
        dayplus: int,
        timestamp: str | pd.Timestamp,
    ) -> float:
        origin_ts = self._normalize_request_day(origin_day, "origin_day")
        target_ts = self._normalize_request_day(target_day, "target_day")
        timestamp_ts = pd.Timestamp(timestamp)
        if timestamp_ts.tzinfo is not None:
            raise FutureDPVAlignmentError(
                f"requested timestamp must be timezone-naive: {timestamp!r}"
            )
        dayplus_int = int(dayplus)
        if (target_ts - origin_ts).days != dayplus_int:
            raise FutureDPVAlignmentError(
                "requested future DPV keys are horizon-inconsistent: "
                f"origin_day={origin_ts.date()}, target_day={target_ts.date()}, "
                f"dayplus={dayplus_int}"
            )
        if timestamp_ts.normalize() != target_ts:
            raise FutureDPVAlignmentError(
                "requested future DPV timestamp is outside target_day: "
                f"target_day={target_ts.date()}, timestamp={timestamp_ts}"
            )
        key = (origin_ts, target_ts, dayplus_int, timestamp_ts)
        try:
            return float(self._series.loc[key])
        except KeyError as exc:
            raise FutureDPVAlignmentError(
                "future DPV strict-key lookup is missing; no timestamp-only, target-only, "
                "cross-origin, or cross-horizon fallback is permitted: "
                f"origin_day={origin_ts.date()}, target_day={target_ts.date()}, "
                f"dayplus={dayplus_int}, timestamp={timestamp_ts}"
            ) from exc

    def assert_coverage(
        self,
        *,
        origin_days: Iterable[date | str | pd.Timestamp],
        required_dayplus_range: tuple[int, int],
        scope: str,
    ) -> dict:
        origins = sorted(
            {self._normalize_request_day(value, "origin_day") for value in origin_days}
        )
        start_dayplus, end_dayplus = (int(required_dayplus_range[0]), int(required_dayplus_range[1]))
        if not origins:
            raise FutureDPVAlignmentError(f"future DPV coverage request has no origins: scope={scope}")
        if start_dayplus <= 0 or end_dayplus < start_dayplus:
            raise FutureDPVAlignmentError(
                f"invalid future DPV dayplus range {required_dayplus_range}: scope={scope}"
            )

        expected_keys: list[tuple] = []
        for origin_ts in origins:
            for dayplus in range(start_dayplus, end_dayplus + 1):
                target_ts = origin_ts + pd.Timedelta(days=dayplus)
                for timestamp_ts in pd.date_range(target_ts, periods=96, freq="15min"):
                    expected_keys.append((origin_ts, target_ts, dayplus, timestamp_ts))
        expected_index = pd.MultiIndex.from_tuples(expected_keys, names=list(STRICT_KEY_COLUMNS))
        missing = expected_index.difference(self._series.index)
        if len(missing):
            examples = [
                {
                    "origin_day": str(pd.Timestamp(item[0]).date()),
                    "target_day": str(pd.Timestamp(item[1]).date()),
                    "dayplus": int(item[2]),
                    "timestamp": str(pd.Timestamp(item[3])),
                }
                for item in list(missing[:5])
            ]
            raise FutureDPVAlignmentError(
                "future DPV coverage is incomplete for exact origin/horizon requests; "
                f"scope={scope}, expected_rows={len(expected_index)}, missing_rows={len(missing)}, "
                f"examples={examples}"
            )

        audit = {
            "scope": str(scope),
            "origin_day_start": str(origins[0].date()),
            "origin_day_end": str(origins[-1].date()),
            "n_origin_days": int(len(origins)),
            "dayplus_start": start_dayplus,
            "dayplus_end": end_dayplus,
            "expected_rows": int(len(expected_index)),
            "matched_rows": int(len(expected_index)),
            "missing_rows": 0,
        }
        self._coverage_checks.append(audit)
        return dict(audit)

    def audit_snapshot(self) -> dict:
        return {
            **self._source_audit,
            "coverage_checks": [dict(item) for item in self._coverage_checks],
        }
