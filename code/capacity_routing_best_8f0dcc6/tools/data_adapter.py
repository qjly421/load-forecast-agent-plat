from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_utils import resolve_path, write_json


def _feature_allowed(column: str, base_cols: set[str]) -> bool:
    if not base_cols:
        return True
    for base in base_cols:
        if column == base or column.endswith(f"__{base}") or column.endswith(f"_{base}"):
            return True
    return False


def _load_day_frame(frame: pd.DataFrame, target_day: date) -> pd.DataFrame:
    start = pd.Timestamp(target_day)
    end = start + pd.Timedelta(hours=23, minutes=45)
    return frame.loc[start:end]


@dataclass
class MergedShandongData:
    frame: pd.DataFrame
    load_path: str
    weather_root: str
    weather_source: str
    weather_base_cols: tuple[str, ...]
    cache_path: str
    _day_load_cache: dict[date, np.ndarray | None] = field(default_factory=dict, init=False, repr=False)
    _day_feature_cache: dict[tuple[date, str], np.ndarray | None] = field(default_factory=dict, init=False, repr=False)
    _prefix_cols_cache: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _available_dates_cache: list[date] | None = field(default=None, init=False, repr=False)
    _last_complete_load_day_cache: date | None = field(default=None, init=False, repr=False)
    _last_complete_load_day_ready: bool = field(default=False, init=False, repr=False)

    def day_load(self, target_day: date) -> np.ndarray | None:
        if target_day in self._day_load_cache:
            return self._day_load_cache[target_day]
        part = _load_day_frame(self.frame[["load"]], target_day)
        if len(part) != 96:
            self._day_load_cache[target_day] = None
            return None
        values = part["load"].to_numpy(dtype=float)
        self._day_load_cache[target_day] = values
        return values

    def day_feature_matrix(self, target_day: date, prefix: str) -> np.ndarray | None:
        cache_key = (target_day, prefix)
        if cache_key in self._day_feature_cache:
            return self._day_feature_cache[cache_key]
        cols = self._columns_for_prefix(prefix)
        if not cols:
            self._day_feature_cache[cache_key] = None
            return None
        part = _load_day_frame(self.frame[cols], target_day)
        if len(part) != 96:
            self._day_feature_cache[cache_key] = None
            return None
        values = part.to_numpy(dtype=float)
        self._day_feature_cache[cache_key] = values
        return values

    def prefix_feature_columns(self, prefix: str) -> list[str]:
        return list(self._columns_for_prefix(prefix))

    def _columns_for_prefix(self, prefix: str) -> list[str]:
        cols = self._prefix_cols_cache.get(prefix)
        if cols is None:
            cols = [col for col in self.frame.columns if col.startswith(prefix)]
            if self.weather_base_cols:
                allowed = set(self.weather_base_cols)
                cols = [col for col in cols if _feature_allowed(col, allowed)]
            cols = sorted(cols, key=lambda item: item.split("__", 1)[-1])
            self._prefix_cols_cache[prefix] = cols
        return cols

    def available_dates(self) -> list[date]:
        if self._available_dates_cache is None:
            self._available_dates_cache = sorted({ts.date() for ts in self.frame.index})
        return self._available_dates_cache

    def day_is_complete(self, target_day: date) -> bool:
        return self.day_load(target_day) is not None

    def last_complete_load_day(self) -> date | None:
        if self._last_complete_load_day_ready:
            return self._last_complete_load_day_cache
        for current in reversed(self.available_dates()):
            if self.day_load(current) is not None:
                self._last_complete_load_day_cache = current
                self._last_complete_load_day_ready = True
                return current
        self._last_complete_load_day_cache = None
        self._last_complete_load_day_ready = True
        return None

    def origin_day_eligibility_report(
        self,
        origin_day: date,
        *,
        segment_range: tuple[int, int],
        required_target_load_segment: tuple[int, int] | None = None,
        history_days: int = 7,
        require_gap_weather: bool = True,
    ) -> dict:
        origin = pd.Timestamp(origin_day).date()
        required_target_load_segment = required_target_load_segment or segment_range
        base = {
            "eligible": False,
            "origin_day": origin.isoformat(),
            "segment_range": [int(segment_range[0]), int(segment_range[1])],
            "required_target_load_segment": [
                int(required_target_load_segment[0]),
                int(required_target_load_segment[1]),
            ],
            "history_days": int(history_days),
        }

        if require_gap_weather and self.day_feature_matrix(origin, "D_0__") is None:
            return {
                **base,
                "reason_code": "gap_weather_missing",
                "reason_message": f"D_0 gap weather unavailable on origin_day={origin.isoformat()}",
                "missing_origin_day": origin.isoformat(),
            }

        missing_history = []
        for offset in range(history_days, 0, -1):
            history_day = (pd.Timestamp(origin) - pd.Timedelta(days=offset)).date()
            if self.day_load(history_day) is None:
                missing_history.append(history_day.isoformat())
        if missing_history:
            return {
                **base,
                "reason_code": "history_load_missing",
                "reason_message": (
                    f"history load unavailable before origin_day={origin.isoformat()} "
                    f"first_missing_history_day={missing_history[0]}"
                ),
                "missing_history_days": missing_history,
            }

        missing_target_load = []
        missing_target_weather = []
        for dayplus in range(segment_range[0], segment_range[1] + 1):
            target_day = (pd.Timestamp(origin) + pd.Timedelta(days=dayplus)).date()
            require_target_load = required_target_load_segment[0] <= dayplus <= required_target_load_segment[1]
            if require_target_load and self.day_load(target_day) is None:
                missing_target_load.append(
                    {
                        "dayplus": int(dayplus),
                        "target_day": target_day.isoformat(),
                    }
                )
            if require_target_load and self.day_feature_matrix(target_day, f"D_{dayplus}__") is None:
                missing_target_weather.append(
                    {
                        "dayplus": int(dayplus),
                        "target_day": target_day.isoformat(),
                    }
                )

        if missing_target_load:
            first = missing_target_load[0]
            last_complete_load_day = self.last_complete_load_day()
            return {
                **base,
                "reason_code": "target_load_missing",
                "reason_message": (
                    f"target load unavailable for origin_day={origin.isoformat()} "
                    f"dayplus={first['dayplus']} target_day={first['target_day']} "
                    f"last_complete_load_day={last_complete_load_day.isoformat() if last_complete_load_day else 'unknown'}"
                ),
                "missing_target_load": missing_target_load,
                "first_missing_target_day": first["target_day"],
                "last_complete_load_day": (
                    last_complete_load_day.isoformat() if last_complete_load_day is not None else None
                ),
            }

        if missing_target_weather:
            first = missing_target_weather[0]
            return {
                **base,
                "reason_code": "target_weather_missing",
                "reason_message": (
                    f"target weather unavailable for origin_day={origin.isoformat()} "
                    f"dayplus={first['dayplus']} target_day={first['target_day']}"
                ),
                "missing_target_weather": missing_target_weather,
                "first_missing_target_day": first["target_day"],
            }

        return {
            **base,
            "eligible": True,
            "reason_code": "ok",
            "reason_message": "eligible",
        }

    def origin_day_is_eligible(
        self,
        origin_day: date,
        *,
        segment_range: tuple[int, int],
        history_days: int = 7,
        require_gap_weather: bool = True,
    ) -> bool:
        report = self.origin_day_eligibility_report(
            origin_day,
            segment_range=segment_range,
            required_target_load_segment=segment_range,
            history_days=history_days,
            require_gap_weather=require_gap_weather,
        )
        return bool(report["eligible"])

    def eligible_origin_days(
        self,
        *,
        start_day: date,
        end_day: date,
        segment_range: tuple[int, int],
        history_days: int = 7,
        require_gap_weather: bool = True,
    ) -> list[date]:
        days = pd.date_range(start_day, end_day, freq="D")
        rows: list[date] = []
        for ts in days:
            current = ts.date()
            if self.origin_day_is_eligible(
                current,
                segment_range=segment_range,
                history_days=history_days,
                require_gap_weather=require_gap_weather,
            ):
                rows.append(current)
        return rows


class ShandongD0D14Adapter:
    def __init__(
        self,
        *,
        load_path: str | Path,
        weather_root: str | Path,
        weather_source: str,
        cache_root: str | Path,
        weather_base_cols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.load_path = resolve_path(load_path)
        self.weather_root = resolve_path(weather_root)
        self.weather_source = str(weather_source)
        self.cache_root = resolve_path(cache_root)
        self.weather_base_cols = tuple(weather_base_cols or [])

    def _cache_paths(self) -> tuple[Path, Path]:
        key = {
            "load_path": str(self.load_path),
            "weather_root": str(self.weather_root),
            "weather_source": self.weather_source,
            "weather_base_cols": list(self.weather_base_cols),
        }
        digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        parquet_path = self.cache_root / f"shandong_targetday_{self.weather_source}_{digest}.parquet"
        meta_path = self.cache_root / f"shandong_targetday_{self.weather_source}_{digest}.json"
        return parquet_path, meta_path

    def load_or_build(self, *, force_rebuild: bool = False) -> MergedShandongData:
        cache_path, meta_path = self._cache_paths()
        if cache_path.exists() and not force_rebuild:
            frame = pd.read_parquet(cache_path)
            if "time" in frame.columns:
                frame["time"] = pd.to_datetime(frame["time"])
                frame = frame.set_index("time", drop=True)
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index()
            return MergedShandongData(
                frame=frame,
                load_path=str(self.load_path),
                weather_root=str(self.weather_root),
                weather_source=self.weather_source,
                weather_base_cols=self.weather_base_cols,
                cache_path=str(cache_path),
            )

        frame = self._build_from_raw()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)
        write_json(
            meta_path,
            {
                "load_path": str(self.load_path),
                "weather_root": str(self.weather_root),
                "weather_source": self.weather_source,
                "weather_base_cols": list(self.weather_base_cols),
                "cache_path": str(cache_path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
            },
        )
        return MergedShandongData(
            frame=frame,
            load_path=str(self.load_path),
            weather_root=str(self.weather_root),
            weather_source=self.weather_source,
            weather_base_cols=self.weather_base_cols,
            cache_path=str(cache_path),
        )

    def _build_from_raw(self) -> pd.DataFrame:
        if not self.load_path.exists():
            raise FileNotFoundError(f"load parquet not found: {self.load_path}")
        source_root = self.weather_root / self.weather_source
        if not source_root.exists():
            raise FileNotFoundError(f"weather source root not found: {source_root}")

        load_df = pd.read_parquet(self.load_path).copy()
        if "time" not in load_df.columns or "load" not in load_df.columns:
            raise ValueError(f"load parquet must contain time/load: {self.load_path}")
        load_df["time"] = pd.to_datetime(load_df["time"], errors="coerce")
        load_df = load_df.dropna(subset=["time"]).sort_values("time")
        load_df = load_df.set_index("time", drop=True)
        load_df = load_df[["load"]].copy()
        # Strict protocol: never backfill future load into earlier timestamps.
        load_df["load"] = pd.to_numeric(load_df["load"], errors="coerce")
        target_index = pd.DatetimeIndex(load_df.index)

        merged = load_df.copy()
        for dayplus in range(0, 15):
            delay_dir = source_root / f"D_{dayplus}"
            if not delay_dir.exists():
                raise FileNotFoundError(f"missing weather delay dir: {delay_dir}")
            delay_frame = self._load_delay_frame(delay_dir, dayplus, target_index)
            merged = merged.join(delay_frame, how="left")

        weather_cols = [col for col in merged.columns if col != "load"]
        # Strict protocol: keep raw missingness visible so preflight / day checks fail
        # instead of silently filling future weather into earlier rows.
        merged[weather_cols] = merged[weather_cols].apply(pd.to_numeric, errors="coerce")
        merged = merged.sort_index()
        return merged

    def _load_delay_frame(self, delay_dir: Path, dayplus: int, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        files = sorted(delay_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {delay_dir}")

        frames: list[pd.DataFrame] = []
        for weather_file in files:
            station = weather_file.stem
            frame = pd.read_parquet(weather_file).copy()
            if "ts" in frame.columns:
                frame = frame.rename(columns={"ts": "time"})
            if "time" not in frame.columns:
                raise ValueError(f"weather parquet must contain ts/time: {weather_file}")
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
            frame = frame.dropna(subset=["time"]).sort_values("time")

            feature_cols = [col for col in frame.columns if col != "time" and _feature_allowed(col, set(self.weather_base_cols))]
            if not feature_cols:
                continue
            rename_map = {
                col: f"D_{dayplus}__{station}__{col}"
                for col in feature_cols
            }
            frame = frame[["time", *feature_cols]].rename(columns=rename_map)
            frame = frame.set_index("time", drop=True)
            frame = frame[~frame.index.duplicated(keep="last")]
            frames.append(frame)

        if not frames:
            raise ValueError(f"no weather features selected under {delay_dir}")

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="outer")
        merged = merged.sort_index()
        upsampled = merged.reindex(merged.index.union(target_index)).sort_index()
        # Interpolate only across interior gaps. Do not backfill leading NaNs from
        # future values; leave them missing so eligibility checks reject them.
        upsampled = upsampled.interpolate(method="time", limit_area="inside").ffill()
        upsampled = upsampled.reindex(target_index)
        upsampled.index = target_index
        return upsampled
