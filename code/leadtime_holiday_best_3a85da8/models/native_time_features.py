from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

try:
    import chinese_calendar as calendar
except ModuleNotFoundError:  # pragma: no cover - remote env may vary
    calendar = None


def _holiday_level(day: date, *, major_window: int, minor_window: int) -> int:
    if calendar is None:
        return 0
    try:
        is_holiday, holiday_name = calendar.get_holiday_detail(day)
    except Exception:  # pragma: no cover - upstream holiday table may not cover all years
        return 0
    if not is_holiday:
        return 0
    if holiday_name == "Spring Festival":
        return 3
    if holiday_name in {"National Day", "Labour Day"}:
        return 2
    return 1


def build_holiday_feature(
    timestamps,
    *,
    major_holiday_window: int = 2,
    minor_holiday_window: int = 1,
) -> np.ndarray:
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(ts) == 0:
        return np.asarray([], dtype=np.int64)
    day_values = sorted({item.date() for item in ts})
    day_to_value: dict[date, int] = {}

    for current_day in day_values:
        current_value = _holiday_level(
            current_day,
            major_window=major_holiday_window,
            minor_window=minor_holiday_window,
        )
        if current_value > 0:
            day_to_value[current_day] = max(day_to_value.get(current_day, 0), current_value)
            window = major_holiday_window if current_value >= 2 else minor_holiday_window
            for offset in range(1, max(0, int(window)) + 1):
                prev_day = current_day - timedelta(days=offset)
                next_day = current_day + timedelta(days=offset)
                day_to_value[prev_day] = max(day_to_value.get(prev_day, 0), current_value)
                day_to_value[next_day] = max(day_to_value.get(next_day, 0), current_value)

    return np.asarray([day_to_value.get(item.date(), 0) for item in ts], dtype=np.int64)
