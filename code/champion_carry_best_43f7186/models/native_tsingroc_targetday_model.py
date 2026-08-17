from __future__ import annotations

import copy
import pickle
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.utils
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - local env may not have torch
    torch = None
    DataLoader = None
    TensorDataset = None

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_14D_ROOT = REPO_ROOT / "repo-benchmark-14days"
if str(REPO_14D_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_14D_ROOT))

from TsingRoc_code.utils.scale import RevIN  # type: ignore
from model_zw.TsingRoc_Baseline import TsingRoc_Baseline  # type: ignore

from .native_dlinear_basic_time_backbone import DLinearBasicTimeBackbone
from .native_time_features import build_holiday_feature


@dataclass
class NativeOriginExample:
    origin_day: date
    history_load: np.ndarray
    history_weather: np.ndarray
    future_weather: np.ndarray
    target_load: np.ndarray
    future_timestamps: pd.DatetimeIndex
    history_timestamps: pd.DatetimeIndex
    history_holiday: np.ndarray
    future_holiday: np.ndarray


class ShandongNativeTsingRocTargetDayModel:
    def __init__(self, config: dict):
        if torch is None:
            raise ModuleNotFoundError("torch is required for native TsingRoc target-day benchmark runs")
        self.config = copy.deepcopy(config)
        self.model_cfg = dict(self.config["model"])
        self.feature_cfg = dict(self.config.get("features", {}))
        self.seq_len = int(self.model_cfg["seq_len"])
        self.gap_steps = int(self.model_cfg["gap_steps"])
        self.pred_len = int(self.model_cfg["pred_len"])
        self.d_model = int(self.model_cfg["d_model"])
        self.dropout = float(self.model_cfg["dropout"])
        self.lr = float(self.model_cfg["lr"])
        self.epochs = int(self.model_cfg["epochs"])
        self.batch_size = int(self.model_cfg["batch_size"])
        self.max_train_samples = int(self.model_cfg["max_train_samples"])
        self.weight_decay = float(self.model_cfg["weight_decay"])
        self.normalization_method = str(self.model_cfg.get("normalization_method", "combined")).lower()
        self.revin_affine = bool(self.model_cfg.get("revin_affine", True))
        self.revin_eps = float(self.model_cfg.get("revin_eps", 1e-5))
        self.early_stop_patience = int(self.model_cfg.get("early_stop_patience", 6))
        self.day2_weight = float(self.model_cfg.get("day2_weight", 1.0))
        self.midday_penalty = float(self.model_cfg.get("midday_penalty", 1.0))
        self.midday_start = int(self.model_cfg.get("midday_start", 40))
        self.midday_end = int(self.model_cfg.get("midday_end", 64))
        self.only_output_after_gap = bool(self.model_cfg.get("only_output_after_gap", False))
        self.use_cosine_scheduler = bool(self.model_cfg.get("use_cosine_scheduler", True))
        self.time_embedding_type = str(self.model_cfg.get("time_embedding_type", "BaseTimeEmbedding"))
        self.use_multi_point_weather = bool(self.model_cfg.get("use_multi_point_weather", False))
        self.enable_weather_feature_filter = bool(self.model_cfg.get("enable_weather_feature_filter", False))
        self.weather_base_cols = list(self.feature_cfg.get("weather_base_cols", []))
        self.seed = int(self.model_cfg.get("seed", 3407))
        self.backbone_type = str(self.model_cfg.get("benchmark_backbone_type", self.model_cfg.get("backbone_type", "baseline"))).lower()
        self.use_holiday_features = bool(self.model_cfg.get("use_holiday_features", False))
        self.holiday_major_window = int(self.model_cfg.get("holiday_major_window", 2))
        self.holiday_minor_window = int(self.model_cfg.get("holiday_minor_window", 1))

        device_str = str(self.model_cfg.get("device", "cpu"))
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.model = None
        self.revin_layer = None
        self.optimizer = None
        self.scheduler = None
        self.loss_weights = None
        self.load_mean = 0.0
        self.load_std = 1.0
        self.weather_mean = None
        self.weather_std = None
        self.segment_range: tuple[int, int] | None = None
        self.training_summary: dict = {}

    def _effective_pred_len(self, segment_range: tuple[int, int]) -> int:
        return (int(segment_range[1]) - int(segment_range[0]) + 1) * 96

    def _native_model_args(self, pred_len: int) -> SimpleNamespace:
        return SimpleNamespace(
            seq_len=self.seq_len,
            pred_len=pred_len,
            d_model=self.d_model,
            dropout=self.dropout,
            time_embedding_type=self.time_embedding_type,
            use_multi_point_weather=self.use_multi_point_weather,
            use_holiday_features=self.use_holiday_features,
            channels=1,
            dlinear_moving_avg=int(self.model_cfg.get("dlinear_moving_avg", 25)),
            dlinear_individual=bool(self.model_cfg.get("dlinear_individual", True)),
            dlinear_use_load_emb=bool(self.model_cfg.get("dlinear_use_load_emb", True)),
            dlinear_weather_heads=int(self.model_cfg.get("dlinear_weather_heads", 4)),
            dlinear_weather_layers=int(self.model_cfg.get("dlinear_weather_layers", 2)),
            dlinear_weather_activation=str(self.model_cfg.get("dlinear_weather_activation", "gelu")).lower(),
        )

    def _build_model(self, weather_dim: int, pred_len: int):
        args = self._native_model_args(pred_len)
        if self.backbone_type == "dlinearbasictime":
            return DLinearBasicTimeBackbone(args, num_weather_vars=weather_dim).to(self.device)
        if self.backbone_type == "baseline":
            return TsingRoc_Baseline(args, num_weather_vars=weather_dim).to(self.device)
        raise ValueError(f"unsupported backbone_type: {self.backbone_type}")

    def _loss_weight_tensor(self, pred_len: int) -> torch.Tensor:
        weights = np.ones(pred_len, dtype=np.float32)
        for start in range(0, pred_len, 96):
            end = min(pred_len, start + 96)
            weights[start + self.midday_start : min(end, start + self.midday_end)] *= self.midday_penalty
        return torch.tensor(weights, dtype=torch.float32, device=self.device).view(1, pred_len, 1)

    def _day_weather_matrix(self, merged, target_day: date, prefix: str) -> np.ndarray | None:
        return merged.day_feature_matrix(target_day, prefix)

    def _example_from_origin(
        self,
        merged,
        origin_day: date,
        segment_range: tuple[int, int],
        *,
        required_target_load_segment: tuple[int, int] | None = None,
    ) -> NativeOriginExample | None:
        start_day, end_day = segment_range
        required_target_load_segment = required_target_load_segment or segment_range

        history_days = [origin_day - timedelta(days=offset) for offset in range(7, 0, -1)]
        history_loads = []
        history_weather_parts = []
        history_timestamps = []
        for day in history_days:
            load_vals = merged.day_load(day)
            weather_vals = self._day_weather_matrix(merged, day, "D_0__")
            if load_vals is None or weather_vals is None:
                return None
            history_loads.append(load_vals)
            history_weather_parts.append(weather_vals)
            history_timestamps.extend(pd.date_range(pd.Timestamp(day), periods=96, freq="15min"))

        gap_weather = self._day_weather_matrix(merged, origin_day, "D_0__")
        if gap_weather is None:
            return None

        future_weather_parts = []
        target_parts = []
        future_timestamps = []
        for dayplus in range(start_day, end_day + 1):
            target_day = origin_day + timedelta(days=dayplus)
            target_load = merged.day_load(target_day)
            future_weather = self._day_weather_matrix(merged, target_day, f"D_{dayplus}__")
            require_target = required_target_load_segment[0] <= dayplus <= required_target_load_segment[1]
            if future_weather is None:
                if require_target:
                    return None
                future_weather = future_weather_parts[-1].copy() if future_weather_parts else gap_weather.copy()
            if require_target:
                if target_load is None:
                    return None
                target_parts.append(target_load)
            else:
                target_parts.append(np.full(96, np.nan, dtype=np.float32) if target_load is None else target_load)
            future_weather_parts.append(future_weather)
            future_timestamps.extend(pd.date_range(pd.Timestamp(target_day), periods=96, freq="15min"))

        hist_ts = pd.DatetimeIndex(history_timestamps)
        fut_ts = pd.DatetimeIndex(future_timestamps)
        hist_holiday = build_holiday_feature(
            hist_ts,
            major_holiday_window=self.holiday_major_window,
            minor_holiday_window=self.holiday_minor_window,
        )
        fut_holiday = build_holiday_feature(
            fut_ts,
            major_holiday_window=self.holiday_major_window,
            minor_holiday_window=self.holiday_minor_window,
        )
        return NativeOriginExample(
            origin_day=origin_day,
            history_load=np.concatenate(history_loads).astype(np.float32),
            history_weather=np.concatenate(history_weather_parts, axis=0).astype(np.float32),
            future_weather=np.concatenate(future_weather_parts, axis=0).astype(np.float32),
            target_load=np.concatenate(target_parts).astype(np.float32),
            future_timestamps=fut_ts,
            history_timestamps=hist_ts,
            history_holiday=np.asarray(hist_holiday, dtype=np.int64),
            future_holiday=np.asarray(fut_holiday, dtype=np.int64),
        )

    def _examples(self, merged, origin_days: list[date], segment_range: tuple[int, int]) -> list[NativeOriginExample]:
        rows = []
        for origin_day in origin_days:
            example = self._example_from_origin(
                merged,
                origin_day,
                segment_range,
                required_target_load_segment=segment_range,
            )
            if example is not None:
                rows.append(example)
        if not rows:
            raise ValueError("no valid origin-day examples built for native TsingRoc model")
        return rows

    def _dataset_from_examples(self, examples: list[NativeOriginExample]):
        hist_load = np.stack([ex.history_load for ex in examples]).astype(np.float32)
        hist_weather = np.stack([ex.history_weather for ex in examples]).astype(np.float32)
        fut_weather = np.stack([ex.future_weather for ex in examples]).astype(np.float32)
        target = np.stack([ex.target_load for ex in examples]).astype(np.float32)
        weather_full = np.concatenate([hist_weather, fut_weather], axis=1)

        if self.normalization_method in ("traditional", "combined"):
            hist_load_n = (hist_load - self.load_mean) / self.load_std
            target_n = (target - self.load_mean) / self.load_std
        else:
            hist_load_n = hist_load
            target_n = target

        weather_n = (weather_full - self.weather_mean) / self.weather_std
        time_total = [ex.history_timestamps.append(ex.future_timestamps) for ex in examples]
        hour = np.stack([ts.hour.to_numpy(dtype=np.int64) for ts in time_total])
        dow = np.stack([ts.dayofweek.to_numpy(dtype=np.int64) for ts in time_total])
        month = np.stack([ts.month.to_numpy(dtype=np.int64) for ts in time_total])
        holiday = np.stack([np.concatenate([ex.history_holiday, ex.future_holiday]).astype(np.int64) for ex in examples])

        return TensorDataset(
            torch.tensor(hist_load_n[:, :, None], dtype=torch.float32),
            torch.tensor(weather_n, dtype=torch.float32),
            torch.tensor(hour, dtype=torch.long),
            torch.tensor(dow, dtype=torch.long),
            torch.tensor(month, dtype=torch.long),
            torch.tensor(holiday, dtype=torch.long),
            torch.tensor(target_n[:, :, None], dtype=torch.float32),
        )

    def fit(
        self,
        *,
        merged,
        train_origin_days: list[date],
        valid_origin_days: list[date],
        segment_range: tuple[int, int],
        max_train_examples: int | None = None,
    ) -> dict:
        self.segment_range = segment_range
        train_examples = self._examples(merged, train_origin_days, segment_range)
        valid_examples = self._examples(merged, valid_origin_days, segment_range)
        if max_train_examples is None:
            max_train_examples = self.max_train_samples
        if len(train_examples) > max_train_examples:
            idx = np.linspace(0, len(train_examples) - 1, max_train_examples).astype(int)
            train_examples = [train_examples[i] for i in idx]

        train_hist = np.stack([ex.history_load for ex in train_examples]).astype(np.float32)
        train_target = np.stack([ex.target_load for ex in train_examples]).astype(np.float32)
        train_weather = np.concatenate(
            [
                np.concatenate(
                    [
                        np.stack([ex.history_weather for ex in train_examples]).astype(np.float32),
                        np.stack([ex.future_weather for ex in train_examples]).astype(np.float32),
                    ],
                    axis=1,
                )
            ],
            axis=0,
        )
        self.load_mean = float(np.mean(np.concatenate([train_hist.reshape(-1), train_target.reshape(-1)])))
        self.load_std = float(np.std(np.concatenate([train_hist.reshape(-1), train_target.reshape(-1)])))
        if self.load_std < 1e-6:
            self.load_std = 1.0
        self.weather_mean = train_weather.mean(axis=(0, 1))
        self.weather_std = train_weather.std(axis=(0, 1))
        self.weather_std = np.where(self.weather_std < 1e-6, 1.0, self.weather_std)

        train_dataset = self._dataset_from_examples(train_examples)
        valid_dataset = self._dataset_from_examples(valid_examples)
        pred_len = self._effective_pred_len(segment_range)
        sample_weather = train_examples[0].future_weather
        self.model = self._build_model(sample_weather.shape[1], pred_len)
        self.revin_layer = None
        params = list(self.model.parameters())
        if self.normalization_method in ("revin", "combined"):
            self.revin_layer = RevIN(num_features=1, eps=self.revin_eps, affine=self.revin_affine).to(self.device)
            params += list(self.revin_layer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        if self.use_cosine_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, self.epochs), eta_min=0.0)
        self.loss_weights = self._loss_weight_tensor(pred_len)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        valid_tensors = [tensor.to(self.device) for tensor in valid_dataset.tensors]

        best_model_state = None
        best_revin_state = None
        best_val_acc = -float("inf")
        patience = 0
        for _ in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                x_load, x_weather, x_hour, x_dow, x_month, _x_holiday, y = [item.to(self.device) for item in batch]
                self.optimizer.zero_grad()
                if self.revin_layer is not None:
                    x_load_model_input = self.revin_layer(x_load, mode="norm")
                    pred = self.model(x_load_model_input, x_weather, (x_hour, x_dow, x_month))
                    if self.normalization_method == "combined":
                        pred_for_loss = self.revin_layer(pred, mode="denorm")
                    else:
                        pred_for_loss = pred
                else:
                    pred = self.model(x_load, x_weather, (x_hour, x_dow, x_month))
                    pred_for_loss = pred
                loss = (((pred_for_loss - y) ** 2) * self.loss_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                x_load, x_weather, x_hour, x_dow, x_month, _x_holiday, y = valid_tensors
                if self.revin_layer is not None:
                    x_load_model_input = self.revin_layer(x_load, mode="norm")
                    pred = self.model(x_load_model_input, x_weather, (x_hour, x_dow, x_month))
                    if self.normalization_method == "combined":
                        pred_eval = self.revin_layer(pred, mode="denorm")
                    else:
                        pred_eval = pred
                else:
                    pred_eval = self.model(x_load, x_weather, (x_hour, x_dow, x_month))
                pred_np = pred_eval.detach().cpu().numpy().reshape(len(valid_examples), -1)
                target_np = y.detach().cpu().numpy().reshape(len(valid_examples), -1)
                actual = target_np * self.load_std + self.load_mean if self.normalization_method in ("traditional", "combined") else target_np
                predicted = pred_np * self.load_std + self.load_mean if self.normalization_method in ("traditional", "combined") else pred_np
                val_acc = float(
                    np.mean(
                        [
                            100.0 - np.mean(np.abs((a - p) / np.where(np.abs(a) < 1e-8, 1e-8, a))) * 100.0
                            for a, p in zip(actual, predicted)
                        ]
                    )
                )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_revin_state = None if self.revin_layer is None else copy.deepcopy(self.revin_layer.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        if self.revin_layer is not None and best_revin_state is not None:
            self.revin_layer.load_state_dict(best_revin_state)
        self.training_summary = {
            "backbone_type": self.backbone_type,
            "segment_range": [segment_range[0], segment_range[1]],
            "effective_pred_len_points": int(pred_len),
            "n_train_examples": int(len(train_examples)),
            "n_valid_examples": int(len(valid_examples)),
            "best_val_acc": float(best_val_acc),
            "only_output_after_gap": bool(self.only_output_after_gap),
            "day2_weight": float(self.day2_weight),
            "midday_penalty": float(self.midday_penalty),
            "normalization_method": self.normalization_method,
        }
        return dict(self.training_summary)

    def predict_origin(
        self,
        *,
        merged,
        origin_day: date,
        segment_range: tuple[int, int] | None = None,
        required_dayplus_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("native TsingRoc model has not been trained")
        segment = segment_range or self.segment_range
        if segment is None:
            raise RuntimeError("segment_range is missing")
        required_range = required_dayplus_range or segment
        example = self._example_from_origin(
            merged,
            origin_day,
            segment,
            required_target_load_segment=required_range,
        )
        if example is None:
            raise ValueError(f"origin example unavailable: {origin_day}")

        if self.normalization_method in ("traditional", "combined"):
            hist_load = (example.history_load - self.load_mean) / self.load_std
        else:
            hist_load = example.history_load
        weather_full = np.concatenate([example.history_weather, example.future_weather], axis=0).astype(np.float32)
        weather_full = (weather_full - self.weather_mean) / self.weather_std
        time_total = example.history_timestamps.append(example.future_timestamps)
        hour = time_total.hour.to_numpy(dtype=np.int64)
        dow = time_total.dayofweek.to_numpy(dtype=np.int64)
        month = time_total.month.to_numpy(dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            x_load = torch.tensor(hist_load[None, :, None], dtype=torch.float32, device=self.device)
            x_weather = torch.tensor(weather_full[None, :, :], dtype=torch.float32, device=self.device)
            x_hour = torch.tensor(hour[None, :], dtype=torch.long, device=self.device)
            x_dow = torch.tensor(dow[None, :], dtype=torch.long, device=self.device)
            x_month = torch.tensor(month[None, :], dtype=torch.long, device=self.device)
            if self.revin_layer is not None:
                x_load_model_input = self.revin_layer(x_load, mode="norm")
                pred = self.model(x_load_model_input, x_weather, (x_hour, x_dow, x_month))
                if self.normalization_method == "combined":
                    pred = self.revin_layer(pred, mode="denorm")
            else:
                pred = self.model(x_load, x_weather, (x_hour, x_dow, x_month))
            pred = pred.detach().cpu().numpy()[0, :, 0]
            if self.normalization_method in ("traditional", "combined"):
                pred = pred * self.load_std + self.load_mean

        rows = []
        point_index = 0
        for dayplus in range(segment[0], segment[1] + 1):
            if dayplus < required_range[0] or dayplus > required_range[1]:
                continue
            day_start = (dayplus - segment[0]) * 96
            day_end = day_start + 96
            target_day = origin_day + timedelta(days=dayplus)
            actual = example.target_load[day_start:day_end]
            predicted = pred[day_start:day_end]
            timestamps = example.future_timestamps[day_start:day_end]
            for ts, act, prd in zip(timestamps, actual, predicted):
                rows.append(
                    {
                        "origin_day": origin_day.isoformat(),
                        "target_day": target_day.isoformat(),
                        "dayplus": int(dayplus),
                        "timestamp": ts.isoformat(),
                        "actual": float(act),
                        "pred": float(prd),
                        "point_index": point_index,
                    }
                )
                point_index += 1
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("native TsingRoc model has not been trained")
        payload = {
            "config": self.config,
            "segment_range": self.segment_range,
            "backbone_type": self.backbone_type,
            "training_summary": self.training_summary,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
