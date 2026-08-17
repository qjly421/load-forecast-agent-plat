from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x):
        dims = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        tail = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, tail], dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class FutureTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.hour_emb = nn.Embedding(24, d_model)
        self.dow_emb = nn.Embedding(7, d_model)
        self.month_emb = nn.Embedding(13, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, hour: torch.Tensor, dow: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        feat = self.hour_emb(hour.long()) + self.dow_emb(dow.long()) + self.month_emb(month.long())
        return self.proj(feat)


class TargetDayDLinearBackbone(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        weather_dim: int,
        d_model: int,
        dropout: float,
        moving_avg: int,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.weather_dim = int(weather_dim)
        self.decomposition = SeriesDecomp(moving_avg)
        self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_seasonal.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
        self.linear_trend.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))

        self.future_weather_proj = nn.Sequential(
            nn.Linear(self.weather_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.gap_weather_proj = nn.Sequential(
            nn.Linear(self.weather_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.time_embedding = FutureTimeEmbedding(d_model=d_model, output_dim=1)
        self.gate = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        history_load: torch.Tensor,
        gap_weather: torch.Tensor,
        future_weather: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
    ) -> torch.Tensor:
        seasonal, trend = self.decomposition(history_load)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        base = (self.linear_seasonal(seasonal) + self.linear_trend(trend)).transpose(1, 2)
        weather_future = self.future_weather_proj(future_weather)
        gap_context = self.gap_weather_proj(gap_weather).mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        time_context = self.time_embedding(hour, dow, month)
        gate_input = torch.cat([base, weather_future, gap_context, time_context], dim=-1)
        alpha = self.gate(gate_input)
        return base + alpha * (weather_future + gap_context + time_context)


@dataclass
class OriginExample:
    origin_day: date
    history_load: np.ndarray
    gap_weather: np.ndarray
    future_weather: np.ndarray
    target_load: np.ndarray
    future_timestamps: pd.DatetimeIndex


class ShandongTargetDayNNModel:
    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        model_cfg = self.config["model"]
        self.feature_cfg = self.config.get("features", {})
        self.seq_len = int(model_cfg["seq_len"])
        self.gap_steps = int(model_cfg["gap_steps"])
        self.pred_len = int(model_cfg["pred_len"])
        self.d_model = int(model_cfg["d_model"])
        self.dropout = float(model_cfg["dropout"])
        self.lr = float(model_cfg["lr"])
        self.epochs = int(model_cfg["epochs"])
        self.batch_size = int(model_cfg["batch_size"])
        self.weight_decay = float(model_cfg["weight_decay"])
        self.max_train_samples = int(model_cfg["max_train_samples"])
        self.early_stop_patience = int(model_cfg["early_stop_patience"])
        self.seed = int(model_cfg["seed"])
        self.normalization_method = str(model_cfg.get("normalization_method", "combined"))
        self.revin_affine = bool(model_cfg.get("revin_affine", True))
        self.revin_eps = float(model_cfg.get("revin_eps", 1e-5))
        self.use_cosine_scheduler = bool(model_cfg.get("use_cosine_scheduler", False))
        self.midday_penalty = float(model_cfg.get("midday_penalty", 2.0))
        self.midday_start = int(model_cfg.get("midday_start", 40))
        self.midday_end = int(model_cfg.get("midday_end", 64))
        self.daily_weights = list(model_cfg.get("daily_weights", [1.0] * max(1, self.pred_len // 96)))
        self.moving_avg = int(model_cfg.get("dlinear_moving_avg", 25))

        device_str = str(model_cfg.get("device", "cpu"))
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.model: TargetDayDLinearBackbone | None = None
        self.revin_layer: RevIN | None = None
        self.optimizer = None
        self.scheduler = None
        self.loss_weights: torch.Tensor | None = None
        self.load_mean = 0.0
        self.load_std = 1.0
        self.gap_weather_mean = None
        self.gap_weather_std = None
        self.future_weather_mean = None
        self.future_weather_std = None
        self.segment_range: tuple[int, int] | None = None

    def _example_from_origin(
        self,
        merged,
        origin_day: date,
        segment_range: tuple[int, int],
        *,
        required_target_load_segment: tuple[int, int] | None = None,
    ) -> OriginExample | None:
        start_day, end_day = segment_range
        required_target_load_segment = required_target_load_segment or segment_range
        history_days = [origin_day - timedelta(days=offset) for offset in range(7, 0, -1)]
        history_loads = []
        for day in history_days:
            values = merged.day_load(day)
            if values is None:
                return None
            history_loads.append(values)
        gap_weather = merged.day_feature_matrix(origin_day, "D_0__")
        if gap_weather is None:
            return None

        future_weather_parts = []
        target_parts = []
        timestamps = []
        for dayplus in range(start_day, end_day + 1):
            target_day = origin_day + timedelta(days=dayplus)
            target_load = merged.day_load(target_day)
            future_weather = merged.day_feature_matrix(target_day, f"D_{dayplus}__")
            require_target_load = required_target_load_segment[0] <= dayplus <= required_target_load_segment[1]
            if future_weather is None:
                if require_target_load:
                    return None
                future_weather = gap_weather.copy()
            if require_target_load:
                if target_load is None:
                    return None
                target_parts.append(target_load)
            else:
                if target_load is None:
                    target_parts.append(np.full(96, np.nan, dtype=np.float32))
                else:
                    target_parts.append(target_load)
            future_weather_parts.append(future_weather)
            timestamps.extend(pd.date_range(pd.Timestamp(target_day), periods=96, freq="15min"))

        history_load = np.concatenate(history_loads).astype(np.float32)
        future_weather = np.concatenate(future_weather_parts, axis=0).astype(np.float32)
        target_load = np.concatenate(target_parts).astype(np.float32)
        return OriginExample(
            origin_day=origin_day,
            history_load=history_load,
            gap_weather=gap_weather.astype(np.float32),
            future_weather=future_weather,
            target_load=target_load,
            future_timestamps=pd.DatetimeIndex(timestamps),
        )

    def _examples(self, merged, origin_days: list[date], segment_range: tuple[int, int]) -> list[OriginExample]:
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
            raise ValueError("no valid origin-day examples built for NN model")
        return rows

    def _loss_weight_tensor(self, pred_len: int, segment_range: tuple[int, int]) -> torch.Tensor:
        start_day, end_day = segment_range
        weights = []
        for dayplus in range(start_day, end_day + 1):
            idx = min(max(dayplus - 1, 0), len(self.daily_weights) - 1)
            day_weight = float(self.daily_weights[idx])
            day_arr = np.full(96, day_weight, dtype=np.float32)
            day_arr[self.midday_start : self.midday_end] *= self.midday_penalty
            weights.append(day_arr)
        flat = np.concatenate(weights).astype(np.float32)
        if len(flat) != pred_len:
            raise ValueError(f"weight length mismatch: {len(flat)} != {pred_len}")
        return torch.tensor(flat, dtype=torch.float32, device=self.device).view(1, pred_len, 1)

    def _dataset_from_examples(self, examples: list[OriginExample]):
        hist = np.stack([ex.history_load for ex in examples]).astype(np.float32)
        gap_w = np.stack([ex.gap_weather for ex in examples]).astype(np.float32)
        fut_w = np.stack([ex.future_weather for ex in examples]).astype(np.float32)
        target = np.stack([ex.target_load for ex in examples]).astype(np.float32)
        hist_n = (hist - self.load_mean) / self.load_std
        target_n = (target - self.load_mean) / self.load_std
        gap_n = (gap_w - self.gap_weather_mean) / self.gap_weather_std
        fut_n = (fut_w - self.future_weather_mean) / self.future_weather_std

        timestamps = pd.DatetimeIndex(np.concatenate([ex.future_timestamps.values for ex in examples]))
        time_len = fut_n.shape[1]
        hour = np.stack([ex.future_timestamps.hour.to_numpy(dtype=np.int64) for ex in examples])
        dow = np.stack([ex.future_timestamps.dayofweek.to_numpy(dtype=np.int64) for ex in examples])
        month = np.stack([ex.future_timestamps.month.to_numpy(dtype=np.int64) for ex in examples])

        dataset = TensorDataset(
            torch.tensor(hist_n[:, :, None], dtype=torch.float32),
            torch.tensor(gap_n, dtype=torch.float32),
            torch.tensor(fut_n, dtype=torch.float32),
            torch.tensor(hour, dtype=torch.long),
            torch.tensor(dow, dtype=torch.long),
            torch.tensor(month, dtype=torch.long),
            torch.tensor(target_n[:, :, None], dtype=torch.float32),
        )
        return dataset

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
        train_gap = np.stack([ex.gap_weather for ex in train_examples]).astype(np.float32)
        train_fut = np.stack([ex.future_weather for ex in train_examples]).astype(np.float32)
        train_target = np.stack([ex.target_load for ex in train_examples]).astype(np.float32)
        self.load_mean = float(np.mean(np.concatenate([train_hist.reshape(-1), train_target.reshape(-1)])))
        self.load_std = float(np.std(np.concatenate([train_hist.reshape(-1), train_target.reshape(-1)])))
        if self.load_std < 1e-6:
            self.load_std = 1.0
        self.gap_weather_mean = train_gap.mean(axis=(0, 1))
        self.gap_weather_std = train_gap.std(axis=(0, 1))
        self.gap_weather_std = np.where(self.gap_weather_std < 1e-6, 1.0, self.gap_weather_std)
        self.future_weather_mean = train_fut.mean(axis=(0, 1))
        self.future_weather_std = train_fut.std(axis=(0, 1))
        self.future_weather_std = np.where(self.future_weather_std < 1e-6, 1.0, self.future_weather_std)

        train_dataset = self._dataset_from_examples(train_examples)
        valid_dataset = self._dataset_from_examples(valid_examples)
        pred_len = (segment_range[1] - segment_range[0] + 1) * 96
        weather_dim = train_examples[0].future_weather.shape[1]
        self.model = TargetDayDLinearBackbone(
            seq_len=self.seq_len,
            pred_len=pred_len,
            weather_dim=weather_dim,
            d_model=self.d_model,
            dropout=self.dropout,
            moving_avg=self.moving_avg,
        ).to(self.device)
        self.revin_layer = RevIN(num_features=1, eps=self.revin_eps, affine=self.revin_affine).to(self.device)
        params = list(self.model.parameters()) + list(self.revin_layer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        if self.use_cosine_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, self.epochs), eta_min=0.0)
        self.loss_weights = self._loss_weight_tensor(pred_len, segment_range)

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        valid_tensors = [tensor.to(self.device) for tensor in valid_dataset.tensors]

        best_model_state = None
        best_revin_state = None
        best_val_acc = -float("inf")
        patience = 0
        for epoch in tqdm(range(self.epochs), desc="NN training", unit="epoch"):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                hist, gap_w, fut_w, hour, dow, month, target = [item.to(self.device) for item in batch]
                self.optimizer.zero_grad()
                hist_norm = self.revin_layer(hist, mode="norm")
                pred = self.model(hist_norm, gap_w, fut_w, hour, dow, month)
                pred = self.revin_layer(pred, mode="denorm")
                loss = (((pred - target) ** 2) * self.loss_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total_loss += float(loss.item())
            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()
            with torch.no_grad():
                hist, gap_w, fut_w, hour, dow, month, target = valid_tensors
                hist_norm = self.revin_layer(hist, mode="norm")
                pred = self.model(hist_norm, gap_w, fut_w, hour, dow, month)
                pred = self.revin_layer(pred, mode="denorm")
                pred_np = pred.detach().cpu().numpy().reshape(len(valid_examples), -1)
                target_np = target.detach().cpu().numpy().reshape(len(valid_examples), -1)
                actual = target_np * self.load_std + self.load_mean
                predicted = pred_np * self.load_std + self.load_mean
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
                best_revin_state = copy.deepcopy(self.revin_layer.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        if best_revin_state is not None:
            self.revin_layer.load_state_dict(best_revin_state)
        return {
            "n_train_examples": len(train_examples),
            "n_valid_examples": len(valid_examples),
            "best_val_acc": best_val_acc,
            "segment_range": [segment_range[0], segment_range[1]],
        }

    def predict_origin(
        self,
        *,
        merged,
        origin_day: date,
        segment_range: tuple[int, int] | None = None,
        required_dayplus_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if self.model is None or self.revin_layer is None:
            raise RuntimeError("NN model has not been trained")
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
        hist = ((example.history_load - self.load_mean) / self.load_std)[None, :, None]
        gap_w = ((example.gap_weather - self.gap_weather_mean) / self.gap_weather_std)[None, :, :]
        fut_w = ((example.future_weather - self.future_weather_mean) / self.future_weather_std)[None, :, :]
        hour = example.future_timestamps.hour.to_numpy(dtype=np.int64)[None, :]
        dow = example.future_timestamps.dayofweek.to_numpy(dtype=np.int64)[None, :]
        month = example.future_timestamps.month.to_numpy(dtype=np.int64)[None, :]

        self.model.eval()
        with torch.no_grad():
            hist_t = torch.tensor(hist, dtype=torch.float32, device=self.device)
            gap_t = torch.tensor(gap_w, dtype=torch.float32, device=self.device)
            fut_t = torch.tensor(fut_w, dtype=torch.float32, device=self.device)
            hour_t = torch.tensor(hour, dtype=torch.long, device=self.device)
            dow_t = torch.tensor(dow, dtype=torch.long, device=self.device)
            month_t = torch.tensor(month, dtype=torch.long, device=self.device)
            hist_n = self.revin_layer(hist_t, mode="norm")
            pred = self.model(hist_n, gap_t, fut_t, hour_t, dow_t, month_t)
            pred = self.revin_layer(pred, mode="denorm").detach().cpu().numpy()[0, :, 0]
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
                        "dayplus": dayplus,
                        "timestamp": ts.isoformat(),
                        "actual": float(act),
                        "pred": float(prd),
                        "point_index": point_index,
                    }
                )
                point_index += 1
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        if self.model is None or self.revin_layer is None:
            raise RuntimeError("NN model has not been trained")
        payload = {
            "config": self.config,
            "segment_range": self.segment_range,
            "model_state": self.model.state_dict(),
            "revin_state": self.revin_layer.state_dict(),
            "load_mean": self.load_mean,
            "load_std": self.load_std,
            "gap_weather_mean": self.gap_weather_mean,
            "gap_weather_std": self.gap_weather_std,
            "future_weather_mean": self.future_weather_mean,
            "future_weather_std": self.future_weather_std,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
