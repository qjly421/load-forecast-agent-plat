from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, end], dim=1)
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
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor, month: torch.Tensor) -> torch.Tensor:
        time_feat = self.hour_emb(hour.long()) + self.dow_emb(dow.long()) + self.month_emb(month.long())
        return self.proj(time_feat)


class WeatherVariableEncoder(nn.Module):
    def __init__(
        self,
        pred_len: int,
        num_weather_vars: int,
        d_model: int,
        dropout: float,
        nhead: int,
        num_layers: int,
        activation,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.num_weather_vars = int(num_weather_vars)
        self.inverted_proj = nn.Linear(self.pred_len, d_model)
        self.var_emb = nn.Embedding(max(self.num_weather_vars, 1), d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(4 * d_model, 128),
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, self.pred_len)

    def forward(self, future_weather: torch.Tensor) -> torch.Tensor:
        if future_weather.size(-1) == 0:
            return future_weather
        tokens = future_weather.transpose(1, 2)
        token_ids = torch.arange(tokens.size(1), device=tokens.device)
        token_emb = self.var_emb(token_ids).unsqueeze(0)
        tokens = self.inverted_proj(tokens) + token_emb
        encoded = self.encoder(tokens)
        return self.out_proj(encoded).transpose(1, 2)


class DLinearBasicTimeBackbone(nn.Module):
    def __init__(self, args, num_weather_vars: int = 10):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.channels = int(getattr(args, "channels", 1))
        self.individual = bool(getattr(args, "dlinear_individual", True))
        self.use_load_emb = bool(getattr(args, "dlinear_use_load_emb", True))
        moving_avg = int(getattr(args, "dlinear_moving_avg", 25))
        d_model = int(args.d_model)
        dropout = float(args.dropout)
        weather_heads = int(getattr(args, "dlinear_weather_heads", 4))
        weather_layers = int(getattr(args, "dlinear_weather_layers", 2))
        weather_activation_name = str(getattr(args, "dlinear_weather_activation", "gelu")).lower()
        if weather_activation_name == "gelu":
            weather_activation = "gelu"
        elif weather_activation_name == "relu":
            weather_activation = "relu"
        elif weather_activation_name == "sigmoid":
            weather_activation = torch.sigmoid
        elif weather_activation_name == "silu":
            weather_activation = F.silu
        else:
            raise ValueError(f"Unsupported dlinear_weather_activation: {weather_activation_name}")
        self.decomposition = SeriesDecomp(moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.linear_trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            for idx in range(self.channels):
                self.linear_seasonal[idx].weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
                self.linear_trend[idx].weight = nn.Parameter(
                    (1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len)
                )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
            self.linear_trend.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))

        self.weather_dim = int(num_weather_vars)
        self.weather_encoder = None
        self.weather_fusion = None
        if self.weather_dim > 0:
            self.weather_encoder = WeatherVariableEncoder(
                pred_len=self.pred_len,
                num_weather_vars=self.weather_dim,
                d_model=d_model,
                dropout=dropout,
                nhead=max(1, weather_heads),
                num_layers=max(1, weather_layers),
                activation=weather_activation,
            )
            self.weather_fusion = nn.Linear(self.weather_dim + self.channels, self.channels)

        self.time_embedding = None
        if self.use_load_emb:
            self.time_embedding = FutureTimeEmbedding(d_model=d_model, output_dim=self.channels)

        self.fusion_gate = None
        if self.weather_dim > 0:
            gate_hidden = max(d_model // 2, 16)
            self.fusion_gate = nn.Sequential(
                nn.Linear(24 + self.channels, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.channels),
                nn.Sigmoid(),
            )

    def _dlinear_forward(self, history_load: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(history_load)
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)
        if self.individual:
            seasonal_output = []
            trend_output = []
            for idx in range(self.channels):
                seasonal_output.append(self.linear_seasonal[idx](seasonal_init[:, idx, :]))
                trend_output.append(self.linear_trend[idx](trend_init[:, idx, :]))
            seasonal_output = torch.stack(seasonal_output, dim=1)
            trend_output = torch.stack(trend_output, dim=1)
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        return (seasonal_output + trend_output).transpose(1, 2)

    def forward(self, history_load: torch.Tensor, weather: torch.Tensor, time_features):
        base_output = self._dlinear_forward(history_load)

        if self.use_load_emb and len(time_features) >= 3:
            hour, dow, month = time_features[:3]
            future_hour = hour[:, -self.pred_len :]
            future_dow = dow[:, -self.pred_len :]
            future_month = month[:, -self.pred_len :]
            load_time_feat = self.time_embedding(future_hour, future_dow, future_month)
            gate = torch.sigmoid(load_time_feat)
            base_output = base_output + gate * load_time_feat

        if self.weather_encoder is None or self.weather_fusion is None:
            return base_output

        future_weather = weather[:, -self.pred_len :, :]
        weather_repr = self.weather_encoder(future_weather)
        fused = torch.cat([weather_repr, base_output], dim=-1)
        fused_output = self.weather_fusion(fused)

        if self.fusion_gate is not None and len(time_features) >= 1:
            hour = time_features[0][:, -self.pred_len :]
            hour_onehot = F.one_hot(hour.long().clamp(0, 23), num_classes=24).float()
            gate_input = torch.cat([hour_onehot, base_output.detach()], dim=-1)
            alpha = self.fusion_gate(gate_input)
        else:
            alpha = torch.tensor(0.5, device=base_output.device)

        return alpha * fused_output + (1.0 - alpha) * base_output
