"""DLinear plus historical daily/weekly templates and an input-dependent gate.

All experts use x_enc only. Decoder values, including future targets accidentally
passed by callers, cannot affect these forecasts. This is a coursework extension
of DLinear, not an implementation of a separately published architecture.
"""

import torch
from torch import nn
from models.DLinear import Model as DLinear


class HistoricalPeriodBranch(nn.Module):
    """Average complete historical cycles and learn a residual phase correction."""

    def __init__(self, seq_len, pred_len, period, hidden, dropout):
        super().__init__()
        if period < 1 or seq_len < period:
            raise ValueError(f"History length {seq_len} must cover period {period}")
        self.period = period
        self.history_length = (seq_len // period) * period
        self.correction = nn.Sequential(
            nn.Linear(period, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, period),
        )
        # Initially behave like a purely historical seasonal-average forecast.
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer("forecast_phase", torch.arange(pred_len) % period)

    def historical_template(self, x):
        history = x[:, -self.history_length:, :].transpose(1, 2)
        cycles = history.reshape(x.shape[0], x.shape[2], -1, self.period)
        return cycles.mean(dim=2)

    def forward(self, x):
        template = self.historical_template(x)
        template = template + self.correction(template)
        return template.index_select(-1, self.forecast_phase).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("PeriodGatedDLinear supports forecasting tasks only")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_mode = getattr(configs, "period_mode", "both")
        self.gate_mode = getattr(configs, "gate_mode", "dynamic")
        self.daily_period = getattr(configs, "daily_period", 24)
        self.weekly_period = getattr(configs, "weekly_period", 168)
        hidden = getattr(configs, "period_hidden_size", 32)
        gate_hidden = getattr(configs, "gate_hidden_size", 32)
        if self.period_mode not in {"none", "daily", "weekly", "both"}:
            raise ValueError("period_mode must be none, daily, weekly, or both")
        if self.gate_mode not in {"dynamic", "static", "uniform"}:
            raise ValueError("gate_mode must be dynamic, static, or uniform")
        if min(self.seq_len, self.pred_len, self.channels, hidden, gate_hidden,
               self.daily_period, self.weekly_period) < 1:
            raise ValueError("Model dimensions and periods must be positive")
        if configs.moving_avg % 2 == 0:
            raise ValueError("DLinear moving_avg must be an odd integer")
        self.base = DLinear(configs)
        self.period_branches = nn.ModuleDict()
        self.expert_names = ["DLinear"]
        for name, period in [("daily", self.daily_period), ("weekly", self.weekly_period)]:
            if self.period_mode in {name, "both"}:
                self.period_branches[name] = HistoricalPeriodBranch(
                    self.seq_len, self.pred_len, period, hidden, configs.dropout
                )
                self.expert_names.append(name)
        experts = len(self.expert_names)
        self.gate = None
        self.register_parameter("gate_logits", None)
        if experts > 1 and self.gate_mode == "dynamic":
            self.gate = nn.Sequential(
                nn.Linear(8, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, experts)
            )
            nn.init.normal_(self.gate[-1].weight, std=0.01)
            nn.init.zeros_(self.gate[-1].bias)
            with torch.no_grad():
                self.gate[-1].bias[0] = 2.0
        elif experts > 1 and self.gate_mode == "static":
            logits = torch.zeros(experts)
            logits[0] = 2.0
            self.gate_logits = nn.Parameter(logits)

    def gate_features(self, x):
        recent = x[:, -min(self.daily_period, x.shape[1]):, :]

        def lag_error(period):
            if x.shape[1] <= period:
                return torch.zeros_like(x[:, 0, :])
            return (x[:, period:, :] - x[:, :-period, :]).abs().mean(dim=1)

        return torch.stack([
            x.mean(dim=1), x.std(dim=1, unbiased=False), x[:, -1, :],
            recent.mean(dim=1), recent.std(dim=1, unbiased=False),
            x[:, -1, :] - x[:, 0, :], lag_error(self.daily_period),
            lag_error(self.weekly_period),
        ], dim=-1)

    def predict_with_weights(self, x_enc):
        if x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError("Unexpected history length or number of input channels")
        predictions = [self.base(x_enc, None, None, None)]
        predictions.extend(branch(x_enc) for branch in self.period_branches.values())
        experts = torch.stack(predictions, dim=-1)  # [B, H, C, K]
        if len(predictions) == 1 or self.gate_mode == "uniform":
            weights = x_enc.new_full(
                (x_enc.shape[0], self.channels, len(predictions)), 1.0 / len(predictions)
            )
        elif self.gate_mode == "static":
            weights = self.gate_logits.softmax(-1).expand(x_enc.shape[0], self.channels, -1)
        else:
            weights = self.gate(self.gate_features(x_enc)).softmax(dim=-1)
        prediction = (experts * weights.unsqueeze(1)).sum(dim=-1)
        return prediction, weights

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.predict_with_weights(x_enc)[0]
