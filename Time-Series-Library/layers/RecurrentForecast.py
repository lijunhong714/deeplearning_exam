"""Conventional multivariate RNN encoders with direct multi-horizon heads."""

from torch import nn


class RecurrentForecaster(nn.Module):
    def __init__(self, configs, cell):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("The LSTM/GRU baselines support forecasting tasks only")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        hidden = getattr(configs, "rnn_hidden_size", 64)
        layers = getattr(configs, "rnn_layers", 2)
        if min(self.seq_len, self.pred_len, self.channels, hidden, layers) < 1:
            raise ValueError("Sequence lengths, channels, and RNN dimensions must be positive")
        self.encoder = cell(
            input_size=self.channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=configs.dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, self.pred_len * self.channels)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError("Unexpected history length or number of input channels")
        _, hidden = self.encoder(x_enc)
        if isinstance(hidden, tuple):  # LSTM: (hidden state, cell state)
            hidden = hidden[0]
        return self.head(hidden[-1]).reshape(-1, self.pred_len, self.channels)
