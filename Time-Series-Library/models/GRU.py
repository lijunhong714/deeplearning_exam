"""Classic GRU baseline with a direct multi-step prediction head."""

from torch import nn
from layers.RecurrentForecast import RecurrentForecaster


class Model(RecurrentForecaster):
    def __init__(self, configs):
        super().__init__(configs, nn.GRU)
