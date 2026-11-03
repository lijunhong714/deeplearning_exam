"""Behavioral tests for forecasting shapes, historical boundaries, and gating."""

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1] / "Time-Series-Library"
sys.path.insert(0, str(REPO_ROOT))

from models import GRU, LSTM, PeriodGatedDLinear
from data_provider.data_loader import Dataset_Custom
from exp.exp_electricity_forecasting import Exp_Electricity_Forecast


def config(**overrides):
    values = dict(task_name="long_term_forecast", seq_len=336, pred_len=96, enc_in=3,
                  dropout=0.0, moving_avg=25, rnn_hidden_size=8, rnn_layers=2,
                  period_mode="both", gate_mode="dynamic", daily_period=24, weekly_period=168,
                  period_hidden_size=8, gate_hidden_size=8, augmentation_ratio=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class ForecastModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_shapes_and_no_decoder_target_leakage(self):
        for module in (LSTM, GRU, PeriodGatedDLinear):
            for horizon in (24, 96):
                with self.subTest(model=module.__name__, horizon=horizon):
                    model = module.Model(config(pred_len=horizon)).eval()
                    history = torch.randn(2, 336, 3)
                    with torch.no_grad():
                        before = model(history, None, torch.zeros(2, horizon + 48, 3), None)
                        after = model(history, None, torch.full((2, horizon + 48, 3), 1e8), None)
                    self.assertEqual(before.shape, (2, horizon, 3))
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_historical_period_extension_is_aligned_beyond_one_day(self):
        for period in (24, 168):
            length, horizon = 337, 192
            branch = PeriodGatedDLinear.HistoricalPeriodBranch(length, horizon, period, 8, 0)
            timeline = torch.arange(length + horizon, dtype=torch.float32) % period
            history = timeline[:length].reshape(1, length, 1)
            expected = timeline[length:].reshape(1, horizon, 1)
            torch.testing.assert_close(branch(history), expected)

    def test_dynamic_gates_are_probabilities_and_depend_on_history(self):
        model = PeriodGatedDLinear.Model(config()).eval()
        history = torch.randn(2, 336, 3)
        _, weights = model.predict_with_weights(history)
        changed = history.clone()
        changed[:, -24:, :] += 8
        _, changed_weights = model.predict_with_weights(changed)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))
        self.assertTrue(bool((weights >= 0).all()))
        self.assertGreater((changed_weights - weights).abs().max().item(), 1e-5)

    def test_core_is_equivariant_to_user_order(self):
        model = PeriodGatedDLinear.Model(config()).eval()
        history = torch.randn(2, 336, 3)
        order = [2, 0, 1]
        torch.testing.assert_close(model(history[:, :, order]), model(history)[:, :, order])

    def test_no_period_ablation_exactly_matches_its_dlinear(self):
        model = PeriodGatedDLinear.Model(config(period_mode="none"))
        history = torch.randn(2, 336, 3)
        torch.testing.assert_close(model(history), model.base(history, None, None, None), rtol=0, atol=0)

    def test_all_ablation_modes_have_finite_gradients(self):
        history, target = torch.randn(2, 336, 3), torch.randn(2, 96, 3)
        for periods in ("none", "daily", "weekly", "both"):
            for gate in ("dynamic", "static", "uniform"):
                with self.subTest(periods=periods, gate=gate):
                    model = PeriodGatedDLinear.Model(config(period_mode=periods, gate_mode=gate))
                    torch.nn.functional.mse_loss(model(history), target).backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))
                    for branch in model.period_branches.values():
                        self.assertGreater(branch.correction[-1].weight.grad.abs().sum().item(), 0)
                    if model.gate is not None:
                        self.assertGreater(model.gate[-1].weight.grad.abs().sum().item(), 0)

    def test_recurrent_baselines_receive_gradients(self):
        for module in (LSTM, GRU):
            model = module.Model(config())
            model(torch.randn(2, 336, 3)).square().mean().backward()
            self.assertGreater(model.encoder.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_history_must_cover_enabled_period(self):
        with self.assertRaises(ValueError):
            PeriodGatedDLinear.Model(config(seq_len=96, period_mode="weekly"))
        PeriodGatedDLinear.Model(config(seq_len=96, period_mode="daily"))

    def test_scaler_never_fits_validation_or_test_values(self):
        with tempfile.TemporaryDirectory() as folder:
            values = np.arange(300, dtype=float)
            values[210:] = 1e8
            pd.DataFrame({"date": pd.date_range("2024-01-01", periods=300, freq="h"),
                          "a": values, "OT": values * 2}).to_csv(Path(folder) / "test.csv", index=False)
            for split in ("train", "val", "test"):
                dataset = Dataset_Custom(config(), root_path=folder, flag=split,
                                         size=[168, 24, 24], features="M", data_path="test.csv", timeenc=1)
                np.testing.assert_allclose(dataset.scaler.mean_, [104.5, 209.0])
                x, y, _, _ = dataset[0]
                self.assertEqual(x.shape, (168, 2))
                self.assertEqual(y.shape, (48, 2))
                self.assertGreater(len(dataset), 0)

    def test_streaming_metrics_include_partial_batch_and_original_scale(self):
        class ZeroForecast(torch.nn.Module):
            def forward(self, x, *unused):
                return x.new_zeros(x.shape[0], 2, 2)

        for features in ("M", "MS"):
            with self.subTest(features=features), tempfile.TemporaryDirectory() as folder:
                targets = torch.arange(20, dtype=torch.float32).reshape(5, 2, 2) / 10
                dataset = torch.utils.data.TensorDataset(
                    torch.zeros(5, 3, 2), targets, torch.zeros(5, 3, 4), torch.zeros(5, 2, 4)
                )
                dataset.scale = True
                dataset.scaler = SimpleNamespace(scale_=np.array([2.0, 3.0]))
                experiment = object.__new__(Exp_Electricity_Forecast)
                experiment.args = config(model="ZeroForecast", seq_len=3, pred_len=2,
                                         features=features, label_len=0, use_amp=False, seed=42,
                                         results_root=folder, max_eval_batches=0, max_train_batches=0)
                experiment.device = torch.device("cpu")
                experiment.model = ZeroForecast()
                experiment.training_summary = {}
                experiment.data_cache = {"test": (dataset, torch.utils.data.DataLoader(dataset, batch_size=2))}
                # Suppress the JSON report; assertions below inspect saved artifacts too.
                from contextlib import redirect_stdout
                from io import StringIO
                with redirect_stdout(StringIO()):
                    metrics = experiment.test("test")
                expected = targets.numpy()[:, :, -1:] if features == "MS" else targets.numpy()
                scale = np.array([3.0]) if features == "MS" else np.array([2.0, 3.0])
                self.assertEqual(metrics["test_windows_evaluated"], 5)
                self.assertAlmostEqual(metrics["MAE"], np.abs(expected.astype(float)).mean())
                self.assertAlmostEqual(metrics["MSE"], np.square(expected.astype(float)).mean())
                self.assertAlmostEqual(metrics["raw_MAE"], np.abs(expected * scale).mean())
                saved = json.loads((Path(folder) / "test/metrics.json").read_text(encoding="utf-8"))
                self.assertFalse(saved["smoke_test"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
