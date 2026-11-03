"""Coursework experiment engine: validation-only selection and streaming metrics.

The upstream experiment remains available without --coursework. This engine
avoids keeping all overlapping forecasts in memory or saving gigabytes per run.
"""

import csv
import json
import math
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


class Exp_Electricity_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        if args.loss.upper() != "MSE":
            raise ValueError("The coursework engine currently implements MSE training loss")
        self.data_cache = {}
        self.training_summary = {}

    def _build_model(self):
        return self.model_dict[self.args.model](self.args).float()

    def _get_data(self, flag):
        if flag not in self.data_cache:
            self.data_cache[flag] = data_provider(self.args, flag)
        return self.data_cache[flag]

    def _autocast(self):
        if self.args.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _forward_batch(self, batch, with_weights=False):
        x, y, x_mark, y_mark = [item.float().to(self.device) for item in batch]
        target = y[:, -self.args.pred_len:, :]
        weights = None
        with self._autocast():
            if with_weights and hasattr(self.model, "predict_with_weights"):
                prediction, weights = self.model.predict_with_weights(x)
            else:
                decoder = torch.cat([
                    y[:, :self.args.label_len, :], torch.zeros_like(target)
                ], dim=1)
                prediction = self.model(x, x_mark, decoder, y_mark)
        prediction = prediction[:, -self.args.pred_len:, :]
        if self.args.features == "MS":
            prediction, target = prediction[:, :, -1:], target[:, :, -1:]
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction {prediction.shape} != target {target.shape}")
        return prediction.float(), target, x, weights

    def _validation_loss(self, loader):
        self.model.eval()
        total = count = 0
        with torch.no_grad():
            for batch in islice(loader, self.args.max_eval_batches or None):
                prediction, target, _, _ = self._forward_batch(batch)
                total += nn.functional.mse_loss(prediction, target, reduction="sum").item()
                count += target.numel()
        if count == 0 or not math.isfinite(total):
            raise ValueError("Empty validation split or nonfinite validation loss")
        return total / count

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, val_loader = self._get_data("val")
        checkpoint_dir = Path(self.args.checkpoints) / setting
        result_dir = Path(self.args.results_root) / setting
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(result_dir / "config.json", vars(self.args))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=self.args.use_amp and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        best, stale, best_epoch = math.inf, 0, 0
        history = []
        for epoch in range(self.args.train_epochs):
            self.model.train()
            loss_sum = elements = steps = 0
            for batch in islice(train_loader, self.args.max_train_batches or None):
                optimizer.zero_grad(set_to_none=True)
                prediction, target, _, _ = self._forward_batch(batch)
                loss = nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(loss).backward()
                if self.args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * target.numel()
                elements += target.numel()
                steps += 1
            if not elements:
                raise ValueError("No training windows; check history and forecast lengths")
            val_loss = self._validation_loss(val_loader)
            row = {"epoch": epoch + 1, "train_mse": loss_sum / elements,
                   "validation_mse": val_loss, "train_batches": steps}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_loss < best:
                best, stale, best_epoch = val_loss, 0, epoch + 1
                torch.save(self.model.state_dict(), checkpoint_dir / "checkpoint.pth")
            else:
                stale += 1
            if stale >= self.args.patience:
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.training_summary = {
            "best_epoch": best_epoch, "best_validation_mse": best,
            "epochs_completed": len(history), "training_seconds": time.perf_counter() - started,
            "train_windows": len(train_loader.dataset), "validation_windows": len(val_loader.dataset),
            "peak_allocated_mib": (torch.cuda.max_memory_allocated(self.device) / 2**20
                                   if self.device.type == "cuda" else None),
        }
        write_json(result_dir / "history.json", history)
        write_json(result_dir / "training.json", self.training_summary)
        self.model.load_state_dict(torch.load(
            checkpoint_dir / "checkpoint.pth", map_location=self.device, weights_only=True
        ))
        return self.model

    def test(self, setting, test=0):
        dataset, loader = self._get_data("test")
        result_dir = Path(self.args.results_root) / setting
        result_dir.mkdir(parents=True, exist_ok=True)
        if test:
            self.model.load_state_dict(torch.load(
                Path(self.args.checkpoints) / setting / "checkpoint.pth",
                map_location=self.device, weights_only=True
            ))
            training_path = result_dir / "training.json"
            if training_path.exists():
                self.training_summary = json.loads(training_path.read_text(encoding="utf-8"))
        self.model.eval()
        absolute = squared = gate_sum = None
        windows = 0
        first_example = None
        with torch.no_grad():
            for batch in islice(loader, self.args.max_eval_batches or None):
                prediction, target, x, weights = self._forward_batch(batch, with_weights=True)
                error = prediction.cpu().numpy().astype(np.float64) - target.cpu().numpy()
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite test predictions")
                batch_absolute = np.abs(error).sum(axis=0)
                batch_squared = np.square(error).sum(axis=0)
                absolute = batch_absolute if absolute is None else absolute + batch_absolute
                squared = batch_squared if squared is None else squared + batch_squared
                windows += error.shape[0]
                if weights is not None:
                    values = weights.float().sum(dim=0).cpu().numpy().astype(np.float64)
                    gate_sum = values if gate_sum is None else gate_sum + values
                if first_example is None:
                    first_example = (x[0].cpu().numpy(), target[0].cpu().numpy(),
                                     prediction[0].cpu().numpy())
        if not windows:
            raise ValueError("No test windows")
        mae, mse = (absolute / windows).mean(), (squared / windows).mean()
        scale = dataset.scaler.scale_ if dataset.scale else np.ones(absolute.shape[-1])
        if self.args.features == "MS":
            scale = scale[-1:]
        raw_mae = (absolute * scale / windows).mean()
        raw_mse = (squared * scale**2 / windows).mean()
        metrics = {
            "model": self.args.model, "setting": setting,
            "seed": getattr(self.args, "effective_seed", self.args.seed),
            "history_length": self.args.seq_len, "prediction_length": self.args.pred_len,
            "MAE": float(mae), "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
            "raw_MAE": float(raw_mae), "raw_MSE": float(raw_mse), "raw_RMSE": float(np.sqrt(raw_mse)),
            "metric_scale": "train-standardized per channel; raw_* uses original CSV scale",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "torch_version": torch.__version__, "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "test_windows_evaluated": windows, "test_windows_available": len(dataset),
            "smoke_test": bool(self.args.max_train_batches or self.args.max_eval_batches),
            "period_mode": self.args.period_mode, "gate_mode": self.args.gate_mode,
            **self.training_summary,
        }
        manifest_path = Path(__file__).resolve().parents[2] / "data_manifest.json"
        if manifest_path.exists():
            metrics["dataset_sha256"] = json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"]
        if gate_sum is not None:
            mean_gate = gate_sum / windows
            metrics["mean_gate_weights"] = dict(zip(self.model.expert_names, mean_gate.mean(axis=0).tolist()))
            with (result_dir / "gate_weights_mean.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["channel_index", *self.model.expert_names])
                writer.writerows([index, *row] for index, row in enumerate(mean_gate))
        with (result_dir / "per_horizon.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["forecast_step", "MAE", "MSE"])
            writer.writerows((step + 1, (absolute[step] / windows).mean(),
                              (squared[step] / windows).mean()) for step in range(self.args.pred_len))
        np.savez_compressed(result_dir / "forecast_example.npz", history=first_example[0],
                            truth=first_example[1], prediction=first_example[2])
        write_json(result_dir / "metrics.json", metrics)
        np.save(result_dir / "metrics.npy", np.array([mae, mse, np.sqrt(mse)]))
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
