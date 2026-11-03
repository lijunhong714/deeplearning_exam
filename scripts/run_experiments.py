"""Run reproducible Electricity baselines and ablations, sequentially on one GPU."""

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from importlib.metadata import version
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "Time-Series-Library"
VARIANTS = {
    "DLinear": ("DLinear", {}),
    "LSTM": ("LSTM", {}),
    "GRU": ("GRU", {}),
    "PatchTST": ("PatchTST", {}),
    "Core": ("PeriodGatedDLinear", {"period_mode": "both", "gate_mode": "dynamic"}),
    "NoPeriod": ("PeriodGatedDLinear", {"period_mode": "none"}),
    "Daily": ("PeriodGatedDLinear", {"period_mode": "daily", "gate_mode": "dynamic"}),
    "Weekly": ("PeriodGatedDLinear", {"period_mode": "weekly", "gate_mode": "dynamic"}),
    "Uniform": ("PeriodGatedDLinear", {"period_mode": "both", "gate_mode": "uniform"}),
    "Static": ("PeriodGatedDLinear", {"period_mode": "both", "gate_mode": "static"}),
}
SUITES = {
    "smoke": ["DLinear", "LSTM", "GRU", "Core"],
    "baselines": ["DLinear", "LSTM", "GRU", "PatchTST", "Core"],
    "ablation": ["NoPeriod", "Daily", "Weekly", "Uniform", "Static", "Core"],
    "all": list(VARIANTS),
}
SUMMARY_FIELDS = ["variant", "prediction_length", "seed", "MAE", "MSE", "RMSE",
                  "parameters", "training_seconds", "peak_allocated_mib", "smoke_test"]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_summaries(folder, rows):
    save_json(folder / "summary.json", rows)
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["variant"], row["prediction_length"]), []).append(row)
    aggregates = []
    for (variant, horizon), values in groups.items():
        item = {"variant": variant, "prediction_length": horizon,
                "n_seeds": len(values), "smoke_test": values[0]["smoke_test"]}
        for field in ["MAE", "MSE", "RMSE"]:
            numbers = [row[field] for row in values]
            item[field + "_mean"] = statistics.mean(numbers)
            item[field + "_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        aggregates.append(item)
    save_json(folder / "aggregate.json", aggregates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="smoke")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--smoke", action="store_true", help="limit training/evaluation; not paper results")
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--history", type=int, default=336)
    parser.add_argument("--run-name")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke or args.suite == "smoke"
    if args.history < 168 or min(args.epochs, args.batch_size) < 1:
        parser.error("history must be at least 168; epochs and batch-size must be positive")
    horizons = args.horizons or ([24] if smoke else [24, 96])
    seeds = args.seeds or ([2021] if smoke else [2021, 2022, 2023])
    if min(horizons) < 1 or len(set(seeds)) != len(seeds) or len(set(horizons)) != len(horizons):
        parser.error("horizons must be positive; horizons and seeds cannot contain duplicates")
    variants = list(dict.fromkeys(args.variants or SUITES[args.suite]))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{'smoke' if smoke else args.suite}_{stamp}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", run_name):
        parser.error("run-name must be 1-48 letters, digits, underscores, or hyphens")
    folder = PROJECT_ROOT / "outputs" / run_name
    jobs = []
    for variant in variants:
        model, options = VARIANTS[variant]
        for horizon in horizons:
            for seed in seeds:
                job_name = f"{variant}_h{horizon}_s{seed}"
                job_dir = folder / job_name
                command = [sys.executable, "-u", "run.py", "--coursework",
                           "--task_name", "long_term_forecast", "--is_training", "1",
                           "--model_id", variant, "--model", model, "--data", "custom",
                           "--root_path", "./dataset/electricity/", "--data_path", "electricity.csv",
                           "--features", "M", "--target", "OT", "--freq", "h",
                           "--seq_len", str(args.history), "--label_len", "48", "--pred_len", str(horizon),
                           "--enc_in", "321", "--dec_in", "321", "--c_out", "321",
                           "--batch_size", str(min(args.batch_size, 2) if model == "PatchTST" else args.batch_size),
                           "--num_workers", "0", "--gpu", "0", "--seed", str(seed),
                           "--train_epochs", str(1 if smoke else args.epochs), "--patience", "3",
                           "--learning_rate", "0.001", "--lradj", "cosine", "--dropout", "0.1",
                           "--d_model", "64", "--d_ff", "128", "--n_heads", "4", "--e_layers", "2",
                           "--rnn_hidden_size", "64", "--rnn_layers", "2",
                           "--checkpoints", str(job_dir / "checkpoints"),
                           "--results_root", str(job_dir / "results")]
                for key, value in options.items():
                    command.extend(["--" + key, str(value)])
                if smoke:
                    command.extend(["--max_train_batches", "3", "--max_eval_batches", "2"])
                if args.cpu:
                    command.append("--no_use_gpu")
                jobs.append({"variant": variant, "horizon": horizon, "seed": seed,
                             "job_name": job_name, "command": command})
    if args.dry_run:
        print(json.dumps({"folder": str(folder), "smoke_test": smoke, "jobs": jobs}, indent=2))
        return
    if not (REPO_ROOT / "dataset/electricity/electricity.csv").is_file():
        parser.error("Electricity CSV missing; run scripts/prepare_electricity.py first")
    folder.mkdir(parents=True, exist_ok=False)
    source_paths = ["run.py", "layers/RecurrentForecast.py", "models/LSTM.py", "models/GRU.py",
                    "models/PeriodGatedDLinear.py", "models/DLinear.py", "exp/exp_electricity_forecasting.py",
                    "data_provider/data_loader.py", "data_provider/data_factory.py", "exp/exp_basic.py",
                    "models/PatchTST.py", "layers/SelfAttention_Family.py"]
    manifest = {"status": "running", "smoke_test": smoke, "jobs": jobs,
                "python_version": sys.version,
                "packages": {name: version(name) for name in ["torch", "numpy", "pandas", "scikit-learn", "einops"]},
                "source_sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                                  for name in source_paths}}
    save_json(folder / "run_manifest.json", manifest)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    rows = []
    for index, job in enumerate(jobs, 1):
        log_path = folder / (job["job_name"] + ".log")
        print(f"[{index}/{len(jobs)}] {job['job_name']} -> {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(job["command"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            manifest.update(status="failed", failed_job=job["job_name"])
            save_json(folder / "run_manifest.json", manifest)
            print("\n".join(log_path.read_text(encoding="utf-8").splitlines()[-25:]), flush=True)
            raise SystemExit(result.returncode)
        metric_files = list((folder / job["job_name"] / "results").glob("*/metrics.json"))
        if len(metric_files) != 1:
            raise RuntimeError(f"Expected exactly one metrics file for {job['job_name']}")
        row = json.loads(metric_files[0].read_text(encoding="utf-8"))
        row["variant"] = job["variant"]
        rows.append(row)
        save_summaries(folder, rows)
        print(f"  MAE={row['MAE']:.6f}, RMSE={row['RMSE']:.6f}, "
              f"train={row['training_seconds']:.1f}s, peak={row['peak_allocated_mib']} MiB", flush=True)
    manifest["status"] = "complete"
    save_json(folder / "run_manifest.json", manifest)
    print(f"Finished. Results: {folder / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
