"""Validate recorded runs and generate paper tables without retraining."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RECORDS = PROJECT / 'outputs'
if not (RECORDS / 'paper_main' / 'summary.csv').is_file():
    RECORDS = PROJECT / 'experiment_records'
SEEDS = [2021, 2022, 2023]
GROUPS = {'paper_main': ['DLinear', 'LSTM', 'GRU', 'Core'],
          'paper_ablation_three': ['Weekly', 'Daily', 'Static']}
records = {}
sources = []
for group, variants in GROUPS.items():
    base = RECORDS / group
    manifest = json.loads((base / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['smoke_test'] is False
    if group == 'paper_main':
        recorded_source_hashes = manifest['source_sha256']
    else:
        assert recorded_source_hashes == manifest['source_sha256'], 'Source differs across suites'
    summary = list(csv.DictReader((base / 'summary.csv').open(encoding='utf-8-sig', newline='')))
    selected = [r for r in summary if r['variant'] in variants]
    assert len(selected) == len(variants) * 2 * len(SEEDS)
    for variant in variants:
        for horizon in [24, 96]:
            runs = []
            for seed in SEEDS:
                row, = [r for r in summary if r['variant'] == variant and
                        int(r['prediction_length']) == horizon and int(r['seed']) == seed]
                folder, = (base / f'{variant}_h{horizon}_s{seed}' / 'results').glob('*')
                metric_path = folder / 'metrics.json'
                metrics = json.loads(metric_path.read_text(encoding='utf-8'))
                config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
                assert metrics['smoke_test'] is False
                assert metrics['seed'] == seed and metrics['history_length'] == 336
                assert metrics['prediction_length'] == horizon
                assert config['train_epochs'] == 20 and config['patience'] == 3
                assert config['features'] == 'M' and config['enc_in'] == 321
                assert metrics['test_windows_evaluated'] == metrics['test_windows_available']
                assert metrics['test_windows_evaluated'] == {24: 5237, 96: 5165}[horizon]
                assert config['max_train_batches'] == config['max_eval_batches'] == 0
                assert metrics['gpu_name'] == 'NVIDIA GeForce RTX 3050 Laptop GPU'
                for key in ['MAE', 'RMSE', 'training_seconds', 'peak_allocated_mib']:
                    assert math.isfinite(metrics[key])
                    assert math.isclose(float(row[key]), metrics[key], rel_tol=1e-12)
                steps = list(csv.DictReader((folder / 'per_horizon.csv').open(newline='')))
                assert [int(r['forecast_step']) for r in steps] == list(range(1, horizon + 1))
                assert math.isclose(st.mean(float(r['MAE']) for r in steps), metrics['MAE'], rel_tol=1e-12)
                assert math.isclose(st.mean(float(r['MSE']) for r in steps), metrics['MSE'], rel_tol=1e-12)
                expected_period = {'Core': 'both', 'Weekly': 'weekly', 'Daily': 'daily', 'Static': 'both'}
                if variant in expected_period:
                    assert metrics['period_mode'] == expected_period[variant]
                    assert metrics['gate_mode'] == ('static' if variant == 'Static' else 'dynamic')
                sources.append({'variant': variant, 'horizon': horizon, 'seed': seed,
                                'path': str(metric_path.relative_to(PROJECT)),
                                'sha256': hashlib.sha256(metric_path.read_bytes()).hexdigest()})
                runs.append(metrics)
            assert len({r['parameters'] for r in runs}) == 1
            records[f'{variant}_{horizon}'] = {
                **{key: {'mean': st.mean(r[key] for r in runs), 'sd': st.stdev(r[key] for r in runs)}
                   for key in ['MAE', 'RMSE', 'training_seconds', 'peak_allocated_mib']},
                'parameters': runs[0]['parameters'],
                'epochs_completed': [r['epochs_completed'] for r in runs],
            }

assert len({json.loads((PROJECT / source['path']).read_text())['dataset_sha256'] for source in sources}) == 1
(HERE / 'tables').mkdir(exist_ok=True)

def error(variant, horizon, metric, best=False):
    result = records[f'{variant}_{horizon}'][metric]
    value = f"{result['mean']:.4f}\\pm{result['sd']:.4f}"
    return '$' + (r'\mathbf{' + value + '}' if best else value) + '$'

def write(name, text):
    (HERE / 'tables' / name).write_text(text, encoding='utf-8')

main = [r'\begin{table*}[t]', r'\centering',
        r'\caption{对比实验结果（3个种子的均值$\pm$样本标准差，误差越小越好；粗体为各列最优）}',
        r'\label{tab:main}', r'\vspace{4pt}', r'\small\setlength{\tabcolsep}{9pt}',
        r'\begin{tabular}{lcccc}\toprule',
        r'模型 & \multicolumn{2}{c}{$H=24$} & \multicolumn{2}{c}{$H=96$}\\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & MAE & RMSE & MAE & RMSE\\\midrule']
for variant in GROUPS['paper_main']:
    main.append(variant + ' & ' + ' & '.join(error(variant, h, m, variant == 'Core')
                for h in [24, 96] for m in ['MAE', 'RMSE']) + r'\\')
main += [r'\bottomrule\end{tabular}', r'\end{table*}']
write('main_results.tex', '\n'.join(main))

ablation = [r'\begin{table*}[t]', r'\centering',
            r'\caption{三项消融实验结果（3个种子的均值$\pm$样本标准差；完整模型复用对比实验结果）}',
            r'\label{tab:ablation}', r'\vspace{4pt}', r'\small\setlength{\tabcolsep}{7pt}',
            r'\begin{tabular}{lcccc}\toprule',
            r'模型变体 & \multicolumn{2}{c}{$H=24$} & \multicolumn{2}{c}{$H=96$}\\',
            r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & MAE & RMSE & MAE & RMSE\\\midrule']
for variant, label in [('Core', '完整模型 Core'), ('Weekly', '去除日分支（Weekly）'),
                       ('Daily', '去除周分支（Daily）'), ('Static', '去除动态门控（Static）')]:
    ablation.append(label + ' & ' + ' & '.join(error(variant, h, m, variant == 'Core')
                    for h in [24, 96] for m in ['MAE', 'RMSE']) + r'\\')
ablation += [r'\bottomrule\end{tabular}', r'\end{table*}']
write('ablation_results.tex', '\n'.join(ablation))

efficiency = [r'\begin{table*}[t]', r'\centering',
              r'\caption{对比模型的实测资源开销（训练时间为3个种子的均值$\pm$样本标准差；显存为GPU张量峰值）}',
              r'\label{tab:efficiency}', r'\vspace{4pt}', r'\small\setlength{\tabcolsep}{7pt}',
              r'\begin{tabular}{lrrrrrr}\toprule',
              r' & \multicolumn{3}{c}{$H=24$} & \multicolumn{3}{c}{$H=96$}\\',
              r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
              r'模型 & 参数量 & 时间/s & 显存/MiB & 参数量 & 时间/s & 显存/MiB\\\midrule']
for variant in GROUPS['paper_main']:
    cells = []
    for horizon in [24, 96]:
        r = records[f'{variant}_{horizon}']
        assert r['peak_allocated_mib']['sd'] == 0
        cells += [f"{r['parameters']:,}",
                  f"${r['training_seconds']['mean']:.1f}\\pm{r['training_seconds']['sd']:.1f}$",
                  f"{r['peak_allocated_mib']['mean']:.2f}"]
    efficiency.append(variant + ' & ' + ' & '.join(cells) + r'\\')
efficiency += [r'\bottomrule\end{tabular}', r'\end{table*}']
write('efficiency_results.tex', '\n'.join(efficiency))

comparisons = {}
for h in [24, 96]:
    core, base = (records[f'{v}_{h}'] for v in ['Core', 'DLinear'])
    comparisons[h] = {
        'mae_reduction_vs_dlinear_percent': (1 - core['MAE']['mean'] / base['MAE']['mean']) * 100,
        'rmse_reduction_vs_dlinear_percent': (1 - core['RMSE']['mean'] / base['RMSE']['mean']) * 100,
        **{f'{v}_mae_increase_vs_core_percent': (records[f'{v}_{h}']['MAE']['mean'] / core['MAE']['mean'] - 1) * 100
           for v in ['Weekly', 'Daily', 'Static']},
    }
(HERE / 'results_summary.json').write_text(json.dumps({'seeds': SEEDS, 'runs': len(sources),
    'aggregates': records, 'comparisons': comparisons, 'sources': sources}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'validated_runs': len(sources), 'comparisons': comparisons}, ensure_ascii=False, indent=2))
