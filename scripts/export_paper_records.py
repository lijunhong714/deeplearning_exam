"""Copy exact lightweight records for the paper; leave original outputs intact."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

PROJECT = Path(__file__).resolve().parents[1]
GROUPS = {'paper_main': ['DLinear', 'LSTM', 'GRU', 'Core'],
          'paper_ablation_three': ['Weekly', 'Daily', 'Static']}
for group, variants in GROUPS.items():
    source = PROJECT / 'outputs' / group
    target = PROJECT / 'experiment_records' / group
    manifest_path = source / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['status'] == 'complete' and manifest['smoke_test'] is False
    with (source / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r['variant'] in variants]
    expected = {(v, str(h), str(s)) for v in variants for h in [24,96] for s in [2021,2022,2023]}
    assert {(r['variant'],r['prediction_length'],r['seed']) for r in rows} == expected
    assert len(rows) == len(expected)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, target / manifest_path.name)
    copied = []
    for row in rows:
        name = f"{row['variant']}_h{row['prediction_length']}_s{row['seed']}"
        folder, = (source / name / 'results').glob('*')
        for filename in ['metrics.json','config.json','history.json','training.json','per_horizon.csv']:
            path = folder / filename
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
            copied.append({'path':destination.relative_to(target).as_posix(),'sha256':digest})
    with (target / 'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selection={'variants':variants,'runs':len(rows),'seeds':[2021,2022,2023],'horizons':[24,96],
        'original_summary_sha256':hashlib.sha256((source/'summary.csv').read_bytes()).hexdigest(),
        'original_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'note':'Original run manifest is unchanged and may describe additional jobs; the paper selects the variants listed here.',
        'files':copied}
    (target/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{group}: exported {len(rows)} runs, {len(copied)} exact record files')
