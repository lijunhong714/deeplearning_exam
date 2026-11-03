"""Download and verify the Electricity benchmark without extra dependencies."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Time-Series-Library/dataset/electricity/electricity.csv"
SOURCE_URL = (
    "https://huggingface.co/datasets/thuml/Time-Series-Library/"
    "resolve/main/electricity/electricity.csv"
)
EXPECTED_SHA256 = "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_csv(path):
    rows = 0
    first = previous = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        if columns[0] != "date" or len(columns) != 322:
            raise ValueError("Expected a date column and 321 numeric columns")
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate column names")
        for row in reader:
            if len(row) != len(columns):
                raise ValueError(f"Incorrect column count on data row {rows + 1}")
            current = datetime.fromisoformat(row[0])
            if previous is not None and current - previous != timedelta(hours=1):
                raise ValueError(f"Non-hourly or duplicate timestamp: {row[0]}")
            if any(not math.isfinite(float(value)) for value in row[1:]):
                raise ValueError(f"Missing or nonfinite measurement: {row[0]}")
            first = first or current
            previous = current
            rows += 1
    if rows != 26304:
        raise ValueError(f"Expected 26304 data rows, got {rows}")
    return {
        "rows": rows,
        "numeric_columns": len(columns) - 1,
        "timestamp_column": columns[0],
        "first_timestamp": first.isoformat(sep=" "),
        "last_timestamp": previous.isoformat(sep=" "),
        "frequency": "1 hour",
        "missing_or_nonfinite_values": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional proxy URL, e.g. http://127.0.0.1:7892")
    args = parser.parse_args()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        partial_path = DATA_PATH.with_suffix(".csv.part")
        proxies = {"http": args.proxy, "https": args.proxy} if args.proxy else None
        opener = build_opener(ProxyHandler(proxies))
        request = Request(SOURCE_URL, headers={"User-Agent": "electricity-course-project/1.0"})
        downloaded = 0
        next_report = 10 * 1024 * 1024
        with opener.open(request, timeout=90) as response, partial_path.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"Downloaded {downloaded / 1024**2:.0f} MiB", flush=True)
                    next_report += 10 * 1024 * 1024
        actual_hash = sha256_file(partial_path)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(f"SHA-256 mismatch: {actual_hash}; partial file retained")
        stats = verify_csv(partial_path)
        partial_path.replace(DATA_PATH)
    else:
        actual_hash = sha256_file(DATA_PATH)
        if actual_hash != EXPECTED_SHA256:
            raise ValueError(f"Existing dataset SHA-256 mismatch: {actual_hash}")
        stats = verify_csv(DATA_PATH)

    manifest = {
        "dataset": "Electricity / ECL (preprocessed hourly benchmark)",
        "source_url": SOURCE_URL,
        "original_source": "https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014",
        "preprocessing_description": "https://github.com/laiguokun/multivariate-time-series-data",
        "timestamp_note": (
            "The preprocessed CSV timestamps differ from the 2012-2014 period described "
            "for the benchmark source. Treat them as this release's time index; do not "
            "assume alignment with historical holidays or external weather data."
        ),
        "license": "CC BY 4.0",
        "relative_path": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "size_bytes": DATA_PATH.stat().st_size,
        "sha256": actual_hash,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        **stats,
    }
    manifest_path = PROJECT_ROOT / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
