"""Build Supplementary Table S1: Lung-RADS Assessment Category Distribution in Cohort 1."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from cohort1_metrics import CATEGORY_ORDER, MISSING_LABEL as MISSING_DISPLAY
from cohort2_metrics import round_half_up

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S01.csv"

REFERENCE_COLUMN = "max_rads_annotated"

COLUMNS = ["Lung-RADS Assessment Category", "Samples (%)"]


def load_reference(path: Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if REFERENCE_COLUMN not in frame.columns:
        raise ValueError(f"{path.name} is missing required column: {REFERENCE_COLUMN}")
    values = frame[REFERENCE_COLUMN].fillna(MISSING_DISPLAY).str.strip()
    return values.replace("", MISSING_DISPLAY).tolist()


def build_table(values: list[str]) -> list[dict[str, str]]:
    counts = Counter(values)
    unexpected = set(counts) - set(CATEGORY_ORDER) - {MISSING_DISPLAY}
    if unexpected:
        raise ValueError(f"Unrecognised Lung-RADS categories: {sorted(unexpected)}")

    total = len(values)
    order = [c for c in CATEGORY_ORDER if counts[c]] + [MISSING_DISPLAY]
    tallied = sum(counts[c] for c in order)
    if tallied != total:
        raise AssertionError(f"Rows total {tallied}, expected {total}")

    return [
        {
            COLUMNS[0]: category,
            COLUMNS[1]: f"{counts[category]:,} ({round_half_up(counts[category] / total * 100, 2):.2f}%)",
        }
        for category in order
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 1 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_reference(args.input)
    rows = build_table(values)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 1: {len(values):,} reports, {len(rows)} rows")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
