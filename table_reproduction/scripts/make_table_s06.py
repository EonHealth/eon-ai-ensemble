"""Build Supplementary Table S6: Confusion Matrix for CL Lung-RADS Classification in Cohort 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cohort1_metrics import (
    CATEGORY_ORDER,
    MISSING_LABEL,
    MODEL_COLUMNS,
    REFERENCE_COLUMN,
    load_cohort,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S06.csv"

CL_NAME = "Computational Linguistics (CL) Model"
CORNER = "CL output →\nReference ↓"
TOTAL = "Total"


def build_table(frame) -> tuple[list[dict[str, str]], int]:
    reference = normalise(frame[REFERENCE_COLUMN])
    prediction = normalise(frame[MODEL_COLUMNS[CL_NAME]])

    categories = CATEGORY_ORDER + [MISSING_LABEL]
    for name, series in (("reference", reference), ("CL output", prediction)):
        unexpected = set(series) - set(categories)
        if unexpected:
            raise ValueError(f"Unrecognised {name} categories: {sorted(unexpected)}")

    index = {label: i for i, label in enumerate(categories)}
    size = len(categories)
    counts = np.bincount(
        reference.map(index).to_numpy() * size + prediction.map(index).to_numpy(),
        minlength=size * size,
    ).reshape(size, size)

    rows = []
    for label, row in zip(categories, counts):
        rows.append(
            {
                CORNER: label,
                **{column: f"{value:,}" for column, value in zip(categories, row)},
                TOTAL: f"{row.sum():,}",
            }
        )
    column_totals = counts.sum(axis=0)
    rows.append(
        {
            CORNER: TOTAL,
            **{column: f"{value:,}" for column, value in zip(categories, column_totals)},
            TOTAL: f"{counts.sum():,}",
        }
    )
    return rows, int(counts.sum() - np.trace(counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 1 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_cohort(args.input)
    rows, errors = build_table(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 1: {len(frame):,} reports, {errors} CL errors off the diagonal")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
