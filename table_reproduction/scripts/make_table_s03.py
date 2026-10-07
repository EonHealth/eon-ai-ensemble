"""Build Supplementary Table S3: Additional Model Performance Metrics for Lung-RADS Classification."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cohort1_metrics import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    evaluate,
    load_cohort,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S03.csv"

EN_DASH = "–"

# Table S3 names the models more briefly than Table 1 does; rows print CL first.
DISPLAY_NAMES = {
    "Computational Linguistics (CL) Model": "CL",
    "Llama 3.1 70B LLM": "Llama 3.1 70B",
}
PUBLISHED_ROW_ORDER = ["Computational Linguistics (CL) Model", "Llama 3.1 70B LLM"]

COLUMNS = [
    "Model",
    "Accuracy, %\n(95% CI)",
    "Macro Precision, %\n(95% CI)",
    "Macro Recall (Sensitivity), %\n(95% CI)",
    "Macro Specificity, %\n(95% CI)",
    "Macro F1-Score, %\n(95% CI)",
    "Cohen's κ\n(95% CI)",
]

# Column header -> key returned by evaluate(), with its printed precision.
METRIC_COLUMNS = {
    COLUMNS[1]: ("accuracy", 2),
    COLUMNS[2]: ("precision", 2),
    COLUMNS[3]: ("recall", 2),
    COLUMNS[4]: ("specificity", 2),
    COLUMNS[5]: ("f1", 2),
    COLUMNS[6]: ("kappa", 3),
}


def format_ci(point: float, low: float, high: float, digits: int) -> str:
    return f"{point:.{digits}f}\n({low:.{digits}f}{EN_DASH}{high:.{digits}f})"


def build_table(frame, seed: int, replicates: int) -> list[dict[str, str]]:
    per_model = evaluate(frame, seed, replicates)
    rows = []
    for name in PUBLISHED_ROW_ORDER:
        metrics = per_model[name]
        row = {COLUMNS[0]: DISPLAY_NAMES[name]}
        for column, (key, digits) in METRIC_COLUMNS.items():
            row[column] = format_ci(*metrics[key], digits=digits)
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 1 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, help="Bootstrap replicates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_cohort(args.input)
    rows = build_table(frame, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 1: {len(frame):,} reports, {len(rows)} rows")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
