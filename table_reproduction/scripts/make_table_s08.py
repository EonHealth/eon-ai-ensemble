"""Build Supplementary Table S8: Additional Characteristic-Level Performance Results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cohort2_metrics import (
    CHARACTERISTICS,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    MODELS,
    NUMERIC_CHARACTERISTICS,
    evaluate,
    load_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S08.csv"

EN_DASH = "–"
NOT_APPLICABLE = EN_DASH

COLUMNS = [
    "Characteristic",
    "Model",
    "Accuracy, %\n(95% CI)",
    "Macro Precision, %\n(95% CI)",
    "Macro Recall (Sensitivity), %\n(95% CI)",
    "Macro Specificity, %\n(95% CI)",
    "Macro F1-Score, %\n(95% CI)",
    "Cohen's κ\n(95% CI)",
]

# Column header -> key returned by evaluate().
CATEGORICAL_COLUMNS = {
    COLUMNS[3]: "precision",
    COLUMNS[4]: "recall",
    COLUMNS[5]: "specificity",
    COLUMNS[6]: "f1",
    COLUMNS[7]: "kappa",
}


def format_ci(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f}\n({low:.{digits}f}{EN_DASH}{high:.{digits}f})"


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    standalone, _, _ = evaluate(blocks, seed, replicates, categorical=True)

    rows = []
    for characteristic in CHARACTERISTICS:
        for model in MODELS:
            metrics = standalone[(characteristic, model)]
            row = {
                COLUMNS[0]: characteristic,
                COLUMNS[1]: model,
                COLUMNS[2]: format_ci(*metrics["accuracy"]),
            }
            for column, key in CATEGORICAL_COLUMNS.items():
                if characteristic in NUMERIC_CHARACTERISTICS:
                    row[column] = NOT_APPLICABLE
                else:
                    row[column] = format_ci(*metrics[key], digits=3 if key == "kappa" else 2)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 2 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, help="Bootstrap replicates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = load_predictions(args.input)
    rows = build_table(blocks, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 2: {len(blocks[('Size', 'CL')]):,} reports, {len(rows)} rows")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
