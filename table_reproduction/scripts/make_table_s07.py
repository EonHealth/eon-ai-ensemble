"""Build Supplementary Table S7: Category-Specific Model Performance for Lung-RADS Classification."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cohort1_metrics import (
    CATEGORY_ORDER,
    CLASS_METRICS,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    MISSING_LABEL,
    MODEL_COLUMNS,
    REFERENCE_COLUMN,
    evaluate,
    evaluate_hybrid,
    load_cohort,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S07.csv"

CL_NAME = "Computational Linguistics (CL) Model"
LLM_NAME = "Llama 3.1 70B LLM"
DISPLAY_NAMES = {CL_NAME: "CL", LLM_NAME: "Llama 3.1 70B"}
HYBRID_NAME = "Hybrid Model (CL + Llama 3.1 70B)"

ALPHA = 0.05
EN_DASH = "–"

COLUMNS = [
    "Model",
    "Lung-RADS Assessment Category",
    "Samples (%)",
    "Precision, %\n(95% CI)",
    "Recall (Sensitivity), %\n(95% CI)",
    "Specificity, %\n(95% CI)",
    "F1-Score, %\n(95% CI)",
]
METRIC_COLUMNS = dict(zip(COLUMNS[3:], CLASS_METRICS))


def perfect_lower_bound(trials: int) -> float:
    """Exact one-sided 95% lower bound when every trial succeeded."""
    return ALPHA ** (1 / trials) * 100


def zero_upper_bound(trials: int) -> float:
    """Exact one-sided 95% upper bound when no trial succeeded."""
    return (1 - ALPHA ** (1 / trials)) * 100


def format_interval(point: float, low: float, high: float) -> str:
    return f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"


def format_bound(point: float, bound: float, upper: bool) -> str:
    return f"{point:.2f}\n({'<' if upper else '>'}{bound:.2f})"


def metric_cell(
    bootstrap: tuple[float, float, float], successes: int, trials: int
) -> str:
    """Percentile interval, or an exact one-sided bound at a boundary."""
    if trials == 0:
        return "–"
    if successes == trials:
        return format_bound(100.0, perfect_lower_bound(trials), upper=False)
    if successes == 0:
        return format_bound(0.0, zero_upper_bound(trials), upper=True)
    return format_interval(*bootstrap)


def category_rows(display_name, reference, prediction, classes, categories) -> list[dict[str, str]]:
    """One row per category: percentile intervals, or exact one-sided bounds at a boundary."""
    total = len(reference)
    rows = []
    for category in categories:
        is_reference = reference == category
        is_predicted = prediction == category
        tp = int((is_reference & is_predicted).sum())
        fp = int((~is_reference & is_predicted).sum())
        fn = int((is_reference & ~is_predicted).sum())
        tn = total - tp - fp - fn
        count = int(is_reference.sum())

        # F1 is perfect only when precision and recall both are, so it takes
        # the more conservative of their two bounds.
        trials = {
            "precision": (tp, tp + fp),
            "recall": (tp, tp + fn),
            "specificity": (tn, tn + fp),
            "f1": (tp, tp + max(fp, fn)),
        }
        row = {
            COLUMNS[0]: display_name,
            COLUMNS[1]: category,
            COLUMNS[2]: f"{count:,} ({count / total * 100:.2f}%)",
        }
        for column, metric in METRIC_COLUMNS.items():
            row[column] = metric_cell(classes[category][metric], *trials[metric])
        rows.append(row)
    return rows


def build_table(frame, seed: int, replicates: int) -> list[dict[str, str]]:
    per_model = evaluate(frame, seed, replicates, per_class=True)
    reference = normalise(frame[REFERENCE_COLUMN])
    categories = CATEGORY_ORDER + [MISSING_LABEL]

    rows = []
    for name in (CL_NAME, LLM_NAME):
        prediction = normalise(frame[MODEL_COLUMNS[name]])
        rows += category_rows(
            DISPLAY_NAMES[name], reference, prediction, per_model[name]["per_class"], categories
        )

    # The hybrid output is the CL output (equal to the LLM's) on the agreement subset.
    # When that subset is error-free, every hybrid cell is a boundary case and takes
    # the exact one-sided bound; otherwise non-boundary cells use the unconditional
    # bootstrap, which resamples the full cohort and re-selects the subset each time.
    cl = normalise(frame[MODEL_COLUMNS[CL_NAME]])
    llm = normalise(frame[MODEL_COLUMNS[LLM_NAME]])
    agreement = cl == llm
    agreed_reference = reference[agreement].reset_index(drop=True)
    agreed_output = cl[agreement].reset_index(drop=True)
    hybrid = evaluate_hybrid(frame, seed, replicates, per_class=True)
    rows += category_rows(HYBRID_NAME, agreed_reference, agreed_output, hybrid["per_class"], categories)
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

    bounded = sum(1 for r in rows for c in COLUMNS[3:] if ">" in r[c] or "<" in r[c])
    print(f"Cohort 1: {len(frame):,} reports, {len(rows)} rows")
    print(f"{bounded} of {len(rows) * 4} cells are boundary cases carrying a one-sided bound")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
