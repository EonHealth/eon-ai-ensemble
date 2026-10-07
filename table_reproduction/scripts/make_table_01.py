"""Build Table 1: Standalone and Hybrid Model Performance for Lung-RADS Classification."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cohort1_metrics import (
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    MODEL_COLUMNS,
    REFERENCE_COLUMN,
    evaluate,
    evaluate_hybrid,
    load_cohort,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_01.csv"

# Row order as printed in the manuscript; the bootstrap draws CL first.
PUBLISHED_ROW_ORDER = ["Llama 3.1 70B LLM", "Computational Linguistics (CL) Model"]

EN_DASH = "\u2013"

HYBRID_METRICS = ("accuracy", "f1", "precision", "recall")


def format_ci(point: float, low: float, high: float) -> str:
    return f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"


def build_table(
    frame, seed: int, replicates: int
) -> tuple[list[dict[str, str]], dict[str, object]]:
    per_model = evaluate(frame, seed, replicates)
    reference = normalise(frame[REFERENCE_COLUMN])
    n_total = len(frame)

    cl = normalise(frame[MODEL_COLUMNS["Computational Linguistics (CL) Model"]])
    llm = normalise(frame[MODEL_COLUMNS["Llama 3.1 70B LLM"]])
    agreement = cl == llm
    n_agreement = int(agreement.sum())

    hybrid_correct = int((cl[agreement] == reference[agreement]).sum())
    if hybrid_correct == n_agreement:
        # An error-free subset makes the percentile bootstrap degenerate, so every
        # metric is reported with the exact one-sided 95% lower bound instead.
        hybrid_bound = 0.05 ** (1 / n_agreement) * 100
        hybrid_cells = dict.fromkeys(HYBRID_METRICS, f"100\n(>{hybrid_bound:.2f})")
    else:
        # Otherwise use the unconditional bootstrap: resample the full cohort and
        # re-select the agreement subset in every replicate.
        hybrid = evaluate_hybrid(frame, seed, replicates)
        hybrid_cells = {metric: format_ci(*hybrid[metric]) for metric in HYBRID_METRICS}

    rows = []
    for name in PUBLISHED_ROW_ORDER:
        metrics = per_model[name]
        rows.append(
            {
                "Model Name": name,
                "Evaluation Set\n(sample size)": f"Full cohort\n(n = {n_total:,})",
                "Accuracy\n(95% CI)": format_ci(*metrics["accuracy"]),
                "Macro F1\n(95% CI)": format_ci(*metrics["f1"]),
                "Macro Precision\n(95% CI)": format_ci(*metrics["precision"]),
                "Macro Recall\n(95% CI)": format_ci(*metrics["recall"]),
            }
        )
    rows.append(
        {
            "Model Name": "Hybrid Model\n(CL + LLM)",
            "Evaluation Set\n(sample size)": f"Agreement\nsubset\n(n = {n_agreement:,})",
            "Accuracy\n(95% CI)": hybrid_cells["accuracy"],
            "Macro F1\n(95% CI)": hybrid_cells["f1"],
            "Macro Precision\n(95% CI)": hybrid_cells["precision"],
            "Macro Recall\n(95% CI)": hybrid_cells["recall"],
        }
    )

    summary = {
        "n_total": n_total,
        "n_agreement": n_agreement,
        "agreement_rate": n_agreement / n_total * 100,
        "per_model": per_model,
    }
    return rows, summary


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
    rows, summary = build_table(frame, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 1: {summary['n_total']:,} reports")
    print(
        f"CL / LLM agreement: {summary['n_agreement']:,} reports "
        f"({summary['agreement_rate']:.2f}%)"
    )
    for name, metrics in summary["per_model"].items():
        accuracy = metrics["accuracy"]
        print(f"{name}: accuracy {accuracy[0]:.2f}% ({accuracy[1]:.2f}-{accuracy[2]:.2f})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
