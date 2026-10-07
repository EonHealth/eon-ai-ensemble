"""Build Supplementary Table S15: Hybrid Model Accuracy for Seven Characteristics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import fisher_exact

from cohort2_metrics import (
    CHARACTERISTICS,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    PAIRED_LLMS,
    evaluate,
    load_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S15.csv"

EN_DASH = "–"

COLUMNS = [
    "Hybrid Model",
    "Characteristic",
    "Agreement Subset\nSamples (n)",
    "Agreement Subset\nAgreement Rate, %\n(95% CI)",
    "Agreement Subset\nHybrid Model Accuracy, %\n(95% CI)",
    "Disagreement Subset\nCL Accuracy, %\n(95% CI)",
    "Disagreement Subset\nLLM Accuracy, %\n(95% CI)",
    "Statistical Tests\nMC p-val (CL)",
    "Statistical Tests\nMC p-val (LLM)",
    "Statistical Tests\nFE p-val (CL)",
    "Statistical Tests\nFE p-val (LLM)",
]


def format_ci(point: float, low: float, high: float) -> str:
    return f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    _, _, paired = evaluate(blocks, seed, replicates, pairings=True)

    rows = []
    for llm in PAIRED_LLMS:
        for characteristic in CHARACTERISTICS:
            r = paired[(llm, characteristic)]
            agreement = r["agreement_counts"]
            rows.append(
                {
                    COLUMNS[0]: f"CL + {llm}",
                    COLUMNS[1]: characteristic,
                    COLUMNS[2]: f"{r['n']:,}",
                    COLUMNS[3]: format_ci(*r["agreement_rate"]),
                    COLUMNS[4]: format_ci(*r["hybrid_accuracy"]),
                    COLUMNS[5]: format_ci(*r["cl_disagreement"]),
                    COLUMNS[6]: format_ci(*r["llm_disagreement"]),
                    COLUMNS[7]: format_p(r["mc_p_cl"]),
                    COLUMNS[8]: format_p(r["mc_p_llm"]),
                    COLUMNS[9]: format_p(
                        fisher_exact([agreement, r["cl_disagreement_counts"]])[1]
                    ),
                    COLUMNS[10]: format_p(
                        fisher_exact([agreement, r["llm_disagreement_counts"]])[1]
                    ),
                }
            )
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
