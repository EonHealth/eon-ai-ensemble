"""Build Supplementary Table S16: CL + GPT-OSS-120B vs. CL and all LLMs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from cohort2_metrics import (
    CHARACTERISTICS,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    HYBRID_PAIR,
    MODELS,
    agreement_mask,
    correctness,
    evaluate,
    load_predictions,
    randomisation_p_value,
    round_half_up,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S16.csv"

# Separate stream for the comparators S15 does not already cover.
TEST_SEED_OFFSET = 16_000

EN_DASH = "–"
HYBRID_NAME = f"{HYBRID_PAIR[0]} + {HYBRID_PAIR[1]}"

COLUMNS = [
    "Comparator Model",
    "Characteristic",
    f"{HYBRID_NAME}\nAgreement Rate, %\n(95% CI)",
    f"{HYBRID_NAME}\nAgreement Samples (n)",
    f"{HYBRID_NAME}\nAccuracy on Agreement Subset, %\n(95% CI)",
    "Comparator Model\nAccuracy on Full Cohort, %\n(95% CI)",
    "Monte-Carlo p-value",
    "Fisher's Exact p-value",
]


def format_ci(point: float, low: float, high: float, digits: int = 2) -> str:
    return f"{point:.{digits}f}\n({low:.{digits}f}{EN_DASH}{high:.{digits}f})"


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    standalone, hybrid, paired = evaluate(blocks, seed, replicates, pairings=True)
    rng = np.random.default_rng(seed + TEST_SEED_OFFSET)

    rows = []
    for comparator in MODELS:
        for characteristic in CHARACTERISTICS:
            pair = paired[(HYBRID_PAIR[1], characteristic)]
            agreed_correct, agreed_wrong = pair["agreement_counts"]
            subset_size = pair["n"]
            observed = agreed_correct / subset_size

            full = correctness(blocks[(characteristic, comparator)])

            # Table S15 already reports these two against this hybrid; reuse them
            # so the two tables cannot disagree.
            if comparator == HYBRID_PAIR[0]:
                monte_carlo = pair["mc_p_cl"]
            elif comparator == HYBRID_PAIR[1]:
                monte_carlo = pair["mc_p_llm"]
            else:
                monte_carlo = randomisation_p_value(
                    full, observed, subset_size, rng, replicates
                )

            # Agreement subset vs. the comparator on the disagreement subset, as in Table S15.
            disagreed = full[~agreement_mask(blocks, characteristic, HYBRID_PAIR[1])]
            disagreed_correct = int(disagreed.sum())
            fisher = fisher_exact(
                [[agreed_correct, agreed_wrong], [disagreed_correct, len(disagreed) - disagreed_correct]]
            )[1]

            # Shown to one decimal, matching Table 2's standalone columns.
            comparator_accuracy = (
                round_half_up(round_half_up(v, 2), 1)
                for v in standalone[(characteristic, comparator)]["accuracy"]
            )

            rows.append(
                {
                    COLUMNS[0]: comparator,
                    COLUMNS[1]: characteristic,
                    COLUMNS[2]: format_ci(*pair["agreement_rate"]),
                    COLUMNS[3]: f"{subset_size:,}",
                    COLUMNS[4]: format_ci(*hybrid[characteristic]["accuracy"]),
                    COLUMNS[5]: format_ci(*comparator_accuracy, digits=1),
                    COLUMNS[6]: format_p(monte_carlo),
                    COLUMNS[7]: format_p(fisher),
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
