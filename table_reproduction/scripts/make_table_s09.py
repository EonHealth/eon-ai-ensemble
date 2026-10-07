"""Build Supplementary Table S9: Paired CL vs. LLM Accuracy Comparisons in Cohort 2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from cohort2_metrics import (
    CHARACTERISTICS,
    CHUNK,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    PAIRED_LLMS,
    correctness,
    load_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S09.csv"

# Own stream; this table shares no interval with the others.
SEED_OFFSET = 9_000

EN_DASH = "–"

COLUMNS = [
    "Characteristic",
    "LLM",
    "CL Accuracy, %",
    "LLM Accuracy, %",
    "CL Correct /\nLLM Incorrect",
    "CL Incorrect /\nLLM Correct",
    "Accuracy Difference, CL - LLM,\npercentage points (95% CI)",
    "Exact Two-Sided\nMcNemar p",
]


def paired_difference_ci(
    difference: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float]:
    """95% percentile bootstrap for a paired per-report difference."""
    n = len(difference)
    samples = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        samples[start : start + count] = difference[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low) * 100, float(high) * 100


def mcnemar_p(cl_only: int, llm_only: int) -> float:
    """Exact two-sided McNemar test on the discordant pair."""
    discordant = cl_only + llm_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(cl_only, llm_only), discordant, 0.5).pvalue)


def format_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    rows = []
    for characteristic in CHARACTERISTICS:
        cl = correctness(blocks[(characteristic, "CL")])
        for llm in PAIRED_LLMS:
            llm_correct = correctness(blocks[(characteristic, llm)])
            cl_only = int((cl & ~llm_correct).sum())
            llm_only = int((~cl & llm_correct).sum())

            difference = cl.astype(int) - llm_correct.astype(int)
            low, high = paired_difference_ci(difference, rng, replicates)

            rows.append(
                {
                    COLUMNS[0]: characteristic,
                    COLUMNS[1]: llm,
                    COLUMNS[2]: f"{cl.mean() * 100:.2f}",
                    COLUMNS[3]: f"{llm_correct.mean() * 100:.2f}",
                    COLUMNS[4]: f"{cl_only:,}",
                    COLUMNS[5]: f"{llm_only:,}",
                    COLUMNS[6]: (
                        f"{difference.mean() * 100:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"
                    ),
                    COLUMNS[7]: format_p(mcnemar_p(cl_only, llm_only)),
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

    significant = sum(1 for r in rows if r[COLUMNS[7]] == "<0.001" or float(r[COLUMNS[7]]) < 0.05)
    print(f"Cohort 2: {len(blocks[('Size', 'CL')]):,} reports, {len(rows)} rows")
    print(f"CL significantly more accurate in {significant} of {len(rows)} comparisons (p < 0.05)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
