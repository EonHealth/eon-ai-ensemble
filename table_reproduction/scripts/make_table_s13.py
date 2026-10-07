"""Build Supplementary Table S13: Model Concordance With Care-Navigator Annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from cohort2_metrics import CHARACTERISTICS, CHUNK, DEFAULT_REPLICATES, DEFAULT_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "operational_10k_concordance.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S13.csv"

# Own stream; this table shares no interval with the others.
SEED_OFFSET = 13_000

MODELS = ["CL", "GPT-OSS-120B"]
EN_DASH = "–"

COLUMNS = [
    "Characteristic",
    "CL Concordance, %\n(95% CI)",
    "GPT-OSS-120B Concordance, %\n(95% CI)",
]


def load_counts(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    frame = pd.read_csv(path)
    required = {"characteristic", "model", "concordant", "total"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {sorted(missing)}")

    counts = {}
    for row in frame.itertuples():
        concordant, total = int(row.concordant), int(row.total)
        if not 0 <= concordant <= total:
            raise ValueError(f"{row.characteristic}/{row.model}: {concordant} of {total}")
        counts[(row.characteristic, row.model)] = (concordant, total)

    expected = {(c, m) for c in CHARACTERISTICS for m in MODELS}
    if not expected.issubset(counts):
        raise ValueError(f"Input is missing rows: {sorted(expected - set(counts))}")
    return counts


def bootstrap_proportion(
    concordant: int, total: int, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    """Proportion and its 95% percentile bootstrap bounds, as percentages."""
    observations = np.zeros(total, dtype=bool)
    observations[:concordant] = True
    samples = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, total, size=(count, total))
        samples[start : start + count] = observations[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return concordant / total * 100, float(low) * 100, float(high) * 100


def build_table(counts, seed: int, replicates: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    rows = []
    for characteristic in CHARACTERISTICS:
        row = {COLUMNS[0]: characteristic}
        for column, model in zip(COLUMNS[1:], MODELS):
            point, low, high = bootstrap_proportion(
                *counts[(characteristic, model)], rng, replicates
            )
            row[column] = f"{point:.1f}\n({low:.1f}{EN_DASH}{high:.1f})"
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Concordance count table.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, help="Bootstrap replicates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = load_counts(args.input)
    rows = build_table(counts, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total = counts[(CHARACTERISTICS[0], "CL")][1]
    print(f"Operational cohort: {total:,} records, {len(rows)} characteristics")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
