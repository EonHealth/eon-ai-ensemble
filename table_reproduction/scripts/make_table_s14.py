"""Build Supplementary Table S14: Hybrid Agreement and Concordance With Care-Navigator Annotations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from cohort2_metrics import CHUNK, DEFAULT_REPLICATES, DEFAULT_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "operational_10k_agreement.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S14.csv"

# Own stream; this table shares no interval with the others.
SEED_OFFSET = 14_000

# Table S14 orders its rows differently from the other Cohort 2 tables.
PUBLISHED_ROW_ORDER = ["Size", "Laterality", "Lobe", "Density", "Margin", "Shape", "Calcification"]

COUNT_COLUMNS = [
    "agreed_concordant",
    "agreed_discordant",
    "disagreed_concordant",
    "disagreed_discordant",
]

EN_DASH = "–"

COLUMNS = [
    "Characteristic",
    "Agreement Rate, %\n(95% CI)",
    "Concordance on Agreement Subset, %\n(95% CI)",
    "CL Concordance on Disagreement Subset, %\n(95% CI)",
]
METRICS = ("agreement_rate", "concordance_agreed", "concordance_disagreed")


def load_counts(path: Path) -> dict[str, tuple[int, int, int, int]]:
    frame = pd.read_csv(path)
    missing = {"characteristic", *COUNT_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {sorted(missing)}")

    counts = {}
    for row in frame.itertuples():
        cells = tuple(int(getattr(row, c)) for c in COUNT_COLUMNS)
        if any(c < 0 for c in cells):
            raise ValueError(f"{row.characteristic}: negative count")
        counts[row.characteristic] = cells

    absent = [c for c in PUBLISHED_ROW_ORDER if c not in counts]
    if absent:
        raise ValueError(f"Input is missing characteristics: {absent}")

    totals = {sum(counts[c]) for c in PUBLISHED_ROW_ORDER}
    if len(totals) != 1:
        raise AssertionError(f"Characteristics cover different cohort sizes: {sorted(totals)}")
    return counts


def reconstruct(cells: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A cohort of booleans matching the two-by-two counts."""
    agreed_concordant, agreed_discordant, disagreed_concordant, disagreed_discordant = cells
    agreed = np.repeat(
        [True, True, False, False],
        [agreed_concordant, agreed_discordant, disagreed_concordant, disagreed_discordant],
    )
    concordant = np.repeat(
        [True, False, True, False],
        [agreed_concordant, agreed_discordant, disagreed_concordant, disagreed_discordant],
    )
    return agreed, concordant


def rates(agreed: np.ndarray, concordant: np.ndarray) -> dict[str, np.ndarray]:
    """Agreement rate and concordance within each subset, over the trailing axis."""
    n = agreed.shape[-1]
    n_agreed = agreed.sum(axis=-1).astype(float)
    n_disagreed = n - n_agreed
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "agreement_rate": n_agreed / n * 100,
            "concordance_agreed": np.where(
                n_agreed > 0, (agreed & concordant).sum(axis=-1) / n_agreed, np.nan
            ) * 100,
            "concordance_disagreed": np.where(
                n_disagreed > 0, (~agreed & concordant).sum(axis=-1) / n_disagreed, np.nan
            ) * 100,
        }


def build_table(counts, seed: int, replicates: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    rows = []
    for characteristic in PUBLISHED_ROW_ORDER:
        agreed, concordant = reconstruct(counts[characteristic])
        n = len(agreed)
        observed = rates(agreed, concordant)

        samples = {name: np.empty(replicates) for name in METRICS}
        for start in range(0, replicates, CHUNK):
            count = min(CHUNK, replicates - start)
            indices = rng.integers(0, n, size=(count, n))
            batch = rates(agreed[indices], concordant[indices])
            for name in METRICS:
                samples[name][start : start + count] = batch[name]

        row = {COLUMNS[0]: characteristic}
        for column, name in zip(COLUMNS[1:], METRICS):
            low, high = np.nanpercentile(samples[name], [2.5, 97.5])
            row[column] = f"{float(observed[name]):.1f}\n({low:.1f}{EN_DASH}{high:.1f})"
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Agreement count table.")
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

    total = sum(counts[PUBLISHED_ROW_ORDER[0]])
    print(f"Operational cohort: {total:,} records, {len(rows)} characteristics")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
