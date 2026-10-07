"""Build Supplementary Table S12: Selective Risk and Coverage Across Agreement Thresholds."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cohort2_metrics import (
    CHARACTERISTICS,
    CHUNK,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    PAIRED_LLMS,
    load_predictions,
    normalise,
    report_level_correct,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S12.csv"

# Own stream; this table shares no figure with the others.
SEED_OFFSET = 12_000

EN_DASH = "–"
ALPHA = 0.05
THRESHOLDS = range(len(CHARACTERISTICS) + 1)

COLUMNS = [
    "Comparison Pair",
    "Minimum Agreeing Characteristics",
    "Coverage, %\n(95% CI)",
    "Selective Risk, %\n(95% CI)",
]


def report_level_error(blocks) -> np.ndarray:
    """True where CL got at least one of the seven characteristics wrong."""
    return ~report_level_correct(blocks, "CL")


def agreement_score(blocks, llm: str) -> np.ndarray:
    """How many of the seven characteristics CL and ``llm`` agreed on, per report."""
    score = np.zeros(len(blocks[(CHARACTERISTICS[0], "CL")]), dtype=int)
    for characteristic in CHARACTERISTICS:
        score += np.array(
            [
                normalise(a, characteristic) == normalise(b, characteristic)
                for a, b in zip(
                    blocks[(characteristic, "CL")]["prediction"],
                    blocks[(characteristic, llm)]["prediction"],
                )
            ]
        )
    return score


def bootstrap_row(
    covered: np.ndarray, error: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Coverage and selective risk with 95% percentile bootstrap bounds."""
    n = len(covered)
    coverage = np.empty(replicates)
    risk = np.empty(replicates)

    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        mask = covered[indices]
        kept = mask.sum(axis=1)
        coverage[start : start + count] = kept / n
        with np.errstate(invalid="ignore", divide="ignore"):
            risk[start : start + count] = np.where(
                kept > 0, (error[indices] & mask).sum(axis=1) / kept, np.nan
            )

    low, high = np.nanpercentile(coverage, [2.5, 97.5])
    coverage_ci = (covered.mean() * 100, float(low) * 100, float(high) * 100)
    low, high = np.nanpercentile(risk, [2.5, 97.5])
    risk_ci = (error[covered].mean() * 100, float(low) * 100, float(high) * 100)
    return coverage_ci, risk_ci


def format_ci(point: float, low: float, high: float) -> str:
    return f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"


def metric_cell(bootstrap: tuple[float, float, float], successes: int, trials: int) -> str:
    """Percentile interval, or an exact one-sided Clopper-Pearson bound at a boundary."""
    if trials == 0:
        return EN_DASH
    if successes == trials:
        return f"100.00\n(>{ALPHA ** (1 / trials) * 100:.2f})"
    if successes == 0:
        return f"0.00\n(<{(1 - ALPHA ** (1 / trials)) * 100:.2f})"
    return format_ci(*bootstrap)


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    error = report_level_error(blocks)
    scores = {llm: agreement_score(blocks, llm) for llm in PAIRED_LLMS}

    cache: dict[bytes, tuple] = {}
    rows = []
    for llm in PAIRED_LLMS:
        for threshold in THRESHOLDS:
            covered = scores[llm] >= threshold
            key = covered.tobytes()
            if key not in cache:
                cache[key] = bootstrap_row(covered, error, rng, replicates)
            coverage_ci, risk_ci = cache[key]
            n_covered = int(covered.sum())
            # At 0/7 every report is retained by construction, so there is no
            # sampling uncertainty to report.
            coverage = (
                f"100.00\n({EN_DASH})"
                if threshold == 0
                else metric_cell(coverage_ci, n_covered, len(covered))
            )
            rows.append(
                {
                    COLUMNS[0]: f"CL + {llm}",
                    COLUMNS[1]: f"{threshold}/{len(CHARACTERISTICS)}",
                    COLUMNS[2]: coverage,
                    COLUMNS[3]: metric_cell(risk_ci, int(error[covered].sum()), n_covered),
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
