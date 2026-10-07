"""Build Supplementary Table S11: Report-Level Error Overlap Between CL and LLMs in Cohort 2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cohort2_metrics import (
    CHUNK,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    PAIRED_LLMS,
    load_predictions,
    report_level_correct,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S11.csv"

# Own stream; this table shares no interval with the others.
SEED_OFFSET = 11_000

EN_DASH = "–"

COLUMNS = [
    "Comparison Pair",
    "Both Correct, n\n(% of cohort)",
    "CL Correct / LLM Incorrect, n\n(% of cohort)",
    "CL Incorrect / LLM Correct, n\n(% of cohort)",
    "Both Incorrect, n\n(% of cohort)",
    "Phi Error Correlation\n(95% CI)",
]


def phi(cl_error: np.ndarray, llm_error: np.ndarray) -> np.ndarray:
    """Phi coefficient between two boolean arrays, over the trailing axis."""
    both = (cl_error & llm_error).sum(axis=-1).astype(float)
    cl_only = (cl_error & ~llm_error).sum(axis=-1).astype(float)
    llm_only = (~cl_error & llm_error).sum(axis=-1).astype(float)
    neither = (~cl_error & ~llm_error).sum(axis=-1).astype(float)
    denominator = np.sqrt(
        (both + cl_only) * (llm_only + neither) * (both + llm_only) * (cl_only + neither)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (both * neither - cl_only * llm_only) / denominator, np.nan)


def phi_ci(
    cl_error: np.ndarray, llm_error: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float]:
    n = len(cl_error)
    samples = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        samples[start : start + count] = phi(cl_error[indices], llm_error[indices])
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return float(low), float(high)


def build_table(blocks, seed: int, replicates: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed + SEED_OFFSET)
    cl = report_level_correct(blocks, "CL")
    total = len(cl)

    def cell(count: int) -> str:
        return f"{count:,} ({count / total * 100:.2f}%)"

    rows = []
    for llm in PAIRED_LLMS:
        llm_correct = report_level_correct(blocks, llm)
        both = int((cl & llm_correct).sum())
        cl_only = int((cl & ~llm_correct).sum())
        llm_only = int((~cl & llm_correct).sum())
        neither = int((~cl & ~llm_correct).sum())
        if both + cl_only + llm_only + neither != total:
            raise AssertionError("Joint correctness counts do not partition the cohort")

        point = float(phi(~cl, ~llm_correct))
        low, high = phi_ci(~cl, ~llm_correct, rng, replicates)
        rows.append(
            {
                COLUMNS[0]: f"CL + {llm}",
                COLUMNS[1]: cell(both),
                COLUMNS[2]: cell(cl_only),
                COLUMNS[3]: cell(llm_only),
                COLUMNS[4]: cell(neither),
                COLUMNS[5]: f"{point:.3f}\n({low:.3f}{EN_DASH}{high:.3f})",
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

    print(f"Cohort 2: {len(rows)} pairings")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
