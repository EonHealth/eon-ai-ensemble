"""Build Supplementary Table S10: Inter-Model Disagreement as a Screen for CL Error."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cohort1_metrics import MODEL_COLUMNS, REFERENCE_COLUMN, load_cohort
from cohort1_metrics import normalise as normalise_cohort1
from cohort2_metrics import (
    CHARACTERISTICS,
    CHUNK,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    PAIRED_LLMS,
    correctness,
    load_predictions,
    normalise,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT1 = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_COHORT2 = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S10.csv"

# Own stream; this table shares no interval with the others.
SEED_OFFSET = 10_000

ALPHA = 0.05
EN_DASH = "–"

COLUMNS = [
    "Cohort",
    "Comparison Pair",
    "Sensitivity (Recall), %\n(95% CI)",
    "Specificity, %\n(95% CI)",
    "Precision (PPV), %\n(95% CI)",
    "NPV, %\n(95% CI)",
    "F1-Score, %\n(95% CI)",
]
METRICS = ("sensitivity", "specificity", "ppv", "npv", "f1")


def screen_metrics(screen: np.ndarray, wrong: np.ndarray) -> dict[str, np.ndarray]:
    """Screening metrics from stacked (replicates, n) boolean arrays."""
    tp = (screen & wrong).sum(axis=-1).astype(float)
    fp = (screen & ~wrong).sum(axis=-1).astype(float)
    fn = (~screen & wrong).sum(axis=-1).astype(float)
    tn = (~screen & ~wrong).sum(axis=-1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "sensitivity": np.where(tp + fn > 0, tp / (tp + fn), np.nan) * 100,
            "specificity": np.where(tn + fp > 0, tn / (tn + fp), np.nan) * 100,
            "ppv": np.where(tp + fp > 0, tp / (tp + fp), np.nan) * 100,
            "npv": np.where(tn + fn > 0, tn / (tn + fn), np.nan) * 100,
            "f1": np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan) * 100,
        }


def perfect_lower_bound(trials: int) -> float:
    return ALPHA ** (1 / trials) * 100


def zero_upper_bound(trials: int) -> float:
    return (1 - ALPHA ** (1 / trials)) * 100


def metric_cell(point: float, low: float, high: float, successes: int, trials: int) -> str:
    """Percentile interval, or an exact one-sided bound at a boundary."""
    if trials == 0:
        return EN_DASH
    if successes == trials:
        return f"100.00\n(>{perfect_lower_bound(trials):.2f})"
    if successes == 0:
        return f"0.00\n(<{zero_upper_bound(trials):.2f})"
    return f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"


def evaluate_screen(
    screen: np.ndarray, wrong: np.ndarray, rng: np.random.Generator, replicates: int
) -> dict[str, str]:
    n = len(screen)
    observed = screen_metrics(screen, wrong)
    tp = int((screen & wrong).sum())
    fp = int((screen & ~wrong).sum())
    fn = int((~screen & wrong).sum())
    tn = int((~screen & ~wrong).sum())
    # F1 is perfect only when the screen makes neither kind of error.
    trials = {
        "sensitivity": (tp, tp + fn),
        "specificity": (tn, tn + fp),
        "ppv": (tp, tp + fp),
        "npv": (tn, tn + fn),
        "f1": (tp, tp + max(fp, fn)),
    }

    samples = {name: np.empty(replicates) for name in METRICS}
    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        batch = screen_metrics(screen[indices], wrong[indices])
        for name in METRICS:
            samples[name][start : start + count] = batch[name]

    cells = {}
    for column, name in zip(COLUMNS[2:], METRICS):
        low, high = np.nanpercentile(samples[name], [2.5, 97.5])
        cells[column] = metric_cell(
            float(observed[name]), float(low), float(high), *trials[name]
        )
    return cells


def cohort1_screen(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = load_cohort(path)
    reference = normalise_cohort1(frame[REFERENCE_COLUMN])
    cl = normalise_cohort1(frame[MODEL_COLUMNS["Computational Linguistics (CL) Model"]])
    llm = normalise_cohort1(frame[MODEL_COLUMNS["Llama 3.1 70B LLM"]])
    return (cl != llm).to_numpy(), (cl != reference).to_numpy()


def cohort2_screen(blocks, llm: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(blocks[(CHARACTERISTICS[0], "CL")])
    disagreed = np.zeros(n, dtype=bool)
    wrong = np.zeros(n, dtype=bool)
    for characteristic in CHARACTERISTICS:
        cl_block = blocks[(characteristic, "CL")]
        disagreed |= np.array(
            [
                normalise(a, characteristic) != normalise(b, characteristic)
                for a, b in zip(cl_block["prediction"], blocks[(characteristic, llm)]["prediction"])
            ]
        )
        wrong |= ~correctness(cl_block)
    return disagreed, wrong


def build_table(cohort1_path: Path, cohort2_path: Path, seed: int, replicates: int):
    rng = np.random.default_rng(seed + SEED_OFFSET)
    rows = []

    screen, wrong = cohort1_screen(cohort1_path)
    rows.append(
        {
            COLUMNS[0]: "Cohort 1",
            COLUMNS[1]: "CL + Llama 3.1 70B",
            **evaluate_screen(screen, wrong, rng, replicates),
        }
    )

    blocks = load_predictions(cohort2_path)
    for llm in PAIRED_LLMS:
        screen, wrong = cohort2_screen(blocks, llm)
        rows.append(
            {
                COLUMNS[0]: "Cohort 2",
                COLUMNS[1]: f"CL + {llm}",
                **evaluate_screen(screen, wrong, rng, replicates),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort1", type=Path, default=DEFAULT_COHORT1, help="Cohort 1 predictions CSV.")
    parser.add_argument("--cohort2", type=Path, default=DEFAULT_COHORT2, help="Cohort 2 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Bootstrap seed.")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_REPLICATES, help="Bootstrap replicates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_table(args.cohort1, args.cohort2, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    bounded = sum(1 for r in rows for c in COLUMNS[2:] if ">" in r[c] or "<" in r[c])
    print(f"{len(rows)} rows; {bounded} of {len(rows) * 5} cells are boundary cases")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
