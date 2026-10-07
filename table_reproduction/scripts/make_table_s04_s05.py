"""Build Supplementary Tables S4 and S5: Joint CL / LLM Correctness in Cohort 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cohort1_metrics import MODEL_COLUMNS, REFERENCE_COLUMN, load_cohort, normalise

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_1_lung_rads_predictions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_tables"

CL_NAME = "Computational Linguistics (CL) Model"
LLM_NAME = "Llama 3.1 70B LLM"

ROW_LABELS = ["CL Correct", "CL Incorrect", "LLM Total"]
VALUE_COLUMNS = ["LLM Correct", "LLM Incorrect", "CL Total"]


def cell(count: int, subset_size: int) -> str:
    share = count / subset_size * 100 if subset_size else 0.0
    return f"{count:,} ({share:.2f}%)"


def contingency(
    cl_correct: np.ndarray, llm_correct: np.ndarray, label: str
) -> list[dict[str, str]]:
    """2x2 joint correctness with totals, as printed."""
    n = len(cl_correct)
    both = int((cl_correct & llm_correct).sum())
    cl_only = int((cl_correct & ~llm_correct).sum())
    llm_only = int((~cl_correct & llm_correct).sum())
    neither = int((~cl_correct & ~llm_correct).sum())

    grid = [
        [both, cl_only, both + cl_only],
        [llm_only, neither, llm_only + neither],
        [both + llm_only, cl_only + neither, n],
    ]
    header = f"{label}\n(n = {n:,})"
    return [
        {header: row_label, **{col: cell(value, n) for col, value in zip(VALUE_COLUMNS, row)}}
        for row_label, row in zip(ROW_LABELS, grid)
    ]


def build_tables(frame) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    reference = normalise(frame[REFERENCE_COLUMN])
    cl = normalise(frame[MODEL_COLUMNS[CL_NAME]])
    llm = normalise(frame[MODEL_COLUMNS[LLM_NAME]])

    agreement = (cl == llm).to_numpy()
    cl_correct = (cl == reference).to_numpy()
    llm_correct = (llm == reference).to_numpy()

    # Agreeing models made the same prediction, so they are right or wrong together.
    if (cl_correct[agreement] != llm_correct[agreement]).any():
        raise AssertionError("Agreement subset has a report where only one model is correct")
    # Disagreeing models cannot both match the same reference value.
    if (cl_correct[~agreement] & llm_correct[~agreement]).any():
        raise AssertionError("Disagreement subset has a report where both models are correct")

    s04 = contingency(cl_correct[agreement], llm_correct[agreement], "Agreement Subset")
    s05 = contingency(cl_correct[~agreement], llm_correct[~agreement], "Disagreement Subset")

    summary = {
        "n_total": len(frame),
        "n_agreement": int(agreement.sum()),
        "n_disagreement": int((~agreement).sum()),
    }
    return s04, s05, summary


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 1 predictions CSV.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for both CSVs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_cohort(args.input)
    s04, s05, summary = build_tables(frame)

    write_csv(s04, args.output_dir / "Table_S04.csv")
    write_csv(s05, args.output_dir / "Table_S05.csv")

    total = summary["n_total"]
    print(f"Cohort 1: {total:,} reports")
    print(
        f"agreement {summary['n_agreement']:,} ({summary['n_agreement'] / total * 100:.2f}%), "
        f"disagreement {summary['n_disagreement']:,} ({summary['n_disagreement'] / total * 100:.2f}%)"
    )
    print(f"Wrote {args.output_dir / 'Table_S04.csv'}")
    print(f"Wrote {args.output_dir / 'Table_S05.csv'}")


if __name__ == "__main__":
    main()
