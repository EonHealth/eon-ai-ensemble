"""Build Table 2: Model Accuracy for Seven Incidental Pulmonary Nodule Characteristics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cohort2_metrics import (
    CHARACTERISTICS,
    DEFAULT_REPLICATES,
    DEFAULT_SEED,
    MODELS,
    evaluate,
    load_predictions,
    round_half_up,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_02.csv"

EN_DASH = "–"

COLUMNS = [
    "Pulmonary\nNodule\nCharacteristic",
    "CL Model\n(95% CI)",
    "Llama 3.1\n70B (95% CI)",
    "Llama 3.1 405B\n(95% CI)",
    "GPT-OSS-120B\n(95% CI)",
    "DeepSeek R1\n(95% CI)",
    "Hybrid model\nCL + GPT-OSS-120B\nPerformance",
    "Hybrid model\nCL + GPT-OSS-120B\nAgreement Subset",
]


def build_table(blocks, seed: int, replicates: int) -> tuple[list[dict[str, str]], list[dict]]:
    standalone, hybrid, _ = evaluate(blocks, seed, replicates)

    rows: list[dict[str, str]] = []
    summary: list[dict] = []
    for characteristic in CHARACTERISTICS:
        row = {COLUMNS[0]: characteristic}
        for column, model in zip(COLUMNS[1:6], MODELS):
            values = standalone[(characteristic, model)]["accuracy"]
            point, low, high = (round_half_up(round_half_up(v, 2), 1) for v in values)
            row[column] = f"{point:.1f}\n({low:.1f}{EN_DASH}{high:.1f})"

        point, low, high = hybrid[characteristic]["accuracy"]
        row[COLUMNS[6]] = f"{point:.2f}\n({low:.2f}{EN_DASH}{high:.2f})"
        row[COLUMNS[7]] = f"n = {hybrid[characteristic]['n']}"
        rows.append(row)
        summary.append(
            {"characteristic": characteristic, "accuracy": point, "n": hybrid[characteristic]["n"]}
        )

    return rows, summary


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
    rows, summary = build_table(blocks, args.seed, args.bootstrap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 2: {len(blocks[('Size', 'CL')]):,} reports")
    for s in summary:
        print(f"{s['characteristic']:15s} hybrid {s['accuracy']:6.2f}% on {s['n']:,} agreed reports")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
