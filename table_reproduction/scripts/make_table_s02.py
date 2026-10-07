"""Build Supplementary Table S2: Distributions for the Seven Pulmonary Nodule Characteristics."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from cohort2_metrics import (
    CHARACTERISTICS,
    MISSING_TOKEN,
    NUMERIC_CHARACTERISTICS,
    PAIRED_LLMS,
    load_predictions,
    normalise,
    round_half_up,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "input_data" / "cohort_2_nodule_characteristic_predictions.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output_tables" / "Table_S02.csv"

MISSING_DISPLAY = "Missing"

# Size bins in centimetres, as [lower, upper). The last bin is open-ended.
SIZE_BINS = [("0-6 mm", 0.0, 0.6), ("6-8 mm", 0.6, 0.8), ("8-10 mm", 0.8, 1.0), ("10+ mm", 1.0, None)]

COLUMNS = ["Characteristic", "Value", "Samples (%)"]


def reference_values(blocks, characteristic: str) -> list:
    """Reference-standard values for one characteristic (shared by every model)."""
    return [normalise(v, characteristic) for v in blocks[(characteristic, "CL")]["ground_truth"]]


def size_bin(value: object) -> str:
    if value == MISSING_TOKEN:
        return MISSING_DISPLAY
    for label, lower, upper in SIZE_BINS:
        if value >= lower and (upper is None or value < upper):
            return label
    raise ValueError(f"Size {value!r} falls outside every bin")


def ordered_counts(values: list, characteristic: str) -> list[tuple[str, int]]:
    """Counts with "Missing" first, then bin order for Size or alphabetical."""
    if characteristic in NUMERIC_CHARACTERISTICS:
        counts = Counter(size_bin(v) for v in values)
        order = [MISSING_DISPLAY] + [label for label, _, _ in SIZE_BINS]
    else:
        counts = Counter(MISSING_DISPLAY if v == MISSING_TOKEN else str(v) for v in values)
        order = [MISSING_DISPLAY] + sorted(k for k in counts if k != MISSING_DISPLAY)
    return [(label, counts[label]) for label in order if counts[label]]


def build_table(blocks) -> list[dict[str, str]]:
    total = len(blocks[(CHARACTERISTICS[0], PAIRED_LLMS[0])])
    rows = []
    for characteristic in CHARACTERISTICS:
        counts = ordered_counts(reference_values(blocks, characteristic), characteristic)
        tallied = sum(count for _, count in counts)
        if tallied != total:
            raise AssertionError(f"{characteristic}: rows total {tallied}, expected {total}")
        for label, count in counts:
            share = round_half_up(count / total * 100, 2)
            rows.append(
                {
                    COLUMNS[0]: characteristic,
                    COLUMNS[1]: label,
                    COLUMNS[2]: f"{count:,} ({share:.2f}%)",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Cohort 2 predictions CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = load_predictions(args.input)
    rows = build_table(blocks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Cohort 2: {len(blocks[('Size', 'CL')]):,} reports, {len(rows)} rows")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
