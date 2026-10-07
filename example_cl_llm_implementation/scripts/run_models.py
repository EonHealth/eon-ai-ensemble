"""Run the example CL model, the LLM, or both over the synthetic reports.

With both models, the output also records CL / LLM agreement, the hybrid
decision, and whether each model matches the reference label, mirroring the
Cohort 1 analysis file used for the manuscript tables.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parent
DEFAULT_INPUT = EXAMPLE_ROOT / "input_data" / "synthetic_lung_rads_reports.csv"
DEFAULT_OUTPUT = EXAMPLE_ROOT / "output_tables" / "predictions.csv"

REFERENCE_COLUMN = "max_rads_annotated"  # Label column in the synthetic reports CSV.
MODELS = ("cl", "llm")


def run_cl(reports: list[dict[str, str]]) -> list[str]:
    from cl_model import build_cl_pipeline, extract

    nlp = build_cl_pipeline()
    return [extract(nlp, report["text_original"])[0] for report in reports]


def run_llm(reports: list[dict[str, str]], model: str, workers: int) -> list[str]:
    # Imported here so the CL model can run without an OpenAI key.
    from dotenv import load_dotenv
    from openai import OpenAI

    from llm_model import extract

    load_dotenv(REPO_ROOT / ".env")  # Provides OPENAI_API_KEY.
    client = OpenAI()
    # Each report is sent once; results come back in input order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda report: extract(client, report["text_original"], model), reports))


def flag(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def build_rows(reports: list[dict[str, str]], predictions: dict[str, list[str]]) -> list[dict[str, str]]:
    rows = []
    for i, report in enumerate(reports):
        reference = report[REFERENCE_COLUMN]
        row = {"doc_id": report["doc_id"], "reference_lung_rads": reference}
        for name in predictions:
            row[f"{name}_lung_rads"] = predictions[name][i]
        if len(predictions) == 2:
            cl, llm = predictions["cl"][i], predictions["llm"][i]
            row["cl_llm_agree"] = flag(cl == llm)
        for name in predictions:
            row[f"{name}_correct"] = flag(predictions[name][i] == reference)
        if len(predictions) == 2:
            # A CL error the hybrid would send to human review.
            row["cl_error_flagged_by_disagreement"] = flag(cl != llm and cl != reference)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]], predictions: dict[str, list[str]], llm_model: str) -> None:
    n = len(rows)
    for name in predictions:
        label = f"LLM ({llm_model})" if name == "llm" else "CL"
        correct = sum(row[f"{name}_correct"] == "TRUE" for row in rows)
        print(f"{label} accuracy: {correct}/{n} ({correct / n:.2%})")
    if len(predictions) == 2:
        agree = [row for row in rows if row["cl_llm_agree"] == "TRUE"]
        hybrid_correct = sum(row["cl_correct"] == "TRUE" for row in agree)
        cl_errors = sum(row["cl_correct"] == "FALSE" for row in rows)
        flagged = sum(row["cl_error_flagged_by_disagreement"] == "TRUE" for row in rows)
        print(f"CL / LLM agreement: {len(agree)}/{n} ({len(agree) / n:.2%})")
        print(f"Hybrid accuracy on the agreement subset: {hybrid_correct}/{len(agree)}")
        print(f"CL errors flagged for review by disagreement: {flagged}/{cl_errors}")


def parse_args() -> argparse.Namespace:
    from llm_model import DEFAULT_MODEL

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS),
                        help="Which models to run (default: both).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Synthetic reports CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--llm-model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent LLM API requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        reports = list(csv.DictReader(handle))

    predictions = {}
    if "cl" in args.models:
        predictions["cl"] = run_cl(reports)
    if "llm" in args.models:
        predictions["llm"] = run_llm(reports, args.llm_model, args.workers)

    rows = build_rows(reports, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows, predictions, args.llm_model)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
