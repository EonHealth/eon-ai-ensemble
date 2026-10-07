# Hybrid Computational Linguistics and Large Language Model for Reliable Extraction of Structured Data from Radiology Reports — Code Repository

Companion code for the manuscript. The repository has two parts:

| Folder | Contents |
| --- | --- |
| `table_reproduction/` | Scripts that regenerate the tables in the manuscript and supplement from the study's prediction data. |
| `example_cl_llm_implementation/` | An illustrative computational linguistics (CL) model, LLM, and hybrid CL + LLM pipeline for Lung-RADS extraction, run on synthetic reports. |

Both folders follow the same layout: `input_data/`, `scripts/`, and `output_tables/`.

## Data availability

The data used in this study (`table_reproduction/input_data/`) are not included in this repository because they are
proprietary and cannot be publicly shared. Researchers seeking access may contact
Eon <[success@eonhealth.com](mailto:success@eonhealth.com)>. Requests are subject to Eon's review
and approval. The analysis scripts are provided for reproducibility but require
access to the underlying data to reproduce the study results.

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                      # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"    # Windows
```

Nothing else to install — `uv run` fetches the pinned Python and dependencies on
first use.

## Table reproduction

Each script in `table_reproduction/scripts/` reads from `table_reproduction/input_data/`
and writes one CSV to `table_reproduction/output_tables/`. Run from the repository root:

```bash
uv run table_reproduction/scripts/make_table_01.py          # one table
for f in table_reproduction/scripts/make_table_*.py; do uv run "$f"; done    # all tables
```

### Scripts and the tables they produce

Paths are relative to `table_reproduction/`.

| Script | Table | Output |
| --- | --- | --- |
| `make_table_01.py` | Table 1 — Standalone and Hybrid Model Performance for Lung-RADS Classification | `output_tables/Table_01.csv` |
| `make_table_02.py` | Table 2 — Model Accuracy for Seven Incidental Pulmonary Nodule Characteristics | `output_tables/Table_02.csv` |
| `make_table_s01.py` | Table S1 — Lung-RADS Assessment Category Distribution in Cohort 1 | `output_tables/Table_S01.csv` |
| `make_table_s02.py` | Table S2 — Distributions for the Seven Pulmonary Nodule Characteristics | `output_tables/Table_S02.csv` |
| `make_table_s03.py` | Table S3 — Additional Model Performance Metrics for Lung-RADS Classification | `output_tables/Table_S03.csv` |
| `make_table_s04_s05.py` | Table S4 — Model Accuracy Within the Agreement Subset (Cohort 1) | `output_tables/Table_S04.csv` |
| `make_table_s04_s05.py` | Table S5 — Model Accuracy Within the Disagreement Subset (Cohort 1) | `output_tables/Table_S05.csv` |
| `make_table_s06.py` | Table S6 — Confusion Matrix for CL Lung-RADS Classification (Cohort 1) | `output_tables/Table_S06.csv` |
| `make_table_s07.py` | Table S7 — Category-Specific Model Performance for Lung-RADS Classification | `output_tables/Table_S07.csv` |
| `make_table_s08.py` | Table S8 — Additional Characteristic-Level Performance Results | `output_tables/Table_S08.csv` |
| `make_table_s09.py` | Table S9 — Paired CL vs. LLM Accuracy Comparisons (Cohort 2) | `output_tables/Table_S09.csv` |
| `make_table_s10.py` | Table S10 — Inter-Model Disagreement as a Screen for CL Error | `output_tables/Table_S10.csv` |
| `make_table_s11.py` | Table S11 — Report-Level Error Overlap Between CL and LLMs (Cohort 2) | `output_tables/Table_S11.csv` |
| `make_table_s12.py` | Table S12 — Selective Risk and Coverage Across Agreement Thresholds | `output_tables/Table_S12.csv` |
| `make_table_s13.py` | Table S13 — Model Concordance With Care-Navigator Annotations | `output_tables/Table_S13.csv` |
| `make_table_s14.py` | Table S14 — Hybrid Agreement and Concordance With Care-Navigator Annotations | `output_tables/Table_S14.csv` |
| `make_table_s15.py` | Table S15 — Hybrid Model Accuracy for Seven Characteristics | `output_tables/Table_S15.csv` |
| `make_table_s16.py` | Table S16 — CL + GPT-OSS-120B vs. CL and all LLMs | `output_tables/Table_S16.csv` |

## Example CL + LLM implementation

The production CL model and LLM prompts are proprietary, so `example_cl_llm_implementation/`
contains a small, illustrative stand-in for one task: extracting the report-level Lung-RADS
category (Cohort 1). It runs on 100 synthetic reports
(`example_cl_llm_implementation/input_data/synthetic_lung_rads_reports.csv`) that were written
from templates and contain no real patient data. Eight of them are deliberately hard for rules
(a spelled-out category, a category without the Lung-RADS keyword, a reference table, prior-exam
categories, an addendum that revises the category, and diagnostic CTs that mention an earlier
screening category), so the hybrid has disagreements to flag.

The LLM calls the OpenAI API. Copy `.env.example` to `.env` in the repository root and replace
the placeholder with your own key (`.env` is gitignored):

```bash
cp .env.example .env    # then edit .env: OPENAI_API_KEY=sk-...
```

Run from the repository root:

```bash
uv run example_cl_llm_implementation/scripts/run_models.py               # both models (hybrid)
uv run example_cl_llm_implementation/scripts/run_models.py --models cl   # CL only (no API key needed)
uv run example_cl_llm_implementation/scripts/run_models.py --models llm  # LLM only
uv run example_cl_llm_implementation/scripts/make_synthetic_reports.py   # regenerate the synthetic reports
```

`run_models.py` writes `example_cl_llm_implementation/output_tables/predictions.csv`, prints each
model's accuracy and, when both run, the hybrid results. `--llm-model` selects a different OpenAI
model (default `gpt-4o-mini`). With both models the output has one row per report:

| Column | Meaning |
| --- | --- |
| `doc_id` | Report identifier. |
| `reference_lung_rads` | Reference Lung-RADS category (`N/A` when the report states none). |
| `cl_lung_rads`, `llm_lung_rads` | Category extracted by each model. |
| `cl_llm_agree` | The models agree; under the hybrid workflow the report is accepted without review. |
| `cl_correct`, `llm_correct` | The model's output matches the reference. |
| `cl_error_flagged_by_disagreement` | The CL is wrong and the models disagree, so the error would be caught by human review. |

When only one model runs, only that model's columns are written.

### CL model (`scripts/cl_model.py`)

A spaCy pipeline with a handful of deterministic rules, organized in the four tiers described
in Supplementary Section S3:

1. **Mark**: token patterns tag Lung-RADS mentions (e.g. "Lung-RADS category: 4X") and section
   headers, each with a canonical concept identifier; the longest overlapping match wins.
2. **Refine**: mentions inside category legends (boilerplate) are discarded.
3. **Attach context**: each mention is assigned its report section and flagged if it refers to
   a prior exam.
4. **Select**: current mentions in the impression are preferred, falling back to any current
   mention; ties go to the most severe category, and reports with no mention return `N/A`.

This example illustrates the approach only. It is not the production model evaluated in the
manuscript, which has more than 5,000 rules.

### LLM (`scripts/llm_model.py`)

A short zero-shot prompt that loosely follows Supplementary Section S4: the complete report is
sent with instructions to classify it into a Lung-RADS category, considering all sections
including any addendum, and the answer is constrained by a JSON schema to the predefined
categories (0, 1, 2, 3, 4, 4A, 4B, 4X, N/A). It uses OpenAI's Responses API with
`gpt-4o-mini` at temperature 0, and each report is sent once. This prompt and model are
illustrative; they are not the prompt or models evaluated in the manuscript.

The study ran its LLMs (Llama 3.1, GPT-OSS-120B, DeepSeek R1) through AWS Bedrock. This example
uses the OpenAI API instead because it needs only an API key, which makes it easier to set up.

On the synthetic reports the CL model scores 92/100 and the LLM about 91/100, but their errors
differ: the CL misses phrasings outside its rules, while the LLM tends to assign a category to
reports that do not state one. Where the two agree (83 reports), every output is correct, and
all 8 CL errors fall in the disagreement subset that would go to human review.
