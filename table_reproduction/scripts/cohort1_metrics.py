"""Shared Cohort 1 loading, metric and bootstrap helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_COLUMN = "max_rads_annotated"
# Bootstrap draw order: CL first. Table 1 prints the LLM first.
MODEL_COLUMNS = {
    "Computational Linguistics (CL) Model": "PRED_max_rads_CL",
    "Llama 3.1 70B LLM": "max_rads_llm",
}

# Blank cells in the source spreadsheet mean "no Lung-RADS category was stated",
# which is a scored category in its own right rather than missing data.
MISSING_LABEL = "Missing"

# Lung-RADS categories in clinical order; MISSING_LABEL is appended last.
CATEGORY_ORDER = ["0", "1", "2", "3", "4", "4A", "4B", "4X"]

DEFAULT_SEED = 20260817
# The hybrid bootstrap draws from its own stream so the standalone draws are unchanged.
HYBRID_SEED_OFFSET = 1
DEFAULT_REPLICATES = 10_000
CHUNK = 500

METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "kappa")
# Metrics that also have a per-category value, for Table S7.
CLASS_METRICS = ("precision", "recall", "specificity", "f1")


def load_cohort(path: Path) -> pd.DataFrame:
    """Read the predictions file with every category kept as a literal string."""
    frame = pd.read_csv(path, dtype=str)
    required = [REFERENCE_COLUMN, *MODEL_COLUMNS.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")
    return frame


def normalise(series: pd.Series) -> pd.Series:
    """Fold blanks into the explicit Missing category and trim stray whitespace."""
    return series.fillna(MISSING_LABEL).str.strip().replace("", MISSING_LABEL)


def macro_metrics(counts: np.ndarray, n_reference: int) -> dict[str, np.ndarray]:
    """Accuracy, macro precision / recall / specificity / F1, and Cohen's kappa."""
    square = counts[:, :, :n_reference]
    true_positive = np.einsum("bii->bi", square).astype(float)
    predicted_total = square.sum(axis=1)
    reference_total = counts.sum(axis=2)
    total = counts.sum(axis=(1, 2)).astype(float)

    false_positive = predicted_total - true_positive
    false_negative = reference_total - true_positive
    true_negative = total[:, None] - true_positive - false_positive - false_negative

    precision = np.divide(
        true_positive, predicted_total,
        out=np.zeros_like(true_positive), where=predicted_total > 0,
    )
    recall = np.divide(
        true_positive, reference_total,
        out=np.zeros_like(true_positive), where=reference_total > 0,
    )
    specificity_denominator = true_negative + false_positive
    specificity = np.divide(
        true_negative, specificity_denominator,
        out=np.zeros_like(true_positive), where=specificity_denominator > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(true_positive), where=denominator > 0,
    )

    observed = true_positive.sum(axis=1) / total
    column_total = counts.sum(axis=1).astype(float)
    expected = (reference_total * column_total[:, :n_reference]).sum(axis=1) / (total * total)
    kappa = np.divide(
        observed - expected, 1 - expected,
        out=np.full_like(observed, np.nan), where=~np.isclose(1 - expected, 0),
    )

    return {
        "accuracy": observed,
        "precision": precision.mean(axis=1),
        "recall": recall.mean(axis=1),
        "specificity": specificity.mean(axis=1),
        "f1": f1.mean(axis=1),
        "kappa": kappa,
        # Per-category values behind the macro averages above.
        "class_precision": precision,
        "class_recall": recall,
        "class_specificity": specificity,
        "class_f1": f1,
    }


def confusion_counts(codes: np.ndarray, n_cells: int, shape: tuple[int, int]) -> np.ndarray:
    """Tabulate one confusion matrix per row of ``codes`` (replicates, n)."""
    replicates = codes.shape[0]
    offsets = np.arange(replicates, dtype=np.int64)[:, None] * n_cells
    flat = np.bincount((codes + offsets).ravel(), minlength=replicates * n_cells)
    return flat.reshape(replicates, *shape)


def evaluate_model(
    reference: pd.Series,
    prediction: pd.Series,
    reference_labels: list[str],
    rng: np.random.Generator,
    replicates: int,
    per_class: bool = False,
    mask: np.ndarray | None = None,
) -> dict:
    """Point estimates plus percentile bootstrap bounds for one model.

    With ``mask``, metrics are computed only on the masked reports, but every
    replicate still resamples all reports, so the subset (and its size) is
    re-selected in each replicate. That is the unconditional bootstrap used for
    the hybrid agreement subset, matching the Cohort 2 hybrid intervals.
    """
    n = len(reference)
    n_reference = len(reference_labels)
    kept = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    extra_labels = sorted(set(prediction[kept]) - set(reference_labels))
    predicted_labels = reference_labels + extra_labels
    n_predicted = len(predicted_labels)

    # Reports outside the mask are tallied in an extra reference row that is
    # dropped before scoring; without a mask that row stays empty.
    reference_index = reference.map({label: i for i, label in enumerate(reference_labels)})
    predicted_index = prediction.map({label: i for i, label in enumerate(predicted_labels)})
    reference_index = np.where(kept, reference_index.to_numpy(), n_reference)
    predicted_index = np.where(kept, predicted_index.to_numpy(), 0)
    codes = (reference_index * n_predicted + predicted_index).astype(np.int64)
    n_cells = (n_reference + 1) * n_predicted
    shape = (n_reference + 1, n_predicted)

    def score(counts: np.ndarray) -> dict[str, np.ndarray]:
        return macro_metrics(counts[:, :n_reference, :], n_reference)

    observed = score(confusion_counts(codes[None, :], n_cells, shape))

    # One (replicates, n) draw per model, consumed in MODEL_COLUMNS order.
    indices = rng.integers(0, n, size=(replicates, n))

    samples = {name: np.empty(replicates) for name in METRICS}
    class_samples = (
        {name: np.empty((replicates, n_reference)) for name in CLASS_METRICS} if per_class else {}
    )
    # Chunked only to cap peak memory; the draws themselves are already fixed.
    for start in range(0, replicates, CHUNK):
        block = indices[start : start + CHUNK]
        batch = score(confusion_counts(codes[block], n_cells, shape))
        stop = start + len(block)
        for name in METRICS:
            samples[name][start:stop] = batch[name]
        for name in class_samples:
            class_samples[name][start:stop] = batch[f"class_{name}"]

    results = {}
    for name in METRICS:
        low, high = np.nanpercentile(samples[name], [2.5, 97.5])
        scale = 1.0 if name == "kappa" else 100.0
        results[name] = (float(observed[name][0]) * scale, float(low) * scale, float(high) * scale)

    if per_class:
        results["per_class"] = {
            label: {
                name: (
                    float(observed[f"class_{name}"][0, i]) * 100,
                    float(np.nanpercentile(class_samples[name][:, i], 2.5)) * 100,
                    float(np.nanpercentile(class_samples[name][:, i], 97.5)) * 100,
                )
                for name in CLASS_METRICS
            }
            for i, label in enumerate(reference_labels)
        }
    return results


def evaluate_hybrid(
    frame: pd.DataFrame, seed: int, replicates: int, per_class: bool = False
) -> dict:
    """Bootstrap the hybrid output: the CL output on reports where CL and LLM agree."""
    reference = normalise(frame[REFERENCE_COLUMN])
    cl = normalise(frame[MODEL_COLUMNS["Computational Linguistics (CL) Model"]])
    llm = normalise(frame[MODEL_COLUMNS["Llama 3.1 70B LLM"]])
    agreement = (cl == llm).to_numpy()
    return evaluate_model(
        reference,
        cl,
        sorted(reference[agreement].unique()),
        np.random.default_rng(seed + HYBRID_SEED_OFFSET),
        replicates,
        per_class,
        mask=agreement,
    )


def evaluate(
    frame: pd.DataFrame, seed: int, replicates: int, per_class: bool = False
) -> dict[str, dict]:
    """Bootstrap both models in a fixed order from one seeded generator."""
    reference = normalise(frame[REFERENCE_COLUMN])
    reference_labels = sorted(reference.unique())
    rng = np.random.default_rng(seed)
    return {
        name: evaluate_model(
            reference, normalise(frame[column]), reference_labels, rng, replicates, per_class
        )
        for name, column in MODEL_COLUMNS.items()
    }
