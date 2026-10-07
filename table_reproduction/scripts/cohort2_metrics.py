"""Shared Cohort 2 loading, metric and bootstrap helpers."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

CHARACTERISTICS = ["Size", "Calcification", "Density", "Laterality", "Lobe", "Margin", "Shape"]
MODELS = ["CL", "Llama 3.1 70B", "Llama 3.1 405B", "GPT-OSS-120B", "DeepSeek R1"]
HYBRID_PAIR = ("CL", "GPT-OSS-120B")
# LLMs paired with CL, in Table S15 order.
PAIRED_LLMS = ["Llama 3.1 70B", "Llama 3.1 405B", "GPT-OSS-120B", "DeepSeek R1"]

# Size is a measurement; the rest are categorical labels.
NUMERIC_CHARACTERISTICS = {"Size"}

# Stands in for "no value stated", which is a scored category, not missing data.
MISSING_TOKEN = "__MISSING__"

DEFAULT_SEED = 20260817
DEFAULT_REPLICATES = 10_000
CHUNK = 500
# The randomisation test draws from its own stream so it cannot shift any interval.
TEST_SEED_OFFSET = 8_000

CATEGORICAL_METRICS = ("precision", "recall", "specificity", "f1", "kappa")


def load_predictions(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Read the long-format predictions file and index it by (characteristic, model)."""
    frame = pd.read_csv(path, dtype=str)
    required = {"doc_id", "model", "characteristic", "ground_truth", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {sorted(missing)}")

    blocks = {
        key: block.sort_values("doc_id", ignore_index=True)
        for key, block in frame.groupby(["characteristic", "model"], sort=False)
    }
    expected = {(c, m) for c in CHARACTERISTICS for m in MODELS}
    if not expected.issubset(blocks):
        raise ValueError(f"Input is missing blocks: {sorted(expected - set(blocks))}")

    # Every model is scored against the same reference standard, stored with the
    # CL rows, by comparing canonical values (numerically for Size).
    for characteristic in CHARACTERISTICS:
        reference = blocks[(characteristic, "CL")].set_index("doc_id")["ground_truth"]
        for model in MODELS:
            block = blocks[(characteristic, model)]
            block["ground_truth"] = block["doc_id"].map(reference)
            block["correct"] = [
                normalise(truth, characteristic) == normalise(predicted, characteristic)
                for truth, predicted in zip(block["ground_truth"], block["prediction"])
            ]
    return blocks


def normalise(value: object, characteristic: str) -> object:
    """Canonical form of a stored value, for agreement and confusion matrices."""
    if pd.isna(value):
        return MISSING_TOKEN
    if characteristic in NUMERIC_CHARACTERISTICS:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value).strip().lower()
        return MISSING_TOKEN if np.isclose(number, -1.0) else round(number, 8)
    return str(value).strip().lower()


def round_half_up(value: float, digits: int) -> float:
    """Round halves away from zero, as the manuscript tables do."""
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def correctness(block: pd.DataFrame) -> np.ndarray:
    """Per-report correctness against the reference standard."""
    return block["correct"].to_numpy(dtype=bool)


def report_level_correct(
    blocks: dict[tuple[str, str], pd.DataFrame], model: str
) -> np.ndarray:
    """True where ``model`` got all seven characteristics right for that report."""
    correct = np.ones(len(blocks[(CHARACTERISTICS[0], model)]), dtype=bool)
    for characteristic in CHARACTERISTICS:
        correct &= correctness(blocks[(characteristic, model)])
    return correct


def _percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _confusion_codes(ground_truth: list, prediction: list) -> tuple[np.ndarray, int, int]:
    """Encode each report as a cell of a square confusion matrix."""
    reference_labels = sorted(set(ground_truth), key=lambda x: (x != MISSING_TOKEN, str(x)))
    extras = sorted(set(prediction) - set(reference_labels), key=str)
    labels = reference_labels + extras
    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    codes = np.array(
        [index[g] * size + index[p] for g, p in zip(ground_truth, prediction)], dtype=np.int64
    )
    return codes, size, len(reference_labels)


def _tabulate(codes: np.ndarray, size: int) -> np.ndarray:
    """One square confusion matrix per row of ``codes`` (replicates, n)."""
    replicates = codes.shape[0]
    cells = size * size
    offsets = np.arange(replicates, dtype=np.int64)[:, None] * cells
    flat = np.bincount((codes + offsets).ravel(), minlength=replicates * cells)
    return flat.reshape(replicates, size, size)


def _class_metrics(counts: np.ndarray, n_reference: int) -> dict[str, np.ndarray]:
    """Macro precision / recall / specificity / F1 and Cohen's kappa."""
    total = counts.sum(axis=(1, 2)).astype(float)
    row = counts.sum(axis=2).astype(float)
    column = counts.sum(axis=1).astype(float)
    diagonal = np.einsum("bii->bi", counts).astype(float)

    tp = diagonal[:, :n_reference]
    fn = row[:, :n_reference] - tp
    fp = column[:, :n_reference] - tp
    tn = total[:, None] - tp - fn - fp

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

        observed = diagonal.sum(axis=1) / total
        expected = (row * column).sum(axis=1) / (total * total)
        kappa = np.where(np.isclose(1 - expected, 0), np.nan, (observed - expected) / (1 - expected))

    return {
        "precision": np.nanmean(precision, axis=1) * 100,
        "recall": np.nanmean(recall, axis=1) * 100,
        "specificity": np.nanmean(specificity, axis=1) * 100,
        "f1": np.nanmean(f1, axis=1) * 100,
        "kappa": kappa,
    }


def _evaluate_subset_cell(
    mask: np.ndarray, correct: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float, float]:
    """Bootstrap a metric measured on a subset the models selected."""
    n = len(mask)
    samples = np.empty(replicates)
    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        drawn = mask[indices]
        kept = drawn.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            samples[start : start + count] = np.where(
                kept > 0, (correct[indices] & drawn).sum(axis=1) / kept, np.nan
            )
    low, high = _percentile_ci(samples)
    return float(correct[mask].mean()) * 100, low * 100, high * 100


def _evaluate_cell(
    correct: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
    codes: np.ndarray | None = None,
    size: int = 0,
    n_reference: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap one cell: accuracy always, class-aware metrics when asked."""
    n = len(correct)
    want_classes = codes is not None
    accuracy = np.empty(replicates)
    classes = {name: np.empty(replicates) for name in CATEGORICAL_METRICS} if want_classes else {}

    for start in range(0, replicates, CHUNK):
        count = min(CHUNK, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        accuracy[start : start + count] = correct[indices].mean(axis=1)
        if want_classes:
            batch = _class_metrics(_tabulate(codes[indices], size), n_reference)
            for name in CATEGORICAL_METRICS:
                classes[name][start : start + count] = batch[name]

    low, high = _percentile_ci(accuracy)
    results = {"accuracy": (correct.mean() * 100, low * 100, high * 100)}
    if want_classes:
        observed = _class_metrics(_tabulate(codes[None, :], size), n_reference)
        for name in CATEGORICAL_METRICS:
            low, high = _percentile_ci(classes[name])
            results[name] = (float(observed[name][0]), low, high)
    return results


def agreement_mask(
    blocks: dict[tuple[str, str], pd.DataFrame], characteristic: str, llm: str
) -> np.ndarray:
    """Reports where CL and ``llm`` extracted the same value for ``characteristic``."""
    return np.array(
        [
            normalise(a, characteristic) == normalise(b, characteristic)
            for a, b in zip(
                blocks[(characteristic, "CL")]["prediction"],
                blocks[(characteristic, llm)]["prediction"],
            )
        ]
    )


def randomisation_p_value(
    correct: np.ndarray, observed: float, subset_size: int, rng: np.random.Generator, replicates: int
) -> float:
    """Could an equally sized subset drawn without regard to agreement score this well?"""
    good = int(correct.sum())
    draws = rng.hypergeometric(good, len(correct) - good, subset_size, size=replicates)
    at_least_as_good = int((draws / subset_size >= observed).sum())
    return (at_least_as_good + 1) / (replicates + 1)


def evaluate(
    blocks: dict[tuple[str, str], pd.DataFrame],
    seed: int,
    replicates: int,
    categorical: bool = False,
    pairings: bool = False,
) -> tuple[dict, dict, dict]:
    """Bootstrap every Cohort 2 cell in a fixed order from one seeded generator."""
    rng = np.random.default_rng(seed)
    standalone: dict[tuple[str, str], dict] = {}
    hybrid: dict[str, dict] = {}

    for characteristic in CHARACTERISTICS:
        numeric = characteristic in NUMERIC_CHARACTERISTICS
        for model in MODELS:
            block = blocks[(characteristic, model)]
            codes = size = n_reference = None
            if categorical and not numeric:
                ground_truth = [normalise(v, characteristic) for v in block["ground_truth"]]
                prediction = [normalise(v, characteristic) for v in block["prediction"]]
                codes, size, n_reference = _confusion_codes(ground_truth, prediction)
            standalone[(characteristic, model)] = _evaluate_cell(
                correctness(block), rng, replicates, codes, size or 0, n_reference or 0
            )

    # Hybrid cells come after the standalone walk so that the two groups do not
    # interleave in the random stream.
    left, right = HYBRID_PAIR
    for characteristic in CHARACTERISTICS:
        agreement = agreement_mask(blocks, characteristic, right)
        cl_correct = correctness(blocks[(characteristic, left)])
        hybrid[characteristic] = {
            "accuracy": _evaluate_subset_cell(agreement, cl_correct, rng, replicates),
            "n": int(agreement.sum()),
        }

    if not pairings:
        return standalone, hybrid, {}

    # Second phase. Everything above is already final, so these draws cannot
    # perturb it.
    test_rng = np.random.default_rng(seed + TEST_SEED_OFFSET)
    paired: dict[tuple[str, str], dict] = {}
    for llm in PAIRED_LLMS:
        for characteristic in CHARACTERISTICS:
            agreement = agreement_mask(blocks, characteristic, llm)
            cl_correct = correctness(blocks[(characteristic, "CL")])
            llm_correct = correctness(blocks[(characteristic, llm)])
            subset_size = int(agreement.sum())

            rate = _evaluate_cell(agreement, rng, replicates)["accuracy"]
            if llm == HYBRID_PAIR[1]:
                # Reuse the Table 2 figure so the two tables cannot disagree.
                hybrid_accuracy = hybrid[characteristic]["accuracy"]
            else:
                hybrid_accuracy = _evaluate_subset_cell(agreement, cl_correct, rng, replicates)
            cl_disagree = _evaluate_subset_cell(~agreement, cl_correct, rng, replicates)
            llm_disagree = _evaluate_subset_cell(~agreement, llm_correct, rng, replicates)

            observed = cl_correct[agreement].mean()
            paired[(llm, characteristic)] = {
                "n": subset_size,
                "agreement_rate": rate,
                "hybrid_accuracy": hybrid_accuracy,
                "cl_disagreement": cl_disagree,
                "llm_disagreement": llm_disagree,
                "mc_p_cl": randomisation_p_value(
                    cl_correct, observed, subset_size, test_rng, replicates
                ),
                "mc_p_llm": randomisation_p_value(
                    llm_correct, observed, subset_size, test_rng, replicates
                ),
                "agreement_counts": (
                    int(cl_correct[agreement].sum()), subset_size - int(cl_correct[agreement].sum())
                ),
                "cl_disagreement_counts": (
                    int(cl_correct[~agreement].sum()), int((~agreement).sum()) - int(cl_correct[~agreement].sum())
                ),
                "llm_disagreement_counts": (
                    int(llm_correct[~agreement].sum()), int((~agreement).sum()) - int(llm_correct[~agreement].sum())
                ),
            }

    return standalone, hybrid, paired
