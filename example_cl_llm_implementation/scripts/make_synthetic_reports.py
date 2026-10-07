"""Generate a synthetic set of lung cancer screening reports with Lung-RADS labels.

Every report is assembled from made-up template text: no real report, patient,
clinician, site, date, or accession number is used or referenced. Findings are
written to be consistent with the assigned Lung-RADS category, and a few reports
include the distractors real reports contain (prior-exam categories, the S
modifier, category legends, Fleischner boilerplate) so that the example models
have something non-trivial to do.

A small number of reports are deliberately hard (see ``HARD_CASES``): they use
phrasings that real reports contain but that a simple rule set does not cover,
such as spelled-out categories, prior-exam categories without a cue word, and
addenda that revise the category.
"""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXAMPLE_ROOT / "input_data" / "synthetic_lung_rads_reports.csv"

DEFAULT_SEED = 20261005
DEFAULT_N_REPORTS = 100

MISSING_LABEL = "N/A"
CATEGORIES = ["0", "1", "2", "3", "4", "4A", "4B", "4X", MISSING_LABEL]

LOBES = [
    "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "lingula", "left lower lobe",
]

DESCRIPTORS = {
    "0": "Incomplete",
    "1": "Negative",
    "2": "Benign appearance or behavior",
    "3": "Probably benign",
    "4": "Suspicious",
    "4A": "Suspicious",
    "4B": "Very suspicious",
    "4X": "Very suspicious",
}

RECOMMENDATIONS = {
    "1": "Continue annual screening with low-dose CT in 12 months.",
    "2": "Continue annual screening with low-dose CT in 12 months.",
    "3": "Low-dose chest CT in 6 months.",
    "4A": "Low-dose chest CT in 3 months. PET/CT may be considered if there is a solid component of 8 mm or larger.",
    "4B": "Diagnostic chest CT with or without contrast, PET/CT and/or tissue sampling, depending on the probability of malignancy and comorbidities.",
    "4X": "Diagnostic chest CT with or without contrast, PET/CT and/or tissue sampling. Referral to pulmonology or thoracic surgery is recommended.",
    "4": "Further evaluation with PET/CT and referral to the multidisciplinary lung nodule clinic.",
}

# Each inner list holds mutually exclusive alternatives; at most one is drawn per group.
OTHER_FINDINGS = [
    ["Mild centrilobular emphysema, upper lobe predominant.", "Moderate centrilobular emphysema."],
    ["Mild coronary artery calcification.", "Moderate coronary artery calcification."],
    ["Mild atherosclerosis of the thoracic aorta."],
    ["Mild bronchial wall thickening."],
    ["Small hiatal hernia."],
    ["Mild degenerative changes of the thoracic spine."],
    ["Scattered calcified mediastinal lymph nodes, consistent with prior granulomatous disease."],
    ["No pleural effusion."],
    ["Heart size is normal. No pericardial effusion."],
    ["The visualized upper abdomen is unremarkable.", "Simple-appearing cyst in the partially imaged left kidney."],
]

SCREENING_EXAMS = [
    "CT CHEST LOW DOSE LUNG CANCER SCREENING WITHOUT CONTRAST",
    "Low-dose CT chest for lung cancer screening",
    "CT LUNG SCREENING WO CONTRAST",
    "LDCT LUNG CANCER SCREENING",
]

# (exam, indication, comparison, exam-specific findings, impression)
NON_SCREENING_EXAMS = [
    ("CT ANGIOGRAPHY CHEST WITH CONTRAST", "Shortness of breath, evaluate for pulmonary embolism.", "None.",
     "Pulmonary arteries: No filling defect to suggest pulmonary embolism.", "No pulmonary embolism."),
    ("CT ABDOMEN AND PELVIS WITH CONTRAST", "Abdominal pain.", "None.",
     "Abdomen and pelvis: Normal appendix. No bowel obstruction or free fluid.", "No acute abdominopelvic process."),
    ("CT CHEST WITH CONTRAST", "Follow-up of pneumonia.", "Chest CT from 6 weeks earlier.",
     "Residual patchy opacity in the right lower lobe, decreased from prior.", "Improving right lower lobe pneumonia."),
    ("PET/CT SKULL BASE TO MID-THIGH", "Restaging of lymphoma.", "PET/CT from 3 months earlier.",
     "No hypermetabolic lymphadenopathy in the neck, chest, abdomen, or pelvis.", "Complete metabolic response. No FDG-avid disease."),
    ("CT CHEST WITHOUT CONTRAST", "Chronic cough.", "None.",
     "Airways: Mild bronchial wall thickening without bronchiectasis.", "Mild bronchial wall thickening, which can be seen with bronchitis or reactive airways disease."),
    ("CT CHEST ABDOMEN PELVIS WITH CONTRAST", "Trauma, motor vehicle collision.", "None.",
     "No pneumothorax, hemothorax, solid organ injury, or acute fracture.", "No acute traumatic injury in the chest, abdomen, or pelvis."),
]


# Deliberately hard cases: (name, labels it can be applied to, number of reports).
HARD_CASES = [
    ("spelled_out", {"2", "3", "4A", "4B"}, 1),
    ("no_keyword", {"1", "3", "4X"}, 1),
    ("reference_table", {"1", "2"}, 1),
    ("prior_without_cue", {"2"}, 1),
    ("downgrade_same_sentence", {"3"}, 1),
    ("addendum_revision", {"2"}, 1),
    ("diagnostic_follow_up", {MISSING_LABEL}, 2),
]

SPELLED_OUT = {"1": "one", "2": "two", "3": "three", "4": "four", "4A": "four A", "4B": "four B", "4X": "four X"}

REFERENCE_TABLE = [
    "ACR Lung-RADS reference:",
    "Lung-RADS 1: Negative. Continue annual screening.",
    "Lung-RADS 2: Benign appearance or behavior. Continue annual screening.",
    "Lung-RADS 3: Probably benign. Low-dose CT in 6 months.",
    "Lung-RADS 4A: Suspicious. Low-dose CT in 3 months.",
    "Lung-RADS 4B: Very suspicious. Diagnostic chest CT, PET/CT and/or tissue sampling.",
    "Lung-RADS 4X: Category 3 or 4 nodule with additional suspicious features.",
]


def mm(rng: random.Random, low: float, high: float) -> str:
    """A diameter in [low, high), printed to one decimal place without a trailing .0."""
    value = round(rng.uniform(low, high - 0.1), 1)
    return f"{value:g}"


def a_mm(rng: random.Random, low: float, high: float) -> str:
    """A diameter with its indefinite article, e.g. "a 7.2 mm" or "an 8.7 mm"."""
    value = mm(rng, low, high)
    whole = int(float(value))
    article = "an" if whole in (8, 11, 18) or 80 <= whole <= 89 else "a"
    return f"{article} {value} mm"


def location(rng: random.Random) -> str:
    lobe = rng.choice(LOBES)
    return f"{lobe} (series {rng.randint(2, 6)}, image {rng.randint(20, 280)})"


def findings_for(category: str, rng: random.Random) -> dict:
    """Return nodule findings, impression text and whether a prior exam exists."""
    if category == "1":
        option = rng.randrange(3)
        if option == 0:
            return dict(prior=rng.random() < 0.5, nodules=["No pulmonary nodules."],
                        impression="No pulmonary nodules.")
        if option == 1:
            return dict(prior=rng.random() < 0.5,
                        nodules=[f"{mm(rng, 3, 8)} mm densely calcified nodule in the {location(rng)}, consistent with a granuloma."],
                        impression="Benign calcified granuloma. No suspicious pulmonary nodules.")
        return dict(prior=False,
                    nodules=[f"{mm(rng, 6, 14)} mm nodule in the {location(rng)} containing macroscopic fat, consistent with a hamartoma."],
                    impression="Benign fat-containing nodule. No suspicious pulmonary nodules.")

    if category == "2":
        option = rng.randrange(4)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 2, 6)} mm solid nodule in the {location(rng)}."],
                        impression="Small solid pulmonary nodule with benign appearance.")
        if option == 1:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 4, 10)} mm perifissural nodule along the right minor fissure, triangular in shape, likely an intrapulmonary lymph node."],
                        impression="Perifissural nodule, likely an intrapulmonary lymph node.")
        if option == 2:
            return dict(prior=rng.random() < 0.5,
                        nodules=[f"{mm(rng, 8, 28)} mm ground-glass nodule in the {location(rng)}."],
                        impression="Ground-glass nodule measuring less than 3 cm.")
        return dict(prior=True, prior_category="3", comparison="Prior low-dose screening CT from 6 months earlier.",
                    nodules=[f"{mm(rng, 6, 8)} mm solid nodule in the {location(rng)}, unchanged from the prior exam."],
                    impression="Stable solid pulmonary nodule.")

    if category == "3":
        option = rng.randrange(3)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 6, 8)} mm solid nodule in the {location(rng)}."],
                        impression="Solid pulmonary nodule measuring between 6 and 8 mm.")
        if option == 1:
            return dict(prior=True, prior_category="2",
                        nodules=[f"New {mm(rng, 4, 6)} mm solid nodule in the {location(rng)}."],
                        impression="New small solid pulmonary nodule.")
        return dict(prior=False,
                    nodules=[f"{mm(rng, 7, 14)} mm part-solid nodule in the {location(rng)} with {a_mm(rng, 2, 5)} solid component."],
                    impression="Part-solid nodule with a small solid component.")

    if category == "4A":
        option = rng.randrange(4)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 8, 15)} mm solid nodule in the {location(rng)}."],
                        impression="Solid pulmonary nodule measuring between 8 and 15 mm.")
        if option == 1:
            return dict(prior=True, prior_category="3", comparison="Prior low-dose screening CT from 6 months earlier.",
                        nodules=[f"Solid nodule in the {location(rng)} now measures {mm(rng, 6.5, 8)} mm, previously {mm(rng, 4.5, 6)} mm."],
                        impression="Interval growth of a solid pulmonary nodule.")
        if option == 2:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 10, 18)} mm part-solid nodule in the {location(rng)} with {a_mm(rng, 6, 8)} solid component."],
                        impression="Part-solid nodule with a solid component between 6 and 8 mm.")
        return dict(prior=True,
                    nodules=[f"{mm(rng, 4, 7)} mm endobronchial nodule in the {rng.choice(['right lower lobe', 'left lower lobe'])} segmental bronchus."],
                    impression="Endobronchial nodule.")

    if category == "4B":
        option = rng.randrange(3)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 15, 28)} mm solid nodule in the {location(rng)}."],
                        impression="Solid pulmonary nodule measuring at least 15 mm.")
        if option == 1:
            return dict(prior=True, prior_category=rng.choice(["1", "2"]),
                        nodules=[f"New {mm(rng, 9, 14)} mm solid nodule in the {location(rng)}."],
                        impression="New solid pulmonary nodule larger than 8 mm.")
        return dict(prior=False,
                    nodules=[f"{mm(rng, 14, 25)} mm part-solid nodule in the {location(rng)} with {a_mm(rng, 8.5, 13)} solid component."],
                    impression="Part-solid nodule with a solid component larger than 8 mm.")

    if category == "4X":
        option = rng.randrange(3)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 12, 26)} mm solid nodule with spiculated margins in the {location(rng)}."],
                        impression="Spiculated solid pulmonary nodule, highly suspicious for malignancy.")
        if option == 1:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 16, 30)} mm solid nodule in the {location(rng)}.",
                                 f"Enlarged ipsilateral hilar lymph node measuring {mm(rng, 12, 18)} mm in short axis."],
                        impression="Large solid nodule with associated hilar lymphadenopathy, suspicious for malignancy.")
        return dict(prior=True, prior_category="2",
                    nodules=[f"Ground-glass nodule in the {location(rng)} has doubled in size over 1 year and now measures {mm(rng, 18, 30)} mm."],
                    impression="Ground-glass nodule that has doubled in size within 1 year.")

    if category == "4":
        option = rng.randrange(2)
        if option == 0:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 3, 6.5)} cm mass in the {location(rng)}."],
                        impression="Lung mass suggestive of malignancy.")
        return dict(prior=rng.random() < 0.5,
                    nodules=[f"{mm(rng, 11, 20)} mm irregular solid nodule in the {location(rng)}."],
                    impression="Irregular solid pulmonary nodule, suspicious.")

    if category == "0":
        option = rng.randrange(4)
        if option == 0:
            return dict(prior=False,
                        nodules=["Multiple new patchy nodular opacities in both lower lobes with surrounding ground-glass, favored to be infectious or inflammatory."],
                        impression="Findings favored to be infectious or inflammatory. Short-term follow-up low-dose CT in 1 to 3 months is recommended.",
                        recommendation="Low-dose chest CT in 1 to 3 months.")
        if option == 1:
            return dict(prior=False,
                        nodules=["The lung bases were not included in the scan field of view."],
                        impression="Incomplete examination; the lung bases were not imaged.",
                        recommendation="Repeat low-dose CT to include the entire chest.")
        if option == 2:
            return dict(prior=False,
                        nodules=[f"{mm(rng, 5, 9)} mm solid nodule in the {location(rng)}. An outside prior chest CT is reported to exist but is not available for comparison."],
                        impression="Pulmonary nodule; comparison with the outside prior CT is needed before a final assessment.",
                        recommendation="Please forward the outside prior CT for comparison. An addendum will be issued.")
        return dict(prior=False,
                    nodules=["Severe respiratory motion artifact limits evaluation of the lower lungs."],
                    impression="Nondiagnostic evaluation of the lower lungs due to motion.",
                    recommendation="Repeat low-dose CT.")

    raise ValueError(f"No findings template for category {category!r}")


def lung_rads_statement(category: str, rng: random.Random) -> str:
    """The sentence(s) that state the assigned Lung-RADS category, in one of several house styles."""
    descriptor = DESCRIPTORS[category]
    styles = [
        f"Lung-RADS category: {category}",
        f"LUNG-RADS CATEGORY: {category} - {descriptor.upper()}",
        f"Lung-RADS {category}: {descriptor}.",
        f"Lung-RADS Assessment: Category {category} ({descriptor}).",
        f"LungRADS category: {category}\nLungRADS modifier: None",
        f"ACR Lung-RADS v2022 category {category}, {descriptor.lower()}.",
        f"Lung RADS {category}",
        f"LUNG RADS SCORE: {category}",
    ]
    return rng.choice(styles)


def other_findings(rng: random.Random, already_stated: str = "") -> list[str]:
    chosen = [rng.choice(group) for group in rng.sample(OTHER_FINDINGS, rng.randint(1, 3))]
    return [finding for finding in chosen if finding.rstrip(".") not in already_stated]


def wrap(text: str, rng_wrap: bool) -> str:
    """Hard-wrap each line at 70 characters for roughly half the reports, as dictation systems do."""
    if not rng_wrap:
        return text
    return "\n".join(
        textwrap.fill(line, width=70, break_on_hyphens=False) if line else line for line in text.split("\n")
    )


def hard_details(hard: str, category: str, rng: random.Random) -> dict:
    """Findings and impression for hard cases whose clinical story differs from the templates."""
    if hard == "prior_without_cue":
        return dict(prior=True, comparison="Prior low-dose screening CT from 6 months earlier.",
                    nodules=[f"{mm(rng, 4, 5.5)} mm solid nodule in the {location(rng)}, decreased from {mm(rng, 6, 8)} mm."],
                    impression=[
                        "1. Decreasing solid pulmonary nodule, likely inflammatory.",
                        "On the exam 6 months ago this was Lung-RADS 3.",
                        lung_rads_statement(category, rng),
                    ])
    if hard == "downgrade_same_sentence":
        return dict(prior=True, comparison="Low-dose chest CT from 3 months earlier.",
                    nodules=[f"{mm(rng, 8, 14)} mm solid nodule in the {location(rng)}, unchanged."],
                    impression=[
                        "1. Stable solid pulmonary nodule at 3-month follow-up.",
                        "Previously Lung-RADS 4A, now downgraded to Lung-RADS 3.",
                    ])
    if hard == "addendum_revision":
        details = findings_for("0", rng)
        while "outside prior" not in details["nodules"][0]:
            details = findings_for("0", rng)
        details["impression"] = [f"1. {details['impression']}", lung_rads_statement("0", rng)]
        details["addendum"] = [
            "",
            "ADDENDUM:",
            "The outside chest CT from 2 years earlier has since been reviewed. The nodule is "
            "unchanged over this interval. The Lung-RADS category is revised to 2. Continue "
            "annual screening with low-dose CT in 12 months.",
        ]
        return details
    raise ValueError(hard)


def screening_report(category: str, rng: random.Random, hard: str | None = None) -> str:
    if hard in ("prior_without_cue", "downgrade_same_sentence", "addendum_revision"):
        details = hard_details(hard, category, rng)
    else:
        details = findings_for(category, rng)
    prior = details["prior"]
    findings_header = rng.choice(["FINDINGS:", "Findings:"])
    impression_header = rng.choice(["IMPRESSION:", "Impression:", "CONCLUSION:"])

    lines = [f"EXAM: {rng.choice(SCREENING_EXAMS)}"]
    history = "Lung cancer screening. " + ("Annual follow-up exam." if prior else "Baseline exam.")
    if rng.random() < 0.6:
        history += rng.choice([
            f" Former smoker, {rng.randint(20, 60)} pack-years.",
            f" Current smoker, {rng.randint(20, 60)} pack-years.",
            " Asymptomatic.",
        ])
    lines.append(f"CLINICAL HISTORY: {history}")
    if rng.random() < 0.5:
        lines.append("TECHNIQUE: Low-dose helical CT of the chest without intravenous contrast.")
    default_comparison = "Prior low-dose screening CT from 1 year earlier." if prior else "None."
    lines.append(f"COMPARISON: {details.get('comparison', default_comparison)}")
    lines.append("")
    lines.append(findings_header)
    lines.append("Lungs: " + " ".join(details["nodules"]))
    extra = other_findings(rng)
    lines.extend(extra)
    lines.append("")

    lines.append(impression_header)
    statement = lung_rads_statement(category, rng)
    if hard == "spelled_out":
        statement = rng.choice([f"Lung-RADS category {SPELLED_OUT[category]}.", f"Lung-RADS {SPELLED_OUT[category]}."])
    elif hard == "no_keyword":
        statement = f"{rng.choice(['Assessment category', 'ACR category'])}: {category} ({DESCRIPTORS[category]})."
    recommendation = details.get("recommendation", RECOMMENDATIONS.get(category))
    inline = rng.random() < 0.25 and "\n" not in statement
    if isinstance(details["impression"], list):
        lines.extend(details["impression"])
        statement = ""
    elif inline:
        lines.append(f"1. {details['impression']} {statement}")
    else:
        lines.append(f"1. {details['impression']}")
        if "prior_category" in details and rng.random() < 0.7:
            lines.append(f"2. Prior exam was assessed as Lung-RADS {details['prior_category']}.")
        lines.append(statement)
    # The S modifier is only stated when the findings support it.
    if "Moderate coronary artery calcification." in extra and "modifier" not in statement and rng.random() < 0.6:
        lines.append("Lung-RADS modifier S: Yes, clinically significant coronary artery calcification.")
    # Some templates print the category reference table inside the impression itself.
    if hard == "reference_table":
        lines.extend(REFERENCE_TABLE)
    lines.append(f"RECOMMENDATION: {recommendation}")
    lines.extend(details.get("addendum", []))

    if hard is None and rng.random() < 0.15:
        lines.append("")
        lines.append(
            "Lung-RADS key: 0 incomplete; 1 negative; 2 benign; 3 probably benign; "
            "4A suspicious; 4B and 4X very suspicious; S significant other finding."
        )
    return wrap("\n".join(lines), rng.random() < 0.5)


def diagnostic_follow_up_report(rng: random.Random) -> str:
    """A diagnostic (non-screening) CT that mentions the category from an earlier screening exam."""
    prior_category = rng.choice(["4A", "4B"])
    lines = [
        "EXAM: CT CHEST WITH CONTRAST",
        "INDICATION: Follow-up of a pulmonary nodule found on lung cancer screening.",
        "COMPARISON: Low-dose screening CT from 3 months earlier.",
        "",
        rng.choice(["FINDINGS:", "Findings:"]),
        f"Lungs: {mm(rng, 9, 14)} mm solid nodule in the {location(rng)}, unchanged in size.",
        *other_findings(rng),
        "",
        rng.choice(["IMPRESSION:", "Impression:"]),
        "1. Stable solid pulmonary nodule.",
        f"2. The nodule was assigned Lung-RADS {prior_category} on the screening exam. Lung-RADS "
        "categories apply only to screening CT, so none is assigned to this diagnostic study.",
        "RECOMMENDATION: Discussion at the multidisciplinary lung nodule conference.",
    ]
    return wrap("\n".join(lines), rng.random() < 0.5)


def non_screening_report(rng: random.Random) -> str:
    exam, indication, comparison, exam_finding, impression = rng.choice(NON_SCREENING_EXAMS)
    lines = [f"EXAM: {exam}", f"INDICATION: {indication}", f"COMPARISON: {comparison}", ""]
    lines.append(rng.choice(["FINDINGS:", "Findings:"]))

    incidental_nodule = rng.random() < 0.5
    lines.append(exam_finding)
    if incidental_nodule:
        lines.append(f"Lungs: {mm(rng, 3, 6)} mm solid nodule in the {rng.choice(LOBES)}.")
    lines.extend(other_findings(rng, already_stated=exam_finding))
    lines.append("")

    lines.append(rng.choice(["IMPRESSION:", "Impression:"]))
    lines.append(f"1. {impression}")
    if incidental_nodule:
        lines.append(
            "2. Small incidental pulmonary nodule under 6 mm. Per Fleischner Society guidelines, "
            "no routine follow-up is required; optional CT in 12 months if high risk."
        )
        if rng.random() < 0.6:
            lines.append(
                "Note: Fleischner Society guidelines do not apply to lung cancer screening "
                "patients, for whom Lung-RADS should be used."
            )
    return wrap("\n".join(lines), rng.random() < 0.5)


def build_reports(n_reports: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    # Near-equal representation: cycle through the categories, then shuffle the order.
    labels = [CATEGORIES[i % len(CATEGORIES)] for i in range(n_reports)]
    rng.shuffle(labels)

    # Assign each hard case to randomly chosen reports with a compatible label.
    hard_by_index = {}
    for name, allowed, count in HARD_CASES:
        candidates = [i for i, label in enumerate(labels) if label in allowed and i not in hard_by_index]
        for index in rng.sample(candidates, count):
            hard_by_index[index] = name

    rows = []
    seen = set()
    for index, label in enumerate(labels):
        hard = hard_by_index.get(index)
        while True:
            if hard == "diagnostic_follow_up":
                text = diagnostic_follow_up_report(rng)
            elif label == MISSING_LABEL:
                text = non_screening_report(rng)
            else:
                text = screening_report(label, rng, hard)
            if text not in seen:
                break
        seen.add(text)
        rows.append({"doc_id": str(index + 1), "text_original": text, "max_rads_annotated": label})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination CSV.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--n", type=int, default=DEFAULT_N_REPORTS, help="Number of reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_reports(args.n, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    counts = {c: sum(r["max_rads_annotated"] == c for r in rows) for c in CATEGORIES}
    print(f"{len(rows)} reports: " + ", ".join(f"{c}={n}" for c, n in counts.items()))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
