"""A minimal, illustrative computational linguistics (CL) model for Lung-RADS extraction.

This is NOT the production CL model evaluated in the manuscript, which is
proprietary and has more than 5,000 rules. It is a deliberately small
re-creation of the architecture described in Supplementary Section S3, written
from scratch with a handful of generic rules, so readers can see how a
deterministic, rule-based extractor arrives at a report-level Lung-RADS category.

The pipeline runs four tiers of rules, each a spaCy component:

1. ``tier1_mark``    - token patterns mark Lung-RADS mentions and section headers,
                       each tagged with a canonical concept identifier (``kb_id``).
2. ``tier2_refine``  - mentions inside boilerplate (category legends) are discarded.
3. ``tier3_context`` - each mention is attached to its report section and flagged
                       if it describes a prior exam rather than the current one.
4. ``tier4_select``  - ordered criteria pick one current mention, and a fixed
                       priority order breaks ties.

No component is stochastic, so a given report always yields the same output.
"""

from __future__ import annotations

import re

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

MISSING_LABEL = "N/A"

# Fixed priority order used to break ties: later entries are more severe.
CATEGORY_ORDER = ["0", "1", "2", "3", "4", "4A", "4B", "4X"]

MENTIONS = "lung_rads_mentions"
SECTIONS = "sections"
BOILERPLATE = "boilerplate"

# Section headers and the canonical section each one maps to.
SECTION_HEADERS = {
    "exam": "exam",
    "history": "history",
    "indication": "history",
    "technique": "technique",
    "comparison": "comparison",
    "findings": "findings",
    "impression": "impression",
    "conclusion": "impression",
    "recommendation": "recommendation",
    "recommendations": "recommendation",
}
# Tier 4, criterion 1: mentions in these sections are trusted first.
PREFERRED_SECTIONS = {"impression"}

# Words that place a mention on a prior exam rather than the current one.
PRIOR_CUES = {"prior", "previous", "previously"}

# Tier 1 token patterns: a Lung-RADS keyword, optional connecting words, then a value.
KEYWORD_SHAPES = [
    [{"LOWER": "lungrads"}],                                          # LungRADS
    [{"LOWER": "lung"}, {"ORTH": "-", "OP": "?"}, {"LOWER": "rads"}],  # Lung-RADS, Lung RADS
]
CONNECTORS = [
    {"LOWER": {"REGEX": r"^v\d"}, "OP": "?"},                             # v2022
    {"LOWER": {"IN": ["category", "assessment", "score"]}, "OP": "?"},
    {"IS_PUNCT": True, "OP": "?"},                                        # :
    {"LOWER": "category", "OP": "?"},
]
VALUE = {"LOWER": {"IN": [c.lower() for c in CATEGORY_ORDER]}}
# Dictation systems hard-wrap lines, so a line break may fall between any two tokens.
LINE_BREAK = {"IS_SPACE": True, "OP": "?"}


def allow_line_breaks(pattern: list[dict]) -> list[dict]:
    wrapped = []
    for token in pattern:
        wrapped.extend([token, LINE_BREAK])
    return wrapped[:-1]


# Tier 2: a category legend runs from its heading to the next blank line.
LEGEND = re.compile(r"lung-?\s?rads\s+(?:key|categories|legend)\s*:.*?(?:\n\s*\n|\Z)", re.I | re.S)

Span.set_extension("section", default=None, force=True)
Span.set_extension("is_prior", default=False, force=True)
Doc.set_extension("lung_rads", default=MISSING_LABEL, force=True)
Doc.set_extension("lung_rads_evidence", default=None, force=True)


def category_of(span: Span) -> str:
    """The normalized category stored in a mention's concept identifier, e.g. "4X"."""
    return span.kb_id_.split(":", 1)[1]


@Language.factory("tier1_mark")
def create_tier1_mark(nlp: Language, name: str):
    matcher = Matcher(nlp.vocab)
    matcher.add(
        "LUNG_RADS",
        [allow_line_breaks(shape + CONNECTORS + [VALUE]) for shape in KEYWORD_SHAPES],
        greedy="LONGEST",
    )
    matcher.add(
        "SECTION",
        [[{"LOWER": {"IN": list(SECTION_HEADERS)}}, {"ORTH": ":"}]],
    )

    def tier1_mark(doc: Doc) -> Doc:
        mentions, sections = [], []
        for match_id, start, end in matcher(doc):
            label = nlp.vocab.strings[match_id]
            if label == "LUNG_RADS":
                value = doc[end - 1].text.upper()
                mentions.append(Span(doc, start, end, label=label, kb_id=f"LUNGRADS:{value}"))
            else:
                # A header counts only at the start of a line.
                if start > 0 and "\n" not in doc[start - 1].text:
                    continue
                section = SECTION_HEADERS[doc[start].lower_]
                sections.append(Span(doc, start, end, label=label, kb_id=f"SECTION:{section}"))
                doc[start].is_sent_start = True  # Headers never continue the previous sentence.
        # Longest match wins among overlapping mentions.
        doc.spans[MENTIONS] = filter_spans(mentions)
        doc.spans[SECTIONS] = sorted(sections, key=lambda span: span.start)
        return doc

    return tier1_mark


@Language.component("tier2_refine")
def tier2_refine(doc: Doc) -> Doc:
    legends = [
        doc.char_span(m.start(), m.end(), label=BOILERPLATE, alignment_mode="expand")
        for m in LEGEND.finditer(doc.text)
    ]
    doc.spans[BOILERPLATE] = legends
    doc.spans[MENTIONS] = [
        mention for mention in doc.spans[MENTIONS]
        if not any(legend.start <= mention.start < legend.end for legend in legends)
    ]
    return doc


@Language.component("tier3_context")
def tier3_context(doc: Doc) -> Doc:
    for mention in doc.spans[MENTIONS]:
        headers = [s for s in doc.spans[SECTIONS] if s.start < mention.start]
        mention._.section = headers[-1].kb_id_.split(":", 1)[1] if headers else None
        mention._.is_prior = any(token.lower_ in PRIOR_CUES for token in mention.sent)
    return doc


@Language.component("tier4_select")
def tier4_select(doc: Doc) -> Doc:
    current = [mention for mention in doc.spans[MENTIONS] if not mention._.is_prior]
    # Ordered criteria: stop at the first one that returns any mention.
    criteria = [
        lambda mention: mention._.section in PREFERRED_SECTIONS,
        lambda mention: True,
    ]
    for criterion in criteria:
        hits = [mention for mention in current if criterion(mention)]
        if hits:
            best = max(hits, key=lambda mention: CATEGORY_ORDER.index(category_of(mention)))
            doc._.lung_rads = category_of(best)
            doc._.lung_rads_evidence = best
            break
    return doc


def build_cl_pipeline() -> Language:
    """A blank English pipeline (no trained components) with the four rule tiers."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    for name in ("tier1_mark", "tier2_refine", "tier3_context", "tier4_select"):
        nlp.add_pipe(name)
    return nlp


def extract(nlp: Language, text: str) -> tuple[str, str | None]:
    """Return the report-level Lung-RADS category and the text span supporting it."""
    doc = nlp(text)
    evidence = doc._.lung_rads_evidence
    return doc._.lung_rads, evidence.text if evidence is not None else None
