"""A minimal, illustrative zero-shot LLM extractor for the Lung-RADS category.

This is NOT the prompt or model evaluated in the manuscript; those are described
in Supplementary Section S4 and the full prompt is proprietary. It is a short
prompt written from scratch that follows the same outline: read the complete
report (including addenda), return the documented Lung-RADS category, restrict
the answer to the predefined categories, and constrain the output to a
structured format. It runs on OpenAI's Responses API at temperature 0.

The study itself ran its LLMs through AWS Bedrock; this example uses the OpenAI
API because it only needs an API key, which makes it easier to set up.
"""

from __future__ import annotations

import json

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

MISSING_LABEL = "N/A"
CATEGORIES = ["0", "1", "2", "3", "4", "4A", "4B", "4X", MISSING_LABEL]

INSTRUCTIONS = f"""\
You are a radiology assistant.
Classify the radiology report below into a Lung-RADS category, considering all sections including any addendum.
Answer with one of: {", ".join(CATEGORIES)}."""

# Structured output: the answer must be one of the predefined categories.
OUTPUT_FORMAT = {
    "type": "json_schema",
    "name": "lung_rads",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"lung_rads_category": {"type": "string", "enum": CATEGORIES}},
        "required": ["lung_rads_category"],
        "additionalProperties": False,
    },
}


def extract(client: OpenAI, text: str, model: str = DEFAULT_MODEL) -> str:
    """Return the Lung-RADS category the LLM reads from one report."""
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=text,  # The complete report, with no truncation or preprocessing.
        temperature=0,
        text={"format": OUTPUT_FORMAT},
    )
    return json.loads(response.output_text)["lung_rads_category"]
