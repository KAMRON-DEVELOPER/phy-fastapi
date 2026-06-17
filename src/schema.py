from enum import StrEnum

from pydantic import BaseModel, Field


class InsightCategory(StrEnum):
    """What kind of language feature an insight is flagging."""

    # IDIOM = "idiom"
    # SLANG = "slang"
    # SARCASM = "sarcasm"
    # PHRASAL_VERB = "phrasal_verb"

    HUMOR = "humor"
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    CULTURAL_REFERENCE = "cultural_reference"


class TranscriptInsight(BaseModel):
    excerpt: str
    explanation: str
    category: InsightCategory


class TranscriptInsightsResult(BaseModel):
    """
    Shape the model must return for a single insight-extraction call.
    Wrapped in an object (rather than a bare list) because structured
    outputs require a JSON object at the root.
    """

    insights: list[TranscriptInsight] = Field(default_factory=list)


class TranscriptSegmentResponse(BaseModel):
    start_seconds: float
    end_seconds: float
    transcript: str
    insights: list[TranscriptInsight] = Field(default_factory=list)
