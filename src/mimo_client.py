import base64
import json
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import SecretStr, ValidationError

from src.schema import TranscriptInsight

logger = logging.getLogger("phy.mimo")


class MimoError(Exception):
    pass


class ApiKeyError(MimoError):
    pass


MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"

_model: AsyncOpenAI | None = None


def get_model() -> AsyncOpenAI:
    """
    Creates model instance
    """

    load_dotenv()
    api_key = SecretStr(os.environ.get("MIMO_API_KEY", ""))
    if not api_key.get_secret_value():
        raise ApiKeyError("MIMO_API_KEY environment variable is not set")

    global _model
    if _model is None:
        _model = AsyncOpenAI(
            api_key=api_key.get_secret_value(), base_url=MIMO_BASE_URL
        )
    return _model


async def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Calls mimo-v2.5-asr on a single audio chunk and returns the raw transcript text.
    """

    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    size_mb = len(base64_audio) / (1024 * 1024)
    if size_mb > 10:
        raise MimoError(
            f"Encoded audio is {size_mb:.1f}MB, exceeds MiMo's 10MB ASR limit. "
            "Reduce chunk length or sample rate."
        )

    data = f"data:audio/wav;base64,{base64_audio}"

    model = get_model()

    audio_part: ChatCompletionContentPartInputAudioParam = {
        "type": "input_audio",
        "input_audio": {"data": data, "format": "wav"},
    }
    messages: list[ChatCompletionUserMessageParam] = [
        {"role": "user", "content": [audio_part]}
    ]

    try:
        completion = await model.chat.completions.create(
            model="mimo-v2.5-asr",
            messages=messages,
            extra_body={"asr_options": {"language": "en"}},
        )
    except Exception as e:
        raise MimoError(f"MiMo ASR call failed: {e}") from e

    if not completion.choices:
        raise MimoError("MiMo ASR response had no choices")

    content = completion.choices[0].message.content

    if not content:
        raise MimoError("MiMo ASR response had no message content")

    return content


TRANSCRIPT_INSIGHTS_SYSTEM_PROMPT = """\
You are helping English learners understand a transcript.
 
Given a transcript excerpt, identify only the parts that an intermediate
English learner would likely not fully understand.
 
Categories:
 
- vocabulary:
  idioms, phrasal verbs, slang, uncommon expressions, or words whose
  meaning is not obvious from the literal text
 
- grammar:
  sentence structures that are difficult to parse or understand
 
- humor:
  jokes, sarcasm, wordplay, irony, or statements whose intended meaning
  differs from the literal wording
 
- cultural_reference:
  references to people, events, media, brands, customs, or shared cultural
  knowledge that many learners may not recognize
 
Rules:
 
- Ignore plain literal language.
- Ignore incomplete thoughts or sentence fragments unless they are
  independently understandable.
- The excerpt must be an exact substring from the transcript.
- Return at most 5 insights, ordered by how useful they are to a learner.
- If nothing is worth explaining, return an empty insights list.

Respond with a JSON object of this exact shape:
{
  "insights": [
    {"excerpt": "<exact substring from transcript>", "explanation": "<short explanation>", "category": "vocabulary" | "grammar" | "humor" | "cultural_reference"}
  ]
}
Use exactly the keys "excerpt", "explanation", and "category" — no other key names.
"""


async def extract_transcript_insights(
    transcript: str,
) -> list[TranscriptInsight]:
    """
    Calls the mimo-v2.5 text model to extract humor, grammar, vocabulary,
    and cultural-reference insights from a transcript chunk, using MiMo's
    structured-output mode so the response is guaranteed to match
    TranscriptInsightsResult rather than relying on the model following
    JSON-formatting instructions in the prompt.
    """

    client = get_model()

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": TRANSCRIPT_INSIGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": transcript},
    ]

    try:
        completion = await client.chat.completions.create(
            model="mimo-v2.5",
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        raise MimoError(f"MiMo insight-extraction call failed: {e}") from e

    if not completion.choices:
        raise MimoError("MiMo insight-extraction response had no choices")

    raw_content = completion.choices[0].message.content
    if not raw_content:
        raise MimoError(
            "MiMo insight-extraction response had no message content"
        )

    try:
        content: dict = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise MimoError(
            f"MiMo insight-extraction response was not valid JSON: {e}. "
            f"Raw content (truncated): {raw_content[:300]!r}"
        ) from e

    insights = []
    for insight in content.get("insights", []):
        try:
            insights.append(TranscriptInsight.model_validate(insight))
        except ValidationError as e:
            logger.warning("Dropping malformed insight: %s (%s)", insight, e)

    return insights
