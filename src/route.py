import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.mimo_client import (
    MimoError,
    extract_transcript_insights,
    transcribe_audio,
)
from src.schema import TranscriptSegmentResponse

logger = logging.getLogger("phy.transcribe")

router = APIRouter()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


@router.post("/segments/process", response_model=TranscriptSegmentResponse)
async def process_segments(
    start_seconds: Annotated[float, Form()],
    end_seconds: Annotated[float, Form()],
    audio: Annotated[UploadFile, File()],
):
    if audio.content_type not in ("audio/wav", "audio/x-wav", "audio/wave"):
        msg = f"Unexpected content type {audio.content_type}"
        logger.warning(msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio chunk received",
        )
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Chunk is {len(audio_bytes) / (1024 * 1024):.1f}MB raw, "
                "too large once base64-encoded for MiMo's ASR limit. "
                "Reduce MAX_CHUNK_SECONDS or sample rate on the frontend."
            ),
        )

    logger.info(
        "Transcribing chunk window=%.1f-%.1fs size=%dKB",
        start_seconds,
        end_seconds,
        len(audio_bytes) // 1024,
    )

    try:
        transcript = await transcribe_audio(audio_bytes)
    except MimoError as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e

    try:
        insights = await extract_transcript_insights(transcript)
    except MimoError as e:
        logger.warning("Insight extraction failed: %s", e)
        insights = []

    return TranscriptSegmentResponse(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        transcript=transcript,
        insights=insights,
    )
