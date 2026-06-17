# Phy Backend

FastAPI service for processing short audio segments.

The API accepts a WAV audio chunk, transcribes it using MiMo ASR, and
extracts learner-focused language insights such as vocabulary, grammar,
humor, and cultural references.

## Flow

```text
Audio segment
    ↓
Speech-to-text
    ↓
Transcript
    ↓
Insight extraction
    ↓
Structured response
```

## Requirements

* Python 3.14+
* MiMo API key

## Setup

```bash
uv sync
```

Create a `.env` file:

```env
MIMO_API_KEY=your_api_key
```

## Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
