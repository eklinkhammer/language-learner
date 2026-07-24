# Language Learner

Local-first language learning app with pronunciation correction. Runs entirely on your machine using MLX Whisper for transcription, wav2vec2 for phoneme alignment, and `claude -p` for conversational tutoring.

Target languages: Spanish (MVP), Croatian, German, Mandarin.

## Requirements

- macOS with Apple Silicon (M-series)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (for tutor mode)

## Setup

```bash
scripts/setup.sh
```

## Development

```bash
scripts/dev.sh
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API docs: http://localhost:8000/docs

## Modes

- **Conversation Tutor** — Free-form language practice with an AI tutor
- **Pronunciation Drills** — Record yourself saying phrases and get phoneme-level feedback
