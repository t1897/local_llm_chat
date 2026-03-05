# Local MLX Chat

[中文文档 (Chinese README)](README.zh-CN.md)

Local MLX Chat is a lightweight web app for **fully local LLM conversation** on Apple Silicon.

It is designed to keep setup simple while still supporting practical chat features:

- Local-only model inference with `mlx-lm`
- Multi-turn context-aware conversation
- Streaming and non-streaming generation
- Thought section and final answer section rendering
- Thread history (new/switch/delete)
- Edit-and-regenerate from a previous user turn
- Tunable generation parameters

Default model:

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

## What This Project Is

This is a local chat stack:

- Backend: FastAPI
- Inference: `mlx-lm` (`load`, `stream_generate`, sampler/logits processors)
- Frontend: single-page HTML/CSS/JS (no heavy framework)

Important:

- Inference runs on your local machine (Apple Silicon).
- The model is downloaded from Hugging Face on first use.
- After download, model files are reused from local cache.
- This project does not proxy requests to cloud LLM APIs.

## Features

- Multi-turn chat with history passed to the model each turn
- `Send`, `Stop`, and `New Chat`
- Session thread list with switch/delete
- User-message `Edit & Regenerate`
- Optional streaming output (SSE)
- Optional thinking mode (model/template dependent)
- Language forcing (`Auto`, `中文`, `English`, `日本語`, `한국어`)
- Persistent settings in browser storage

## Requirements

- macOS (Apple Silicon)
- Python 3.10+

## Quick Start

```bash
cd <your-project-directory>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Model Configuration

Default model id:

- `lukey03/Qwen3.5-9B-abliterated-MLX-8bit`

You can append additional models to the dropdown by environment variable:

```bash
EXTRA_MODELS="mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/Llama-3.2-3B-Instruct-4bit" \
uvicorn app:app --host 127.0.0.1 --port 8000
```

## Generation Parameters

- `Temperature`: higher = more diverse, lower = more deterministic
- `Top P`: nucleus sampling probability mass
- `Top K`: sample only from top K candidate tokens
- `Repetition Penalty`: discourages repetition
- `Repeat Context`: token window used by repetition penalty
- `Max Tokens`: max newly generated tokens per response (backend cap: `4096`)
- `Streaming`: stream token deltas or wait for full response
- `Thinking`: enables thinking mode in chat template (if model supports it)
- `System Prompt`: global behavior instruction
- `Response Language`: language forcing rule

Note: `Max Tokens` is generation length per reply, not total context window.

## Session and Persistence

- Threads and messages are stored in `sessionStorage`
- Refreshing the tab keeps the current browser-session history
- Closing the browser clears session chat history
- Settings are stored in `localStorage` (persist across reopen)

## Project Structure

- `app.py`: API routes, model loading, prompt building, generation loop
- `static/index.html`: UI layout, chat rendering, history management
- `requirements.txt`: Python dependencies
- `.model_overrides/`: runtime compatibility overrides for model config

## FAQ

### Why is the first response slow?

First request may download and initialize the model. Later requests are faster.

### How do I remove local model cache?

Model files are managed by `huggingface_hub` cache.  
You can inspect cache root with:

```bash
python -c "from huggingface_hub import scan_cache_dir; print(scan_cache_dir().cache_dir)"
```

Then remove the target repo cache directory if needed.

### Why is `Max Tokens` limited to 4096 here?

This app intentionally caps generation length for stability and predictable latency.
