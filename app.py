import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    label: str


DEFAULT_MODEL_ID = "lukey03/Qwen3.5-9B-abliterated-MLX-8bit"
BASE_MODELS: List[ModelOption] = [
    ModelOption(
        model_id=DEFAULT_MODEL_ID,
        label="Qwen3.5-9B-abliterated-MLX-8bit (default)",
    ),
]


def get_model_options() -> List[ModelOption]:
    options = list(BASE_MODELS)
    extra_models = [x.strip() for x in os.getenv("EXTRA_MODELS", "").split(",") if x.strip()]
    for mid in extra_models:
        options.append(ModelOption(model_id=mid, label=mid))
    return options


MODEL_OPTIONS = get_model_options()
MODEL_MAP = {item.model_id: item for item in MODEL_OPTIONS}
MODEL_TYPE_ALIASES = {
    "qwen3_5_text": "qwen3_5",
}
MODEL_OVERRIDE_DIR = Path(__file__).parent / ".model_overrides"

_loaded_models: Dict[str, Tuple[object, object]] = {}
_model_lock = threading.Lock()
_generation_lock = threading.Lock()
_generation_state_lock = threading.Lock()
_generation_stop_event = threading.Event()
_generation_active = False
_mlx_api: Dict[str, object] = {}


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL_ID)
    message: str = Field(min_length=1)
    system_prompt: str = Field(default="")
    response_language: Literal["auto", "zh", "en", "ja", "ko"] = "auto"
    enable_thinking: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=5000)
    repetition_penalty: float = Field(default=1.08, ge=0.0, le=2.0)
    repetition_context_size: int = Field(default=128, ge=0, le=4096)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    history: List[ChatTurn] = Field(default_factory=list)


LANGUAGE_RULES = {
    "zh": "You must reply in Simplified Chinese.",
    "en": "You must reply in English.",
    "ja": "You must reply in Japanese.",
    "ko": "You must reply in Korean.",
}


def build_system_instruction(system_prompt: str, response_language: str) -> str:
    parts = []
    if system_prompt.strip():
        parts.append(system_prompt.strip())

    language_rule = LANGUAGE_RULES.get(response_language)
    if language_rule:
        parts.append(
            f"{language_rule} Keep terminology accurate, and do not switch language unless explicitly requested."
        )
    return "\n\n".join(parts).strip()


def build_prompt(
    tokenizer: object,
    system_prompt: str,
    response_language: str,
    enable_thinking: bool,
    history: List[ChatTurn],
    message: str,
) -> str:
    messages = []
    system_instruction = build_system_instruction(system_prompt, response_language)
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content.strip()})
    messages.append({"role": "user", "content": message.strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    fallback_parts = []
    if system_instruction:
        fallback_parts.append(f"System: {system_instruction}")
    for turn in history:
        prefix = "User" if turn.role == "user" else "Assistant"
        fallback_parts.append(f"{prefix}: {turn.content.strip()}")
    fallback_parts.append(f"User: {message.strip()}")
    fallback_parts.append("Assistant:")
    return "\n".join(fallback_parts)


def get_or_load_model(model_id: str) -> Tuple[object, object]:
    if model_id not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model_id}")

    if model_id in _loaded_models:
        return _loaded_models[model_id]

    mlx_load = get_mlx_api()["load"]

    with _model_lock:
        if model_id in _loaded_models:
            return _loaded_models[model_id]
        try:
            model, tokenizer = mlx_load(model_id)
        except ValueError as exc:
            patched_path = build_compat_model_override(model_id, str(exc))
            if not patched_path:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            try:
                model, tokenizer = mlx_load(patched_path)
            except Exception as patch_exc:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Model load failed after compatibility patch. "
                        f"Original error: {exc}; patched error: {patch_exc}"
                    ),
                ) from patch_exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc
        _loaded_models[model_id] = (model, tokenizer)
        return model, tokenizer


def as_text(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    return str(chunk)


def get_mlx_api() -> Dict[str, object]:
    if _mlx_api:
        return _mlx_api

    # Delay importing mlx_lm until first generation request.
    from mlx_lm import load as mlx_load  # type: ignore
    from mlx_lm import stream_generate as mlx_stream_generate  # type: ignore
    from mlx_lm.sample_utils import make_logits_processors as mlx_make_logits_processors  # type: ignore
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler  # type: ignore

    _mlx_api["load"] = mlx_load
    _mlx_api["stream_generate"] = mlx_stream_generate
    _mlx_api["make_sampler"] = mlx_make_sampler
    _mlx_api["make_logits_processors"] = mlx_make_logits_processors
    return _mlx_api


def set_generation_active(active: bool) -> None:
    global _generation_active
    with _generation_state_lock:
        _generation_active = active
        if active:
            _generation_stop_event.clear()
        else:
            _generation_stop_event.clear()


def request_generation_stop() -> bool:
    with _generation_state_lock:
        if not _generation_active:
            return False
        _generation_stop_event.set()
        return True


def is_generation_stopped() -> bool:
    return _generation_stop_event.is_set()


def iter_generation_deltas(
    mlx_stream_generate: object,
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int,
    sampler: object,
    logits_processors: object,
):
    last_seen = ""
    for piece in mlx_stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        if is_generation_stopped():
            break
        token_text = as_text(piece)
        if token_text.startswith(last_seen):
            delta = token_text[len(last_seen) :]
            last_seen = token_text
        else:
            delta = token_text
            last_seen += token_text
        if not delta:
            continue
        yield delta


def build_compat_model_override(model_id: str, error_message: str) -> str:
    unsupported_type = extract_unsupported_model_type(error_message)
    aliased_type = MODEL_TYPE_ALIASES.get(unsupported_type)
    if not aliased_type:
        return ""

    # Delay importing huggingface_hub until we actually need a compatibility patch.
    from huggingface_hub import snapshot_download  # type: ignore

    snapshot_dir = Path(snapshot_download(repo_id=model_id))
    source_config = snapshot_dir / "config.json"
    if not source_config.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model config not found at {source_config}",
        )

    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["model_type"] = aliased_type

    patched_dir = MODEL_OVERRIDE_DIR / model_id.replace("/", "__") / snapshot_dir.name
    patched_dir.mkdir(parents=True, exist_ok=True)

    for item in snapshot_dir.iterdir():
        if item.name == "config.json":
            continue
        target = patched_dir / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item)

    patched_config_path = patched_dir / "config.json"
    if patched_config_path.exists() or patched_config_path.is_symlink():
        patched_config_path.unlink()

    patched_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return str(patched_dir)


def extract_unsupported_model_type(error_message: str) -> str:
    match = re.search(r"Model type ([a-zA-Z0-9_]+) not supported", error_message)
    if not match:
        return ""
    return match.group(1)


def create_app() -> FastAPI:
    app = FastAPI(title="Local MLX Chat")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/models")
    def models() -> JSONResponse:
        return JSONResponse(
            {
                "models": [
                    {"id": item.model_id, "label": item.label} for item in MODEL_OPTIONS
                ]
            }
        )

    @app.post("/api/stop")
    def stop_generation() -> JSONResponse:
        stopped = request_generation_stop()
        return JSONResponse({"stopped": stopped})

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> JSONResponse:
        mlx_stream_generate = get_mlx_api()["stream_generate"]
        mlx_make_sampler = get_mlx_api()["make_sampler"]
        mlx_make_logits_processors = get_mlx_api()["make_logits_processors"]
        model, tokenizer = get_or_load_model(req.model)
        prompt = build_prompt(
            tokenizer,
            req.system_prompt,
            req.response_language,
            req.enable_thinking,
            req.history,
            req.message,
        )
        sampler = mlx_make_sampler(
            temp=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        logits_processors = mlx_make_logits_processors(
            repetition_penalty=req.repetition_penalty,
            repetition_context_size=req.repetition_context_size,
        )

        full_text = ""
        stopped = False
        with _generation_lock:
            set_generation_active(True)
            try:
                for delta in iter_generation_deltas(
                    mlx_stream_generate=mlx_stream_generate,
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                ):
                    full_text += delta
                stopped = is_generation_stopped()
            finally:
                set_generation_active(False)
        return JSONResponse({"reply": full_text, "stopped": stopped})

    @app.post("/api/chat/stream")
    def chat_stream(req: ChatRequest) -> StreamingResponse:
        mlx_stream_generate = get_mlx_api()["stream_generate"]
        mlx_make_sampler = get_mlx_api()["make_sampler"]
        mlx_make_logits_processors = get_mlx_api()["make_logits_processors"]
        model, tokenizer = get_or_load_model(req.model)
        prompt = build_prompt(
            tokenizer,
            req.system_prompt,
            req.response_language,
            req.enable_thinking,
            req.history,
            req.message,
        )
        sampler = mlx_make_sampler(
            temp=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
        logits_processors = mlx_make_logits_processors(
            repetition_penalty=req.repetition_penalty,
            repetition_context_size=req.repetition_context_size,
        )

        def event(event_name: str, payload: dict) -> bytes:
            data = json.dumps(payload, ensure_ascii=False)
            return f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")

        def stream():
            try:
                full_text = ""
                with _generation_lock:
                    set_generation_active(True)
                    try:
                        for delta in iter_generation_deltas(
                            mlx_stream_generate=mlx_stream_generate,
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            max_tokens=req.max_tokens,
                            sampler=sampler,
                            logits_processors=logits_processors,
                        ):
                            full_text += delta
                            yield event("token", {"text": delta})
                    finally:
                        stopped = is_generation_stopped()
                        set_generation_active(False)
                if stopped:
                    yield event("stopped", {"text": full_text})
                else:
                    yield event("done", {"text": full_text})
            except Exception as exc:  # pragma: no cover
                yield event("error", {"message": str(exc)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
