#!/usr/bin/env python3
"""Serve the custom HF CSV-memory model through an OpenAI-compatible API."""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import AutoModel

MODEL_DIR = os.environ.get("MODEL_DIR", "./local-model")
MODEL_NAME = os.environ.get("MODEL_NAME", "local-model")
API_KEY = os.environ.get("API_KEY", "local-eval-key")
CHOICE_FORMAT = os.environ.get("CHOICE_FORMAT", "answer")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

if CHOICE_FORMAT not in {"answer", "plain", "paren"}:
    raise ValueError("CHOICE_FORMAT must be answer, plain, or paren")

# Load the exported mapping-backed custom model on CPU.
model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
model.to("cpu")
model.eval()

app = FastAPI(title="CSV Memory OpenAI-Compatible API", version="1.0.0")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "\n".join(chunks)
    return ""


def _extract_prompt(payload: dict) -> str:
    messages = payload.get("messages") or []
    user_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            content = _content_to_text(message.get("content"))
            if content:
                user_parts.append(content)
    if user_parts:
        return "\n".join(user_parts)
    # Compatibility fallback for simple clients that omit the role field.
    return "\n".join(
        _content_to_text(m.get("content")) for m in messages if isinstance(m, dict)
    ).strip()


def _authorize(request: Request) -> None:
    if not API_KEY:
        return
    authorization = request.headers.get("authorization", "")
    provided = authorization.removeprefix("Bearer ").strip()
    if provided != API_KEY and request.headers.get("api-key", "") != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_dir": os.path.abspath(MODEL_DIR),
        "records": len(model.memory.records),
        "lookup_stats": dict(model.memory.stats),
    }


@app.get("/v1/models")
def list_models(request: Request):
    _authorize(request)
    return {"object": "list", "data": [{
        "id": MODEL_NAME,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _authorize(request)
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    prompt = _extract_prompt(payload)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="No user prompt found in messages")

    answer = model.answer_text(prompt, choice_format=CHOICE_FORMAT)
    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(answer.split()))
    response = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
