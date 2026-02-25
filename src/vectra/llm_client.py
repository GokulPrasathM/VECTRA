from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .types import ChatMessage


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_key: str
    base_url: str
    model: str
    timeout_s: float = 60.0
    max_retries: int = 6


class OpenAICompatClient:
    def __init__(self, config: OpenAICompatConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.timeout_s),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )

    @classmethod
    def from_env(cls, *, default_model: str | None = None) -> "OpenAICompatClient":
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not api_key:
            raise LLMError("Missing OPENAI_API_KEY")

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        model = default_model or os.environ.get("OPENAI_MODEL")
        if not model:
            raise LLMError("Missing OPENAI_MODEL (or pass SolveConfig.model)")

        return cls(OpenAICompatConfig(api_key=api_key, base_url=base_url, model=model))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        n: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> list[str]:
        payload: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": float(temperature),
            "n": int(n),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if extra:
            payload.update(extra)

        last_err: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                resp = await self._client.post("/v1/chat/completions", content=json.dumps(payload))
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"Transient LLM error {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                outputs: list[str] = []
                for c in choices:
                    msg = c.get("message") or {}
                    outputs.append(str(msg.get("content", "")))
                if not outputs:
                    raise LLMError(f"No choices returned: {data}")
                return outputs
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= self._config.max_retries:
                    break

                sleep_s = min(10.0, (0.5 * (2**attempt)) + random.random() * 0.25)
                time.sleep(sleep_s)

        raise LLMError(f"LLM request failed after retries: {last_err}")
