from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

from .types import ChatMessage


class TransformersBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransformersClientConfig:
    model_id: str
    max_new_tokens: int = 256
    device_map: str | dict[str, Any] = "auto"
    torch_dtype: str | Any = "auto"
    trust_remote_code: bool = False


class TransformersClient:
    """Local Transformers chat client."""

    def __init__(self, config: TransformersClientConfig):
        self._cfg = config
        self._lock = threading.Lock()

        global _GLOBAL_BACKEND_CACHE  # noqa: PLW0603

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise TransformersBackendError(
                "Transformers backend requires `transformers` and `torch` installed"
            ) from e

        key = _cache_key(config)
        with _GLOBAL_BACKEND_CACHE_LOCK:
            cached = _GLOBAL_BACKEND_CACHE.get(key)
            if cached is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_id, trust_remote_code=config.trust_remote_code
                )
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_id,
                    device_map=config.device_map,
                    torch_dtype=config.torch_dtype,
                    trust_remote_code=config.trust_remote_code,
                )

                try:
                    model.eval()
                except Exception:  # noqa: BLE001
                    pass

                _GLOBAL_BACKEND_CACHE[key] = (tokenizer, model)
                cached = (tokenizer, model)

        self._tokenizer, self._model = cached

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,  # noqa: ARG002
        temperature: float = 0.7,
        max_tokens: int | None = None,
        n: int = 1,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> list[str]:
        if n != 1:
            raise TransformersBackendError("TransformersClient only supports n=1")

        max_new_tokens = int(max_tokens) if max_tokens is not None else int(self._cfg.max_new_tokens)

        import asyncio

        return [
            await asyncio.to_thread(
                self._generate_one,
                messages,
                temperature=float(temperature),
                max_new_tokens=max_new_tokens,
            )
        ]

    def _messages_to_inputs(self, messages: list[ChatMessage]):
        try:
            apply = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply):
                chat = [{"role": m.role, "content": m.content} for m in messages]
                return self._tokenizer.apply_chat_template(
                    chat,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
        except Exception:  # noqa: BLE001
            pass

        parts: list[str] = []
        for m in messages:
            parts.append(f"[{m.role.upper()}]\n{m.content}\n")
        parts.append("[ASSISTANT]\n")
        text = "\n".join(parts)
        return self._tokenizer(text, return_tensors="pt").input_ids

    def _generate_one(self, messages: list[ChatMessage], *, temperature: float, max_new_tokens: int) -> str:
        input_ids = self._messages_to_inputs(messages)

        with self._lock:
            input_ids = input_ids.to(self._model.device)

            do_sample = temperature > 0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(1e-3, float(temperature))

            out = self._model.generate(input_ids=input_ids, **gen_kwargs)

        gen = out[0, input_ids.shape[-1] :]
        text = self._tokenizer.decode(gen, skip_special_tokens=True)
        return str(text).strip()


_GLOBAL_BACKEND_CACHE_LOCK = threading.Lock()
_GLOBAL_BACKEND_CACHE: dict[str, tuple[Any, Any]] = {}


def _cache_key(config: TransformersClientConfig) -> str:
    device_map = config.device_map
    if isinstance(device_map, dict):
        try:
            device_map_key = json.dumps(device_map, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            device_map_key = repr(sorted(device_map.items(), key=lambda kv: str(kv[0])))
    else:
        device_map_key = str(device_map)

    payload = {
        "model_id": config.model_id,
        "device_map": device_map_key,
        "torch_dtype": str(config.torch_dtype),
        "trust_remote_code": bool(config.trust_remote_code),
    }
    return json.dumps(payload, sort_keys=True)
