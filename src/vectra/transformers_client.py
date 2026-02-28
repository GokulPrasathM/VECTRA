from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from collections.abc import Mapping
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

    # Optional throughput optimization: batch together concurrent .chat() calls.
    # This improves GPU utilization (especially on large VRAM GPUs) at the cost
    # of a tiny latency overhead (waiting for other requests).
    batch_max_size: int = 1
    batch_max_wait_ms: int = 8


@dataclass
class _ChatRequest:
    messages: list[ChatMessage]
    temperature: float
    max_new_tokens: int
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[str]


class TransformersClient:
    """Local Transformers chat client."""

    def __init__(self, config: TransformersClientConfig):
        self._cfg = config
        self._lock = threading.Lock()

        self._q: queue.Queue[_ChatRequest] | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

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

        # Start a batching worker if enabled.
        if int(self._cfg.batch_max_size) > 1:
            self._q = queue.Queue()
            self._worker = threading.Thread(target=self._batch_worker, daemon=True)
            self._worker.start()

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

        # If batching is enabled, enqueue and await a batched result.
        if self._q is not None:
            loop = asyncio.get_running_loop()
            fut: asyncio.Future[str] = loop.create_future()
            req = _ChatRequest(
                messages=messages,
                temperature=float(temperature),
                max_new_tokens=max_new_tokens,
                loop=loop,
                future=fut,
            )
            self._q.put(req)
            return [await fut]

        # Otherwise: synchronous single generation offloaded to a thread.
        return [
            await asyncio.to_thread(
                self._generate_one,
                messages,
                temperature=float(temperature),
                max_new_tokens=max_new_tokens,
            )
        ]

    def close(self) -> None:
        self._stop.set()

    def __del__(self) -> None:  # noqa: B027
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _messages_to_prompt_text(self, messages: list[ChatMessage]) -> str:
        try:
            apply = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply):
                chat = [{"role": m.role, "content": m.content} for m in messages]
                rendered = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
        except Exception:  # noqa: BLE001
            pass

        parts: list[str] = []
        for m in messages:
            parts.append(f"[{m.role.upper()}]\n{m.content}\n")
        parts.append("[ASSISTANT]\n")
        return "\n".join(parts)

    def _ensure_pad_token(self) -> None:
        # Many causal LM tokenizers don't define pad token. Use eos as pad.
        try:
            if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
                self._tokenizer, "eos_token_id", None
            ) is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception:  # noqa: BLE001
            pass

    def _generate_batch(
        self,
        reqs: list[_ChatRequest],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        self._ensure_pad_token()

        prompts = [self._messages_to_prompt_text(r.messages) for r in reqs]
        encoding = self._tokenizer(prompts, return_tensors="pt", padding=True)

        # Move tensors to the primary model device.
        try:
            encoding = encoding.to(self._model.device)
        except Exception:  # noqa: BLE001
            device = getattr(self._model, "device", None)
            if device is not None:
                for k, v in list(encoding.items()):
                    try:
                        encoding[k] = v.to(device)
                    except Exception:  # noqa: BLE001
                        pass

        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask")

        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(1e-3, float(temperature))

        if getattr(self._tokenizer, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(self._tokenizer.pad_token_id)
        elif getattr(self._tokenizer, "eos_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(self._tokenizer.eos_token_id)

        # Run one batched generate for throughput.
        with self._lock:
            try:
                import torch

                ctx = torch.inference_mode
            except Exception:  # noqa: BLE001
                ctx = None

            if ctx is None:
                out = self._model.generate(**encoding, **gen_kwargs)
            else:
                with ctx():
                    out = self._model.generate(**encoding, **gen_kwargs)

        if attention_mask is not None:
            prompt_lens = attention_mask.sum(dim=-1).tolist()
        else:
            # Fallback: assume right padding.
            prompt_lens = [int(input_ids.shape[-1])] * int(input_ids.shape[0])

        texts: list[str] = []
        for i, plen in enumerate(prompt_lens):
            gen = out[i, int(plen) :]
            text = self._tokenizer.decode(gen, skip_special_tokens=True)
            texts.append(str(text).strip())
        return texts

    def _batch_worker(self) -> None:
        assert self._q is not None

        batch_max_size = max(1, int(self._cfg.batch_max_size))
        batch_wait_s = max(0.0, float(self._cfg.batch_max_wait_ms) / 1000.0)

        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            reqs = [first]
            start = time.time()
            while len(reqs) < batch_max_size:
                remaining = batch_wait_s - (time.time() - start)
                if remaining <= 0:
                    break
                try:
                    reqs.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break

            # Group by generation params so we don't mix different temperatures/max_new_tokens.
            groups: dict[tuple[float, int], list[_ChatRequest]] = {}
            for r in reqs:
                if r.future.cancelled():
                    continue
                k = (float(r.temperature), int(r.max_new_tokens))
                groups.setdefault(k, []).append(r)

            for (temp, mx), group in groups.items():
                try:
                    outs = self._generate_batch(group, temperature=temp, max_new_tokens=mx)
                    for r, out in zip(group, outs, strict=False):
                        if r.future.cancelled():
                            continue
                        r.loop.call_soon_threadsafe(r.future.set_result, out)
                except Exception as e:  # noqa: BLE001
                    for r in group:
                        if r.future.cancelled():
                            continue
                        r.loop.call_soon_threadsafe(r.future.set_exception, e)

    def _messages_to_inputs(self, messages: list[ChatMessage]) -> Any:
        try:
            apply = getattr(self._tokenizer, "apply_chat_template", None)
            if callable(apply):
                chat = [{"role": m.role, "content": m.content} for m in messages]
                rendered = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                # Depending on tokenizer/transformers version, this can be:
                # - torch.Tensor
                # - BatchEncoding (dict-like)
                if hasattr(rendered, "shape"):
                    return {"input_ids": rendered}
                if isinstance(rendered, Mapping) and "input_ids" in rendered:
                    return rendered
        except Exception:  # noqa: BLE001
            pass

        parts: list[str] = []
        for m in messages:
            parts.append(f"[{m.role.upper()}]\n{m.content}\n")
        parts.append("[ASSISTANT]\n")
        text = "\n".join(parts)
        return self._tokenizer(text, return_tensors="pt")

    def _generate_one(self, messages: list[ChatMessage], *, temperature: float, max_new_tokens: int) -> str:
        encoding = self._messages_to_inputs(messages)

        if isinstance(encoding, Mapping) and "input_ids" in encoding:
            input_ids = encoding["input_ids"]
        else:
            raise TransformersBackendError(
                "Tokenizer did not produce input_ids for generation; check tokenizer.apply_chat_template behavior"
            )

        with self._lock:
            # Move tensors to the primary model device.
            try:
                encoding = encoding.to(self._model.device)
                input_ids = encoding["input_ids"]
            except Exception:  # noqa: BLE001
                device = getattr(self._model, "device", None)
                if device is not None:
                    for k, v in list(encoding.items()):
                        try:
                            encoding[k] = v.to(device)
                        except Exception:  # noqa: BLE001
                            pass
                    input_ids = encoding["input_ids"]

            do_sample = temperature > 0
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(1e-3, float(temperature))

            if getattr(self._tokenizer, "pad_token_id", None) is None and getattr(
                self._tokenizer, "eos_token_id", None
            ) is not None:
                gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id

            out = self._model.generate(**encoding, **gen_kwargs)

        prompt_len = int(input_ids.shape[-1])
        gen = out[0, prompt_len:]
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
