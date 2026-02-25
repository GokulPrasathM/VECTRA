from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .llm_client import OpenAICompatClient
from .types import ChatMessage


@dataclass
class CallResult:
    outputs: list[str]


class Executor:
    def __init__(self, client: OpenAICompatClient, *, max_concurrency: int = 6):
        self._client = client
        self._sem = asyncio.Semaphore(max_concurrency)
        self.calls_made = 0

    async def _call(self, messages: list[ChatMessage], **kwargs) -> CallResult:
        async with self._sem:
            self.calls_made += 1
            outputs = await self._client.chat(messages, **kwargs)
            return CallResult(outputs=outputs)

    async def call_many(self, reqs: list[tuple[list[ChatMessage], dict]]) -> list[CallResult]:
        tasks = [asyncio.create_task(self._call(msgs, **kwargs)) for msgs, kwargs in reqs]
        return await asyncio.gather(*tasks)
