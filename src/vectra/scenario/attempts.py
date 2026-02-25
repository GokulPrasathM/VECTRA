from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..types import ChatMessage
from ..tools.python_tool import PythonTool
from ..tools.tool_loop import ChatClient, ToolLoopConfig, ToolLoopRunner


@dataclass(frozen=True)
class AttemptConfig:
    attempts: int = 8
    early_stop: int = 4
    max_concurrency: int = 8
    time_budget_s: float | None = None
    tool_loop: ToolLoopConfig = ToolLoopConfig()


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: int
    answer: str
    transcript: list[ChatMessage]
    elapsed_s: float


@dataclass(frozen=True)
class AttemptReport:
    final_answer: str
    vote_counts: dict[str, int]
    results: list[AttemptResult]


def normalize_answer(answer: str) -> str:
    return " ".join((answer or "").strip().split())


def _best_by_votes(votes: dict[str, int]) -> str:
    if not votes:
        return ""
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


class ParallelAttemptRunner:
    def __init__(
        self,
        *,
        client: ChatClient,
        python_tool: PythonTool,
        config: AttemptConfig | None = None,
    ) -> None:
        self._client = client
        self._python = python_tool
        self._cfg = config or AttemptConfig()

    async def run(self, *, system_prompt: str, user_prompt: str) -> AttemptReport:
        sem = asyncio.Semaphore(self._cfg.max_concurrency)
        runner = ToolLoopRunner(client=self._client, python_tool=self._python, config=self._cfg.tool_loop)

        votes: dict[str, int] = {}
        results: list[AttemptResult] = []

        start = time.time()
        deadline = None
        if self._cfg.time_budget_s is not None:
            deadline = start + float(self._cfg.time_budget_s)

        async def one_attempt(attempt_id: int) -> AttemptResult:
            async with sem:
                t0 = time.time()
                messages = [
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_prompt),
                ]
                answer, transcript = await runner.run(messages)
                elapsed = time.time() - t0
                return AttemptResult(
                    attempt_id=attempt_id,
                    answer=normalize_answer(answer),
                    transcript=transcript,
                    elapsed_s=elapsed,
                )

        tasks = [asyncio.create_task(one_attempt(i + 1)) for i in range(self._cfg.attempts)]

        try:
            for fut in asyncio.as_completed(tasks):
                if deadline is not None and time.time() >= deadline:
                    break

                r = await fut
                results.append(r)
                if r.answer:
                    votes[r.answer] = votes.get(r.answer, 0) + 1
                    if votes[r.answer] >= self._cfg.early_stop:
                        break
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        final = _best_by_votes(votes)
        return AttemptReport(final_answer=final, vote_counts=votes, results=results)

