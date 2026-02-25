from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from .llm_client import OpenAICompatClient, OpenAICompatConfig
from .solve import SolveConfig, _solve_with_client
from .types import EvaluateReport
from .vllm_server import VLLMServer


@dataclass(frozen=True)
class EvaluateConfig:
    solve: SolveConfig = SolveConfig()
    limit: int | None = None


def evaluate(dataset_path: str, config: EvaluateConfig) -> EvaluateReport:
    total = 0
    correct = 0
    calls_sum = 0
    details: list[dict] = []

    def run_with_client(client: OpenAICompatClient) -> None:
        nonlocal total, correct, calls_sum, details

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                problem = item["problem"]
                reference = item.get("reference")
                if reference is None:
                    raise ValueError("Dataset item missing 'reference'")

                result = _solve_with_client(problem, config.solve, client=client, reference=reference)
                total += 1
                calls_sum += result.calls_made

                is_correct = False
                if result.trace:
                    for ev in result.trace:
                        if ev.type == "final_verdict":
                            is_correct = bool(ev.payload.get("is_correct"))
                            break
                correct += 1 if is_correct else 0

                details.append(
                    {
                        "id": item.get("id"),
                        "calls": result.calls_made,
                        "confidence": result.confidence,
                        "answer": result.answer,
                        "is_correct": is_correct,
                    }
                )

                if config.limit is not None and total >= config.limit:
                    break

    if config.solve.vllm is None:
        client = OpenAICompatClient.from_env(default_model=config.solve.model)
        try:
            run_with_client(client)
        finally:
            asyncio.run(client.aclose())
    else:
        with VLLMServer(config.solve.vllm) as server:
            model = config.solve.model or config.solve.vllm.served_model_name
            client = OpenAICompatClient(
                OpenAICompatConfig(api_key="sk-local", base_url=server.base_url, model=model)
            )
            try:
                run_with_client(client)
            finally:
                asyncio.run(client.aclose())

    accuracy = (correct / total) if total else 0.0
    avg_calls = (calls_sum / total) if total else 0.0
    return EvaluateReport(
        total=total,
        correct=correct,
        accuracy=accuracy,
        avg_calls=avg_calls,
        details=details,
    )
