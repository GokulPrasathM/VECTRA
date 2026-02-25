from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .llm_client import OpenAICompatClient, OpenAICompatConfig
from .scenario.attempts import AttemptConfig, ParallelAttemptRunner
from .scenario.reactor import CognitiveReactor
from .tools.python_tool import SubprocessPythonTool
from .tools.tool_loop import ChatClient, ToolLoopConfig
from .transformers_client import TransformersClient, TransformersClientConfig
from .types import ScenarioAttemptSummary, ScenarioSolveResult, TraceEvent
from .vllm_server import VLLMServer, VLLMServerConfig


DEFAULT_TOOL_PROTOCOL = (
    "You may use an external Python tool. "
    'To call it, output ONLY a JSON object like: {"tool":"python","code":"print(2+2)"}. '
    "To finish, output: FINAL: <your answer>."
)


@dataclass(frozen=True)
class ScenarioSolveConfig:
    model: str | None = None
    vllm: VLLMServerConfig | None = None

    # If provided, this client is used directly (e.g. a preloaded TransformersClient).
    # This avoids loading large local models multiple times.
    transformers_client: ChatClient | None = None
    transformers_model_id: str | None = None
    transformers_max_new_tokens: int = 256

    system_prompt: str = DEFAULT_TOOL_PROTOCOL

    attempts: AttemptConfig = AttemptConfig()

    # Tool loop defaults; if you set this, it overrides AttemptConfig.tool_loop.
    tool_loop: ToolLoopConfig | None = None

    # Optional scenario gating pre-pass. If provided, its assumptions are injected into the prompt.
    reactor: CognitiveReactor | None = None

    return_trace: bool = False


def _inject_assumptions(system_prompt: str, assumptions: list[str]) -> str:
    if not assumptions:
        return system_prompt
    joined = "\n".join(f"- {a}" for a in assumptions)
    return (
        system_prompt
        + "\n\n"
        + "Assumptions/notes from scenario controller (not final answers):\n"
        + joined
    )


def solve_scenario(problem: str, config: ScenarioSolveConfig) -> ScenarioSolveResult:
    return asyncio.run(solve_scenario_async(problem, config))


async def solve_scenario_async(problem: str, config: ScenarioSolveConfig) -> ScenarioSolveResult:
    tool_loop_cfg = config.tool_loop or config.attempts.tool_loop
    if tool_loop_cfg.model is None and config.model is not None:
        tool_loop_cfg = ToolLoopConfig(
            max_turns=tool_loop_cfg.max_turns,
            temperature=tool_loop_cfg.temperature,
            model=config.model,
            python_timeout_s=tool_loop_cfg.python_timeout_s,
        )
    attempts_cfg = AttemptConfig(
        attempts=config.attempts.attempts,
        early_stop=config.attempts.early_stop,
        max_concurrency=config.attempts.max_concurrency,
        time_budget_s=config.attempts.time_budget_s,
        tool_loop=tool_loop_cfg,
    )

    python_tool = SubprocessPythonTool(timeout_s=attempts_cfg.tool_loop.python_timeout_s)

    system_prompt = config.system_prompt
    trace: list[TraceEvent] = []

    if config.reactor is not None:
        state = config.reactor.run(problem)
        assumptions = [a.text for a in state.assumption_log]
        system_prompt = _inject_assumptions(system_prompt, assumptions)
        if config.return_trace:
            trace.append(
                TraceEvent(
                    type="scenario_state",
                    payload={
                        "phase": state.current_phase.value,
                        "commitment_consumed": state.commitment_consumed,
                        "assumptions": assumptions,
                    },
                )
            )

    async def run_with_client(client: ChatClient) -> ScenarioSolveResult:
        runner = ParallelAttemptRunner(client=client, python_tool=python_tool, config=attempts_cfg)
        report = await runner.run(system_prompt=system_prompt, user_prompt=problem)
        summaries = [
            ScenarioAttemptSummary(
                attempt_id=r.attempt_id,
                answer=r.answer,
                elapsed_s=r.elapsed_s,
            )
            for r in report.results
        ]
        out_trace = trace if (config.return_trace and trace) else None
        if config.return_trace:
            out_trace = list(out_trace or [])
            out_trace.append(
                TraceEvent(
                    type="attempts_summary",
                    payload={
                        "vote_counts": report.vote_counts,
                        "attempts": [s.__dict__ for s in summaries],
                    },
                )
            )
        return ScenarioSolveResult(
            answer=report.final_answer,
            vote_counts=report.vote_counts,
            attempts=summaries,
            trace=out_trace,
        )

    if config.transformers_client is not None:
        return await run_with_client(config.transformers_client)

    if config.transformers_model_id is not None:
        client = TransformersClient(
            TransformersClientConfig(
                model_id=config.transformers_model_id,
                max_new_tokens=int(config.transformers_max_new_tokens),
            )
        )
        return await run_with_client(client)

    if config.vllm is None:
        client = OpenAICompatClient.from_env(default_model=config.model)
        try:
            return await run_with_client(client)
        finally:
            await client.aclose()

    with VLLMServer(config.vllm) as server:
        model = config.model or config.vllm.served_model_name
        client = OpenAICompatClient(
            OpenAICompatConfig(api_key="sk-local", base_url=server.base_url, model=model)
        )
        try:
            return await run_with_client(client)
        finally:
            await client.aclose()
