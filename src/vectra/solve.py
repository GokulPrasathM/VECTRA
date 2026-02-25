from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .controller import Controller
from .executor import Executor
from .llm_client import OpenAICompatClient, OpenAICompatConfig
from .policies.base import ControllerPolicy
from .roles.critic import DefaultCritic
from .roles.judge import DefaultJudge
from .roles.solver_variants import DefaultSolverVariants
from .types import SolveResult
from .vllm_server import VLLMServer, VLLMServerConfig


@dataclass(frozen=True)
class SolveConfig:
    model: str | None = None
    max_rounds: int = 2
    max_calls: int = 10
    max_concurrency: int = 6
    temperature: float = 0.7

    vllm: VLLMServerConfig | None = None

    accept_confidence: float = 0.75
    accept_margin: float = 0.15

    return_trace: bool = False

    policy: ControllerPolicy | None = None

    solver_roles: list = field(default_factory=list)
    critic_role: object | None = None
    judge_role: object | None = None


def _solve_with_client(
    problem: str,
    config: SolveConfig,
    *,
    client: OpenAICompatClient,
    reference: str | None = None,
) -> SolveResult:
    executor = Executor(client, max_concurrency=config.max_concurrency)

    solver_roles = config.solver_roles or DefaultSolverVariants().roles
    critic_role = config.critic_role or DefaultCritic()
    judge_role = config.judge_role or DefaultJudge()

    controller = Controller(
        executor=executor,
        solver_roles=solver_roles,
        critic_role=critic_role,
        judge_role=judge_role,
        policy=config.policy,
    )

    return controller.run(
        problem=problem,
        reference=reference,
        max_rounds=config.max_rounds,
        max_calls=config.max_calls,
        temperature=config.temperature,
        accept_confidence=config.accept_confidence,
        accept_margin=config.accept_margin,
        return_trace=config.return_trace,
    )


def solve(problem: str, config: SolveConfig, *, reference: str | None = None) -> SolveResult:
    if config.vllm is None:
        client = OpenAICompatClient.from_env(default_model=config.model)
        try:
            return _solve_with_client(problem, config, client=client, reference=reference)
        finally:
            asyncio.run(client.aclose())

    with VLLMServer(config.vllm) as server:
        model = config.model or config.vllm.served_model_name
        client = OpenAICompatClient(
            OpenAICompatConfig(
                api_key="sk-local",
                base_url=server.base_url,
                model=model,
            )
        )
        try:
            return _solve_with_client(problem, config, client=client, reference=reference)
        finally:
            asyncio.run(client.aclose())
