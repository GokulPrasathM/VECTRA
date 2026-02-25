from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from vectra.types import ChatMessage
from vectra.scenario.reactor import CognitiveReactor
from vectra.scenario.state import AgentAction, AgentStatus, PhaseLabel, SystemState
from vectra.scenario.attempts import AttemptConfig, ParallelAttemptRunner
from vectra.tools.tool_loop import ToolLoopConfig, ToolLoopRunner, parse_tool_request
from vectra import ScenarioSolveConfig, solve_scenario_async
from vectra.types import ChatMessage


def test_parse_tool_request_json() -> None:
    req = parse_tool_request('{"tool":"python","code":"print(2+2)"}')
    assert req is not None
    assert req.tool == "python"
    assert "2+2" in req.code


def test_parse_tool_request_rejects_non_python() -> None:
    assert parse_tool_request('{"tool":"bash","code":"echo hi"}') is None


@dataclass
class FakePythonTool:
    output: str = "4"

    def execute(self, code: str, *, timeout_s: float | None = None) -> str:  # noqa: ARG002
        return self.output


class FakeClient:
    """A minimal stand-in for OpenAICompatClient used in unit tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: list[ChatMessage], **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        last = messages[-1].content
        if "PYTHON_OUTPUT" in last:
            return ["FINAL: 4"]
        return ['{"tool":"python","code":"print(2+2)"}']


@pytest.mark.asyncio
async def test_tool_loop_calls_python_then_finishes() -> None:
    runner = ToolLoopRunner(
        client=FakeClient(),
        python_tool=FakePythonTool(),
        config=ToolLoopConfig(max_turns=4),
    )
    answer, transcript = await runner.run(
        [
            ChatMessage(role="system", content="Use tools"),
            ChatMessage(role="user", content="Compute 2+2"),
        ]
    )
    assert answer == "4"
    assert any("PYTHON_OUTPUT" in m.content for m in transcript)

def test_solve_scenario_uses_provided_client() -> None:
    class FakeClient:
        async def chat(
            self,
            messages: list[ChatMessage],
            *,
            model: str | None = None,
            temperature: float = 0.7,
            max_tokens: int | None = None,
            n: int = 1,
            extra: dict | None = None,
        ) -> list[str]:
            # Ensure the tool loop can terminate immediately.
            return ["FINAL: 123"]

    cfg = ScenarioSolveConfig(
        transformers_client=FakeClient(),
        attempts=AttemptConfig(
            attempts=1,
            early_stop=1,
            max_concurrency=1,
            tool_loop=ToolLoopConfig(max_turns=2, temperature=0.0),
        ),
    )

    import asyncio

    res = asyncio.run(solve_scenario_async("What is 1+2?", cfg))
    assert res.answer == "123"


@dataclass
class CountingAgent:
    name: str
    allowed_phases: set[PhaseLabel]
    status: AgentStatus = AgentStatus.ACTIVE
    calls: int = 0

    def step(self, problem: str, state: SystemState):  # noqa: ANN001, ARG002
        self.calls += 1
        return []


def test_reactor_gates_agents_by_phase() -> None:
    a1 = CountingAgent(name="a1", allowed_phases={PhaseLabel.ORIENTATION})
    a2 = CountingAgent(name="a2", allowed_phases={PhaseLabel.EXPLORATION})
    state = SystemState(current_phase=PhaseLabel.ORIENTATION)

    out = CognitiveReactor(agents=[a1, a2], max_steps=5).run("p", state=state)
    assert out.current_phase == PhaseLabel.ORIENTATION
    assert a1.calls == 1
    assert a2.calls == 0


@pytest.mark.asyncio
async def test_parallel_attempts_early_stop() -> None:
    client = FakeClient()
    python_tool = FakePythonTool()
    cfg = AttemptConfig(attempts=6, early_stop=2, max_concurrency=3, tool_loop=ToolLoopConfig(max_turns=2))
    runner = ParallelAttemptRunner(client=client, python_tool=python_tool, config=cfg)

    rep = await runner.run(system_prompt="Use tool protocol", user_prompt="Compute 2+2")
    assert rep.final_answer == "4"
    assert rep.vote_counts.get("4", 0) >= 2


def test_asyncio_available() -> None:
    # Sanity: ensure event loop can be created in this environment.
    asyncio.new_event_loop().close()
