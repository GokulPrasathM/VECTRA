from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .state import AgentAction, AgentStatus, PhaseLabel, SystemState


class ScenarioAgent(Protocol):
    name: str
    allowed_phases: set[PhaseLabel]
    status: AgentStatus

    def step(self, problem: str, state: SystemState) -> list[AgentAction]:
        ...


class PhasePolicy(Protocol):
    def next_phase(self, state: SystemState) -> PhaseLabel | None:
        ...

    def can_collapse(self, state: SystemState) -> bool:
        ...


@dataclass(frozen=True)
class DefaultPhasePolicy:
    def next_phase(self, state: SystemState) -> PhaseLabel | None:  # noqa: ARG002
        return None

    def can_collapse(self, state: SystemState) -> bool:
        return False


class CognitiveReactor:
    def __init__(
        self,
        *,
        agents: Iterable[ScenarioAgent],
        policy: PhasePolicy | None = None,
        max_steps: int = 128,
    ) -> None:
        self._agents = list(agents)
        self._policy = policy or DefaultPhasePolicy()
        self._max_steps = int(max_steps)

    def run(self, problem: str, state: SystemState | None = None) -> SystemState:
        state = state or SystemState()
        if not state.active_agents:
            state.active_agents = {a.name for a in self._agents}

        for _ in range(self._max_steps):
            action_seen = False

            for agent in self._agents:
                if agent.status != AgentStatus.ACTIVE:
                    continue
                if not state.is_active(agent.name):
                    continue
                if state.current_phase not in agent.allowed_phases:
                    continue

                actions = agent.step(problem, state)
                if actions:
                    action_seen = True
                for action in actions:
                    state.action_log.append(action)
                    state.assumption_log.extend(action.assumptions)
                    for target in action.interrupts:
                        state.silenced_agents.add(target)
                        state.active_agents.discard(target)
                    for target in action.invalidates:
                        state.silenced_agents.add(target)
                        state.active_agents.discard(target)

            requested_phase = None
            for action in reversed(state.action_log):
                if action.request_phase is not None:
                    requested_phase = action.request_phase
                    break

            phase_changed = False
            if requested_phase is not None and requested_phase != state.current_phase:
                state.current_phase = requested_phase
                phase_changed = True
            else:
                proposed = self._policy.next_phase(state)
                if proposed is not None and proposed != state.current_phase:
                    state.current_phase = proposed
                    phase_changed = True

            if state.current_phase != PhaseLabel.COLLAPSE and self._policy.can_collapse(state):
                state.commitment_consumed = True
                state.current_phase = PhaseLabel.COLLAPSE
                phase_changed = True

            if state.current_phase == PhaseLabel.COLLAPSE:
                state.commitment_consumed = True
                break

            if not action_seen and not phase_changed:
                break

        return state

