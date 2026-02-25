from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PhaseLabel(str, Enum):
    ORIENTATION = "orientation"
    EXPLORATION = "exploration"
    COLLAPSE = "collapse"


class AgentStatus(str, Enum):
    ACTIVE = "active"
    SILENCED = "silenced"
    DEAD = "dead"


class TensionLabel(str, Enum):
    DIFFUSE = "diffuse"
    FRACTURING = "fracturing"
    STALE = "stale"
    CRITICAL = "critical"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AssumptionRecord:
    agent_name: str
    text: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentAction:
    actor: str
    kind: str
    assumptions: list[AssumptionRecord] = field(default_factory=list)
    interrupts: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    request_phase: PhaseLabel | None = None


@dataclass
class SystemState:
    current_phase: PhaseLabel = PhaseLabel.ORIENTATION
    commitment_consumed: bool = False
    tension: TensionLabel = TensionLabel.AMBIGUOUS

    active_agents: set[str] = field(default_factory=set)
    dead_agents: set[str] = field(default_factory=set)
    silenced_agents: set[str] = field(default_factory=set)

    assumption_log: list[AssumptionRecord] = field(default_factory=list)
    action_log: list[AgentAction] = field(default_factory=list)

    def is_active(self, name: str) -> bool:
        return (
            name in self.active_agents
            and name not in self.dead_agents
            and name not in self.silenced_agents
        )

