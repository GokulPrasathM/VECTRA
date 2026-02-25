from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RoleName = str


@dataclass(frozen=True)
class ChatMessage:
	role: Literal["system", "user", "assistant"]
	content: str


@dataclass(frozen=True)
class CandidateAnswer:
	answer: str
	solver_name: str
	rationale: str | None = None
	assumptions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JudgeVerdict:
	score: float  # 0..1
	confidence: float  # 0..1
	is_correct: bool | None
	notes: str


@dataclass(frozen=True)
class CriticVerdict:
	risk: float  # 0..1 higher = riskier
	notes: str


@dataclass(frozen=True)
class TraceEvent:
	type: str
	payload: dict[str, Any]


@dataclass(frozen=True)
class SolveResult:
	answer: str
	confidence: float
	calls_made: int
	trace: list[TraceEvent] | None = None


@dataclass(frozen=True)
class EvaluateReport:
	total: int
	correct: int
	accuracy: float
	avg_calls: float
	details: list[dict[str, Any]]


@dataclass(frozen=True)
class ScenarioAttemptSummary:
	attempt_id: int
	answer: str
	elapsed_s: float


@dataclass(frozen=True)
class ScenarioSolveResult:
	answer: str
	vote_counts: dict[str, int]
	attempts: list[ScenarioAttemptSummary]
	trace: list[TraceEvent] | None = None
