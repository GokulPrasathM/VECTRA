from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..types import CandidateAnswer, CriticVerdict, JudgeVerdict


@dataclass(frozen=True)
class ControllerDecision:
    accept: bool
    selected_index: int
    confidence: float
    notes: str


class ControllerPolicy(Protocol):
    def decide(
        self,
        *,
        candidates: list[CandidateAnswer],
        judge: list[JudgeVerdict],
        critic: list[CriticVerdict] | None,
        accept_confidence: float,
        accept_margin: float,
    ) -> ControllerDecision:
        ...
