from __future__ import annotations

import json
from dataclasses import dataclass

from ..types import CandidateAnswer, ChatMessage, CriticVerdict


@dataclass(frozen=True)
class DefaultCritic:
    name: str = "critic"

    def build_messages(self, *, problem: str, candidate: CandidateAnswer, **kwargs) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=(
                    "You are an adversarial math critic. Your job is to find mistakes, missing cases, "
                    "or unjustified steps in a proposed solution/answer."
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Analyze the candidate answer for the problem. "
                    "Return STRICT JSON with keys: risk (0..1), notes.\n\n"
                    f"PROBLEM:\n{problem}\n\n"
                    f"CANDIDATE_FINAL_ANSWER:\n{candidate.answer}\n"
                ),
            ),
        ]

    def parse(self, text: str, **kwargs) -> CriticVerdict:
        try:
            obj = json.loads(text.strip())
            risk = float(obj.get("risk", 0.5))
            notes = str(obj.get("notes", ""))
        except Exception:  # noqa: BLE001
            risk = 0.6
            notes = text.strip()[:500]
        risk = max(0.0, min(1.0, risk))
        return CriticVerdict(risk=risk, notes=notes)
