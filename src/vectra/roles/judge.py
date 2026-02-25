from __future__ import annotations

import json
from dataclasses import dataclass

from ..types import CandidateAnswer, ChatMessage, CriticVerdict, JudgeVerdict


@dataclass(frozen=True)
class DefaultJudge:
    name: str = "judge"

    def build_messages(
        self,
        *,
        problem: str,
        candidate: CandidateAnswer,
        critic: CriticVerdict,
        reference: str | None,
        **kwargs,
    ) -> list[ChatMessage]:
        judge_instructions = (
            "You are a strict grader for mathematical correctness. "
            "Grade the candidate's final answer for the given problem. "
            "Use the critic notes as additional risk information. "
            "Return STRICT JSON with keys: score (0..1), confidence (0..1), is_correct, notes."
        )
        if reference is None:
            ref_block = "REFERENCE: (none provided)\nYou must estimate correctness based on the problem statement alone."
        else:
            ref_block = f"REFERENCE_EXPECTED_FINAL_STATEMENT_OR_ANSWER:\n{reference}"

        return [
            ChatMessage(role="system", content=judge_instructions),
            ChatMessage(
                role="user",
                content=(
                    f"PROBLEM:\n{problem}\n\n"
                    f"CANDIDATE_FINAL_ANSWER:\n{candidate.answer}\n\n"
                    f"CRITIC_RISK_AND_NOTES:\n{critic.risk}\n{critic.notes}\n\n"
                    f"{ref_block}\n"
                ),
            ),
        ]

    def parse(self, text: str, *, reference: str | None, **kwargs) -> JudgeVerdict:
        try:
            obj = json.loads(text.strip())
            score = float(obj.get("score", 0.0))
            confidence = float(obj.get("confidence", 0.0))
            is_correct = obj.get("is_correct")
            if is_correct is not None:
                is_correct = bool(is_correct)
            notes = str(obj.get("notes", ""))
        except Exception:  # noqa: BLE001
            score = 0.0
            confidence = 0.0
            is_correct = None
            notes = text.strip()[:500]

        score = max(0.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))

        if reference is None:
            is_correct = None

        return JudgeVerdict(score=score, confidence=confidence, is_correct=is_correct, notes=notes)
