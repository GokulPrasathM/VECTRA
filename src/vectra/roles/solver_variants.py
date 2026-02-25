from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import CandidateAnswer, ChatMessage


def _extract_final(text: str) -> str:
    m = re.search(r"^\s*FINAL\s*:\s*(.+)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


@dataclass(frozen=True)
class SolverRole:
    name: str
    system_prompt: str

    def build_messages(self, *, problem: str, **kwargs) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=self.system_prompt,
            ),
            ChatMessage(
                role="user",
                content=(
                    "Solve the problem. Think carefully and check edge cases. "
                    "Output your final answer on a single line prefixed with 'FINAL:'.\n\n"
                    f"PROBLEM:\n{problem}\n"
                ),
            ),
        ]

    def parse(self, text: str, **kwargs) -> CandidateAnswer:
        final = _extract_final(text)
        return CandidateAnswer(answer=final, solver_name=self.name, rationale=None, assumptions=[])


class DefaultSolverVariants:
    def __init__(self) -> None:
        self.roles = [
            SolverRole(
                name="solver_algebra",
                system_prompt=(
                    "You are a careful mathematical problem solver. "
                    "Prefer algebraic manipulation and case splits. Avoid leaps."
                ),
            ),
            SolverRole(
                name="solver_invariants",
                system_prompt=(
                    "You are a mathematical problem solver specializing in invariants and number theory. "
                    "Check modular constraints and boundary cases."
                ),
            ),
            SolverRole(
                name="solver_constructive",
                system_prompt=(
                    "You are a mathematical problem solver specializing in constructive arguments and bounds. "
                    "Be explicit about assumptions."
                ),
            ),
        ]
