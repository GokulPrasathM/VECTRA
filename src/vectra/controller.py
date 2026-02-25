from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .executor import Executor
from .policies.base import ControllerDecision, ControllerPolicy
from .types import CandidateAnswer, CriticVerdict, JudgeVerdict, SolveResult, TraceEvent


class DefaultControllerPolicy:
    def decide(
        self,
        *,
        candidates: list[CandidateAnswer],
        judge: list[JudgeVerdict],
        critic: list[CriticVerdict] | None,
        accept_confidence: float,
        accept_margin: float,
    ) -> ControllerDecision:
        scores = []
        for i, j in enumerate(judge):
            risk = 0.0
            if critic is not None:
                risk = float(critic[i].risk)
            s = (0.75 * float(j.confidence) + 0.25 * float(j.score)) - 0.5 * risk
            scores.append(s)

        best = max(range(len(scores)), key=lambda i: scores[i])
        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else -1.0)

        confidence = max(0.0, min(1.0, float(judge[best].confidence)))
        accept = confidence >= accept_confidence and margin >= accept_margin
        notes = f"best_score={scores[best]:.3f} margin={margin:.3f}"
        return ControllerDecision(accept=accept, selected_index=best, confidence=confidence, notes=notes)


@dataclass
class Controller:
    executor: Executor
    solver_roles: list
    critic_role: object
    judge_role: object
    policy: ControllerPolicy | None = None

    def run(
        self,
        *,
        problem: str,
        reference: str | None,
        max_rounds: int,
        max_calls: int,
        temperature: float,
        accept_confidence: float,
        accept_margin: float,
        return_trace: bool,
    ) -> SolveResult:
        policy = self.policy or DefaultControllerPolicy()
        trace: list[TraceEvent] = []

        async def _arun() -> SolveResult:
            candidates: list[CandidateAnswer] = []
            last_decision: ControllerDecision | None = None

            for round_idx in range(max_rounds):
                reqs = []
                for role in self.solver_roles:
                    msgs = role.build_messages(problem=problem)
                    reqs.append((msgs, {"temperature": temperature, "n": 1}))

                results = await self.executor.call_many(reqs)

                round_candidates: list[CandidateAnswer] = []
                for role, res in zip(self.solver_roles, results, strict=False):
                    parsed = role.parse(res.outputs[0])
                    round_candidates.append(parsed)

                candidates = round_candidates
                trace.append(
                    TraceEvent(
                        type="solver_round",
                        payload={
                            "round": round_idx,
                            "candidates": [c.__dict__ for c in candidates],
                        },
                    )
                )

                if self.executor.calls_made >= max_calls:
                    break

                critic_reqs = []
                for c in candidates:
                    msgs = self.critic_role.build_messages(problem=problem, candidate=c)
                    critic_reqs.append((msgs, {"temperature": 0.2, "n": 1}))
                critic_results = await self.executor.call_many(critic_reqs)
                critics: list[CriticVerdict] = []
                for c, res in zip(candidates, critic_results, strict=False):
                    critics.append(self.critic_role.parse(res.outputs[0], candidate=c))
                trace.append(
                    TraceEvent(
                        type="critic",
                        payload={
                            "round": round_idx,
                            "critics": [cv.__dict__ for cv in critics],
                        },
                    )
                )

                if self.executor.calls_made >= max_calls:
                    break

                judge_reqs = []
                for c, cv in zip(candidates, critics, strict=False):
                    msgs = self.judge_role.build_messages(
                        problem=problem,
                        candidate=c,
                        critic=cv,
                        reference=reference,
                    )
                    judge_reqs.append((msgs, {"temperature": 0.0, "n": 1}))

                judge_results = await self.executor.call_many(judge_reqs)
                verdicts: list[JudgeVerdict] = []
                for c, res in zip(candidates, judge_results, strict=False):
                    verdicts.append(self.judge_role.parse(res.outputs[0], reference=reference))

                trace.append(
                    TraceEvent(
                        type="judge",
                        payload={
                            "round": round_idx,
                            "verdicts": [v.__dict__ for v in verdicts],
                        },
                    )
                )

                decision = policy.decide(
                    candidates=candidates,
                    judge=verdicts,
                    critic=critics,
                    accept_confidence=accept_confidence,
                    accept_margin=accept_margin,
                )
                last_decision = decision
                trace.append(
                    TraceEvent(
                        type="decision",
                        payload={
                            "round": round_idx,
                            **decision.__dict__,
                        },
                    )
                )

                if decision.accept:
                    selected = candidates[decision.selected_index]
                    final_is_correct = verdicts[decision.selected_index].is_correct
                    trace.append(
                        TraceEvent(
                            type="final_verdict",
                            payload={
                                "is_correct": final_is_correct,
                                "judge": verdicts[decision.selected_index].__dict__,
                            },
                        )
                    )
                    return SolveResult(
                        answer=selected.answer,
                        confidence=decision.confidence,
                        calls_made=self.executor.calls_made,
                        trace=trace if return_trace else None,
                    )

                if self.executor.calls_made >= max_calls:
                    break

            if candidates:
                selected_idx = last_decision.selected_index if last_decision else 0
                selected_idx = max(0, min(len(candidates) - 1, selected_idx))
                answer = candidates[selected_idx].answer
                conf = last_decision.confidence if last_decision else 0.0
            else:
                answer = ""
                conf = 0.0

            trace.append(
                TraceEvent(
                    type="final_verdict",
                    payload={
                        "is_correct": None,
                        "notes": "budget_exhausted_or_no_accept",
                    },
                )
            )
            return SolveResult(
                answer=answer,
                confidence=float(conf),
                calls_made=self.executor.calls_made,
                trace=trace if return_trace else None,
            )

        return asyncio.run(_arun())
