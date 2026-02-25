from __future__ import annotations

import os

import pytest

from vectra import EvaluateConfig, SolveConfig, evaluate, solve


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_solve_runs() -> None:
    result = solve("Compute 1+1.", SolveConfig(max_rounds=1, max_calls=6, return_trace=True), reference="2")
    assert isinstance(result.answer, str)
    assert result.calls_made >= 1


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
def test_evaluate_runs() -> None:
    report = evaluate("data/smoke.jsonl", EvaluateConfig(solve=SolveConfig(max_rounds=1, max_calls=9)))
    assert report.total >= 1
    assert 0.0 <= report.accuracy <= 1.0
