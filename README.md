# Improving Large Language Model Reasoning Accuracy via Multi-Agent Optimization

**VECTRA** = **V**erification-**E**nhanced **C**onsensus **T**oolchain for **R**easoning **A**ccuracy.

VECTRA is a pip-installable, OpenAI-compatible SDK that improves **reasoning accuracy** by treating inference as a **budgeted optimization problem**:
- generate diverse candidate answers (multi-agent)
- score/risk-assess candidates (judge + critic)
- select/accept deterministically under a fixed call budget

It is implemented as an orchestration layer (not a model fine-tuning method) and is demonstrated primarily on math-style reasoning, but the core control loop is domain-agnostic.

## Quickstart

### 1) Install (editable)

```bash
pip install -e .
```

### 2) Configure environment

Set:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (default: `https://api.openai.com`)
- `OPENAI_MODEL` (example: `gpt-4.1-mini` or your provider model name)

Notes:
- This package uses the OpenAI-compatible endpoint `POST /v1/chat/completions`.
- “Batching” is implemented via client-side concurrency (`max_concurrency`).

### 3) Use from Python

```python
from vectra import SolveConfig, solve

problem = "Prove that for all integers n, n^3 - n is divisible by 3."
result = solve(
	problem,
	SolveConfig(
		max_rounds=2,
		max_calls=10,
		max_concurrency=6,
		accept_confidence=0.75,
		accept_margin=0.15,
		return_trace=False,
	),
)
print(result.answer)
```

### Scenario / tool-loop solving (Transformers or OpenAI-compatible)

For notebook-style "tool execution + multiple attempts + early-stop voting":

```python
from vectra import ScenarioSolveConfig, solve_scenario
from vectra.scenario.attempts import AttemptConfig
from vectra.tools.tool_loop import ToolLoopConfig

result = solve_scenario(
	"Compute 2+2 and show your final answer.",
	ScenarioSolveConfig(
		attempts=AttemptConfig(
			attempts=8,
			early_stop=4,
			max_concurrency=8,
			tool_loop=ToolLoopConfig(max_turns=16, temperature=0.7),
		),
	),
)
print(result.answer)
```

### Debugging / trace

```python
result = solve(problem, SolveConfig(return_trace=True))
print(result.trace[-1])
```

## Developer customization

- Add your own solver roles by implementing `build_messages()` + `parse()` and passing them into `SolveConfig(solver_roles=[...])`.
- Add your own controller policy by implementing `ControllerPolicy`.

## Dataset evaluation

Dataset format is JSONL with at least:

```json
{"id": "1", "problem": "...", "reference": "..."}
```

Then:

```python
from vectra import EvaluateConfig, evaluate
report = evaluate("data/smoke.jsonl", EvaluateConfig())
print(report)
```

## Technical method

See `docs/technical-method.md` for a compact description of the adaptive controller + acceptance criteria.

## Publishing

This repository is structured as a standard `pyproject.toml` Python package. To publish (without changing any code):

1) Build distributions

```bash
python -m pip install --upgrade build twine
python -m build
```

2) Upload to TestPyPI (recommended first)

```bash
python -m twine upload --repository testpypi dist/*
```

3) Upload to PyPI

```bash
python -m twine upload dist/*
```

Notes:
- This project publishes to PyPI as `vectra-reasoning` (distribution name). The import package remains `vectra`.
- The CLI entrypoint is `vectra` (or `python -m vectra`).
