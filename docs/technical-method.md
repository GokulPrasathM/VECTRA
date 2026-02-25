# Technical method (multi-agent optimization controller)

This document describes the core algorithmic method implemented by this package.

## Objective

Given a reasoning problem (often math-style in our demos), produce a final answer with higher accuracy than a single model call by:
- Running multiple specialized agent roles per problem (diversity)
- Using an optimization/controller loop to adaptively allocate more agent calls only when uncertainty is high
- Applying deterministic acceptance criteria based on evidence from a critic and a judge

The intent is to behave like a probabilistic generator with a deterministic selection mindset.

## Roles

### Solver roles
Multiple solver variants are run in parallel (batched client-side concurrency). Each solver outputs a single final answer line prefixed with `FINAL:`.

### Critic role
For each candidate answer, an adversarial critic produces a risk estimate and notes about missing cases or potential invalid reasoning.

### Judge role
A strict grader returns a structured verdict:
- `score` (0..1)
- `confidence` (0..1)
- `is_correct` (bool when a reference is available; otherwise unknown)

## Controller loop (adaptive allocation)

For each round:
1. Fan-out solver calls to generate candidate answers
2. Run critic calls per candidate
3. Run judge calls per candidate (with optional reference)
4. Aggregate judge/critic evidence into a scalar score per candidate
5. Apply deterministic acceptance criteria:
   - accept if best candidate confidence ≥ `accept_confidence`
   - and the score margin between best and runner-up ≥ `accept_margin`

If criteria are not met, the controller proceeds to the next round (up to `max_rounds`) until the call budget (`max_calls`) is exhausted.

## Batching

Because standard OpenAI-compatible chat/completions APIs do not guarantee a server-side batch endpoint, batching is implemented as:
- Many agent calls dispatched concurrently with `max_concurrency`
- Optional use of `n` completions per request if supported by the provider

## Extensibility hooks

Developers can replace:
- The solver set (add/remove solver roles)
- The critic or judge implementation
- The controller policy (decision function mapping evidence → accept/select)

## Evaluation

When a `reference` is provided per problem, the judge outputs `is_correct`, enabling automatic evaluation over datasets.

## Tool-execution loop (scenario solve)

Separately from the judge/critic controller, the library also supports a notebook-style tool loop:
- the model may request Python execution via a strict JSON tool protocol
- multiple attempts run in parallel
- early-stop voting selects the most common normalized final answer

This is useful when correctness depends on intermediate computation, algebraic checking, or sanity tests.
