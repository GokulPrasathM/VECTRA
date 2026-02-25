from __future__ import annotations

import argparse
import json
import os

from .eval import EvaluateConfig, evaluate
from .scenario.attempts import AttemptConfig
from .solve import SolveConfig, solve
from .solve_scenario import ScenarioSolveConfig, solve_scenario
from .tools.tool_loop import ToolLoopConfig
from .vllm_server import VLLMServerConfig


def _apply_openai_env(*, api_key: str | None, base_url: str | None, model: str | None) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if model:
        os.environ["OPENAI_MODEL"] = model


def _maybe_vllm_from_args(args: argparse.Namespace) -> VLLMServerConfig | None:
    if not getattr(args, "vllm_model_path", None):
        return None
    served = getattr(args, "vllm_served_model_name", None) or (
        getattr(args, "model", None) or "local-model"
    )
    return VLLMServerConfig(
        model_path=args.vllm_model_path,
        served_model_name=served,
        host=getattr(args, "vllm_host", "127.0.0.1"),
        port=int(getattr(args, "vllm_port", 8000)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="vectra")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve")
    p_solve.add_argument("--problem", required=True)
    p_solve.add_argument("--reference")
    p_solve.add_argument("--trace", action="store_true")
    p_solve.add_argument("--api-key")
    p_solve.add_argument("--base-url")
    p_solve.add_argument("--model")
    p_solve.add_argument("--vllm-model-path")
    p_solve.add_argument("--vllm-served-model-name")
    p_solve.add_argument("--vllm-host", default="127.0.0.1")
    p_solve.add_argument("--vllm-port", type=int, default=8000)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--api-key")
    p_eval.add_argument("--base-url")
    p_eval.add_argument("--model")
    p_eval.add_argument("--vllm-model-path")
    p_eval.add_argument("--vllm-served-model-name")
    p_eval.add_argument("--vllm-host", default="127.0.0.1")
    p_eval.add_argument("--vllm-port", type=int, default=8000)

    p_scen = sub.add_parser("scenario-solve")
    p_scen.add_argument("--problem", required=True)
    p_scen.add_argument("--trace", action="store_true")
    p_scen.add_argument("--api-key")
    p_scen.add_argument("--base-url")
    p_scen.add_argument("--model")
    p_scen.add_argument("--attempts", type=int, default=8)
    p_scen.add_argument("--early-stop", type=int, default=4)
    p_scen.add_argument("--max-turns", type=int, default=32)
    p_scen.add_argument("--python-timeout", type=float, default=6.0)
    p_scen.add_argument("--vllm-model-path")
    p_scen.add_argument("--vllm-served-model-name")
    p_scen.add_argument("--vllm-host", default="127.0.0.1")
    p_scen.add_argument("--vllm-port", type=int, default=8000)

    args = parser.parse_args()

    if args.cmd == "solve":
        _apply_openai_env(api_key=args.api_key, base_url=args.base_url, model=args.model)
        vllm = _maybe_vllm_from_args(args)
        res = solve(
            args.problem,
            SolveConfig(model=args.model, vllm=vllm, return_trace=bool(args.trace)),
            reference=args.reference,
        )
        print(res.answer)
        if args.trace:
            print(json.dumps([e.__dict__ for e in (res.trace or [])], indent=2))

    if args.cmd == "eval":
        _apply_openai_env(api_key=args.api_key, base_url=args.base_url, model=args.model)
        vllm = _maybe_vllm_from_args(args)
        cfg = EvaluateConfig(solve=SolveConfig(model=args.model, vllm=vllm), limit=args.limit)
        rep = evaluate(args.dataset, cfg)
        print(json.dumps(rep.__dict__, indent=2))

    if args.cmd == "scenario-solve":
        _apply_openai_env(api_key=args.api_key, base_url=args.base_url, model=args.model)
        vllm = _maybe_vllm_from_args(args)
        attempt_cfg = AttemptConfig(
            attempts=int(args.attempts),
            early_stop=int(args.early_stop),
            max_concurrency=min(int(args.attempts), 16),
            tool_loop=ToolLoopConfig(
                max_turns=int(args.max_turns),
                model=args.model,
                python_timeout_s=float(args.python_timeout),
            ),
        )
        res = solve_scenario(
            args.problem,
            ScenarioSolveConfig(
                model=args.model,
                vllm=vllm,
                attempts=attempt_cfg,
                return_trace=bool(args.trace),
            ),
        )
        print(res.answer)
        if args.trace:
            print(json.dumps([e.__dict__ for e in (res.trace or [])], indent=2))


if __name__ == "__main__":
    main()
