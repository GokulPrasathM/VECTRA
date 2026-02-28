from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vectra import (
    ScenarioSolveConfig,
    TransformersClient,
    TransformersClientConfig,
    solve_scenario_async,
)
from vectra.scenario.attempts import AttemptConfig
from vectra.tools.tool_loop import ToolLoopConfig
from vectra.types import ChatMessage


def _utc_ts() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_compact(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "", s)
    return s


def extract_final(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"FINAL\s*:\s*(.+)$", text.strip(), flags=re.IGNORECASE)
    return (m.group(1).strip() if m else text.strip())


def _normalize_numeric(s: str) -> str:
    s = _norm_compact(s).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return m.group(0) if m else s


def _normalize_choice_letter(s: str) -> str:
    s2 = (s or "").strip().upper()
    m = re.search(r"\b([A-E])\b", s2)
    if m:
        return m.group(1)
    if s2[:1] in {"A", "B", "C", "D", "E"}:
        return s2[:1]
    return s2[:1]


def is_correct(pred: str, ref: str, *, kind: str) -> bool:
    if kind == "numeric":
        return _normalize_numeric(pred) == _normalize_numeric(ref)
    if kind == "choice":
        return _normalize_choice_letter(pred) == _normalize_choice_letter(ref)
    return _norm_compact(pred) == _norm_compact(ref)


def _format_freeform_problem(question: str) -> str:
    q = _norm(question)
    return q + "\n\nReturn exactly one line: FINAL: <answer>."


def _format_mc_problem(stem: str, choices: list[tuple[str, str]]) -> str:
    lines = [_norm(stem), "", "Choices:"]
    for label, text in choices:
        lines.append(f"{label}. {_norm(text)}")
    lines.append("")
    lines.append("Return exactly one line: FINAL: <choice-letter>. Example: FINAL: C")
    return "\n".join(lines)


def _require_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing dependency: datasets. Install with: pip install datasets") from e
    return load_dataset


def _sample_indices(n_total: int, n_sample: int, seed: int) -> list[int]:
    n_sample = min(int(n_sample), int(n_total))
    rnd = random.Random(seed)
    return rnd.sample(range(n_total), n_sample)


def _parse_gsm8k_reference(ans: str) -> str:
    if "####" in (ans or ""):
        return (ans or "").split("####")[-1].strip()
    return (ans or "").strip()


def _parse_math_reference(ans: str) -> str:
    if not ans:
        return ""
    m = re.findall(r"\\boxed\{([^}]*)\}", ans)
    if m:
        return m[-1].strip()
    m2 = re.findall(r"\\boxed\s*([^\n\r]+)", ans)
    if m2:
        return m2[-1].strip()
    return ans.strip()


def _load_math500(load_dataset):
    candidates = [
        ("HuggingFaceH4/MATH-500", None),
        ("lighteval/MATH-500", None),
    ]
    last_err: Exception | None = None
    for name, subset in candidates:
        try:
            if subset is None:
                return load_dataset(name, split="test")
            return load_dataset(name, subset, split="test")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to load MATH500 from {candidates}: {last_err}")


def load_real_benchmarks(*, samples_per_suite: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    load_dataset = _require_load_dataset()

    # 1) GSM8K (test)
    gsm = load_dataset("gsm8k", "main", split="test")
    gsm_idx = _sample_indices(len(gsm), samples_per_suite, seed + 1)
    gsm_items: list[dict[str, Any]] = []
    for i in gsm_idx:
        row = gsm[int(i)]
        q = row.get("question", "")
        ref = _parse_gsm8k_reference(row.get("answer", ""))
        gsm_items.append(
            {
                "id": f"gsm8k:test:{int(i)}",
                "problem": _format_freeform_problem(q),
                "reference": str(ref),
                "kind": "numeric",
            }
        )

    # 2) ARC-Challenge (test)
    arc = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    arc_idx = _sample_indices(len(arc), samples_per_suite, seed + 2)
    arc_items: list[dict[str, Any]] = []
    for i in arc_idx:
        row = arc[int(i)]
        q = row.get("question", {}) or {}
        if isinstance(q, dict):
            stem = q.get("stem", "")
            ch = q.get("choices", {}) or {}
            labels = ch.get("label", []) or []
            texts = ch.get("text", []) or []
        else:
            stem = row.get("question", "")
            ch = row.get("choices", {}) or {}
            labels = ch.get("label", []) or []
            texts = ch.get("text", []) or []
        choices = list(zip(labels, texts, strict=False))
        ref = row.get("answerKey", "")
        arc_items.append(
            {
                "id": f"arc_challenge:test:{int(i)}",
                "problem": _format_mc_problem(stem, choices),
                "reference": str(ref),
                "kind": "choice",
            }
        )

    # 3) MATH500 (test)
    math500 = _load_math500(load_dataset)
    m_idx = _sample_indices(len(math500), samples_per_suite, seed + 3)
    m_items: list[dict[str, Any]] = []
    for i in m_idx:
        row = math500[int(i)]
        q = row.get("problem", "") or row.get("question", "")
        ref = row.get("answer", None)
        if ref is None:
            ref = _parse_math_reference(row.get("solution", "") or row.get("final_answer", "") or "")
        m_items.append(
            {
                "id": f"math500:test:{int(i)}",
                "problem": _format_freeform_problem(str(q)),
                "reference": str(ref),
                "kind": "text",
            }
        )

    # 4) CommonsenseQA (validation)
    csqa = load_dataset("commonsense_qa", split="validation")
    csqa_idx = _sample_indices(len(csqa), samples_per_suite, seed + 4)
    csqa_items: list[dict[str, Any]] = []
    for i in csqa_idx:
        row = csqa[int(i)]
        stem = row.get("question", "")
        ch = row.get("choices", {}) or {}
        labels = ch.get("label", []) or []
        texts = ch.get("text", []) or []
        choices = list(zip(labels, texts, strict=False))
        ref = row.get("answerKey", "")
        csqa_items.append(
            {
                "id": f"commonsense_qa:val:{int(i)}",
                "problem": _format_mc_problem(stem, choices),
                "reference": str(ref),
                "kind": "choice",
            }
        )

    return {
        "GSM8K": gsm_items,
        "ARC-Challenge": arc_items,
        "MATH500": m_items,
        "CommonsenseQA": csqa_items,
    }


@dataclass(frozen=True)
class RunConfig:
    model_id: str
    temperature: float
    max_new_tokens: int

    batch_max_size: int
    batch_max_wait_ms: int

    # device map options
    force_single_cuda_device: bool
    cuda_device_index: int
    device_map: str | None

    samples_per_suite: int
    seed: int

    baseline_concurrency: int

    vectra_attempts: int
    vectra_early_stop: int
    vectra_max_turns: int
    vectra_problem_concurrency: int


def _safe_run(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return (p.stdout or "").strip()
    except Exception:
        return ""


def _detect_git_revision() -> str | None:
    rev = _safe_run(["git", "rev-parse", "HEAD"])
    return rev.strip() if rev else None


def _detect_gpu_info() -> dict[str, Any]:
    out = _safe_run(["nvidia-smi", "-L"])
    return {"nvidia_smi_L": out} if out else {}


def _python_env_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "argv": sys.argv,
        "pid": os.getpid(),
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        info: dict[str, Any] = {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_0"] = str(torch.cuda.get_device_name(0))
        return info
    except Exception as e:  # noqa: BLE001
        return {"torch_import_error": repr(e)}


def _resolve_device_map(cfg: RunConfig) -> str | dict[str, Any]:
    if cfg.device_map is not None:
        return cfg.device_map

    try:
        import torch  # type: ignore

        if torch.cuda.is_available() and cfg.force_single_cuda_device:
            return {"": int(cfg.cuda_device_index)}
        if torch.cuda.is_available():
            return "auto"
        return "cpu"
    except Exception:
        return "auto"


async def _baseline_answer(*, client: TransformersClient, problem: str, cfg: RunConfig) -> str:
    system = "You are a careful reasoner. Do not call tools. Return exactly one line: FINAL: <answer>."
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=problem),
    ]
    out = await client.chat(messages, temperature=float(cfg.temperature), max_tokens=int(cfg.max_new_tokens), n=1)
    return extract_final(out[0])


async def _vectra_answer(*, client: TransformersClient, problem: str, cfg: RunConfig) -> str:
    tool_example = json.dumps({"tool": "python", "code": "print(2+2)"})
    system = (
        "You may use an external Python tool. "
        f"To call it, output ONLY a JSON object like: {tool_example}. "
        "To finish, output: FINAL: <your answer>."
    )

    solve_cfg = ScenarioSolveConfig(
        transformers_client=client,
        system_prompt=system,
        attempts=AttemptConfig(
            attempts=int(cfg.vectra_attempts),
            early_stop=int(cfg.vectra_early_stop),
            max_concurrency=int(cfg.vectra_attempts),
            tool_loop=ToolLoopConfig(
                max_turns=int(cfg.vectra_max_turns),
                temperature=float(cfg.temperature),
            ),
        ),
        return_trace=False,
    )
    res = await solve_scenario_async(problem, solve_cfg)
    return extract_final("FINAL: " + res.answer)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    acc = sum(1 for r in rows if r.get("correct")) / max(1, n)
    lat = [float(r.get("latency_s", 0.0)) for r in rows]
    if not lat:
        return {"n": n, "accuracy": acc, "avg_latency_s": 0.0, "p95_latency_s": 0.0}
    lat_sorted = sorted(lat)
    p95 = lat_sorted[max(0, math.ceil(0.95 * len(lat_sorted)) - 1)]
    return {
        "n": n,
        "accuracy": acc,
        "avg_latency_s": statistics.mean(lat),
        "p95_latency_s": p95,
    }


class JsonlLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = (self.log_dir / "run.jsonl").open("a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def event(self, typ: str, payload: dict[str, Any]) -> None:
        rec = {"ts": _utc_ts(), "type": typ, **payload}
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fp.flush()


async def run_suite(
    *,
    mode: str,
    suite: str,
    items: list[dict[str, Any]],
    client: TransformersClient,
    cfg: RunConfig,
    logger: JsonlLogger,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.baseline_concurrency if mode == "baseline" else cfg.vectra_problem_concurrency)

    async def one(it: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            t0 = time.time()
            if mode == "baseline":
                pred = await _baseline_answer(client=client, problem=str(it["problem"]), cfg=cfg)
            else:
                pred = await _vectra_answer(client=client, problem=str(it["problem"]), cfg=cfg)
            dt = time.time() - t0
            ref = str(it.get("reference", ""))
            kind = str(it.get("kind", "text"))
            correct = bool(is_correct(pred, ref, kind=kind))
            row = {
                "suite": suite,
                "mode": mode,
                "id": str(it.get("id")),
                "kind": kind,
                "pred": pred,
                "ref": ref,
                "correct": correct,
                "latency_s": float(dt),
            }
            logger.event("result", row)
            print(f"[{mode}] {suite} id={row['id']} correct={int(correct)} latency_s={dt:.2f}")
            return row

    tasks = [asyncio.create_task(one(it)) for it in items]
    rows: list[dict[str, Any]] = []
    for fut in asyncio.as_completed(tasks):
        rows.append(await fut)
    return rows


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run VECTRA impact demo from a terminal with audit logs.")
    p.add_argument("--model-id", default="openai/gpt-oss-20b")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=256)

    p.add_argument("--batch-max-size", type=int, default=8)
    p.add_argument("--batch-max-wait-ms", type=int, default=8)

    p.add_argument("--samples-per-suite", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--baseline-concurrency", type=int, default=8)

    p.add_argument("--vectra-attempts", type=int, default=4)
    p.add_argument("--vectra-early-stop", type=int, default=2)
    p.add_argument("--vectra-max-turns", type=int, default=16)
    p.add_argument("--vectra-problem-concurrency", type=int, default=2)

    p.add_argument(
        "--force-single-cuda-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid mixed CPU/GPU sharding by forcing all weights to cuda:0 (OOM risk).",
    )
    p.add_argument("--cuda-device-index", type=int, default=0)
    p.add_argument(
        "--device-map",
        default=None,
        help="Override Transformers device_map (e.g. 'auto', 'cpu'). If set, disables --force-single-cuda-device.",
    )

    p.add_argument(
        "--log-dir",
        default=None,
        help="Directory for outputs. Default: runs/<utc timestamp>/",
    )
    return p


async def _amain() -> int:
    args = _build_arg_parser().parse_args()

    run_cfg = RunConfig(
        model_id=str(args.model_id),
        temperature=float(args.temperature),
        max_new_tokens=int(args.max_new_tokens),
        batch_max_size=int(args.batch_max_size),
        batch_max_wait_ms=int(args.batch_max_wait_ms),
        force_single_cuda_device=bool(args.force_single_cuda_device) if args.device_map is None else False,
        cuda_device_index=int(args.cuda_device_index),
        device_map=str(args.device_map) if args.device_map is not None else None,
        samples_per_suite=int(args.samples_per_suite),
        seed=int(args.seed),
        baseline_concurrency=int(args.baseline_concurrency),
        vectra_attempts=int(args.vectra_attempts),
        vectra_early_stop=int(args.vectra_early_stop),
        vectra_max_turns=int(args.vectra_max_turns),
        vectra_problem_concurrency=int(args.vectra_problem_concurrency),
    )

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path(args.log_dir) if args.log_dir else (Path.cwd() / "runs" / ts)
    logger = JsonlLogger(log_dir)

    try:
        random.seed(run_cfg.seed)

        meta = {
            "run_id": ts,
            "git_rev": _detect_git_revision(),
            "python_env": _python_env_info(),
            "torch": _torch_info(),
            "gpu": _detect_gpu_info(),
            "config": run_cfg.__dict__,
        }
        (log_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.event("meta", meta)

        print("Log dir:", str(log_dir.resolve()))
        print("Model:", run_cfg.model_id)
        print("Seed:", run_cfg.seed)

        device_map = _resolve_device_map(run_cfg)
        logger.event("device_map", {"device_map": str(device_map)})
        print("device_map:", device_map)

        client = TransformersClient(
            TransformersClientConfig(
                model_id=run_cfg.model_id,
                max_new_tokens=run_cfg.max_new_tokens,
                device_map=device_map,
                torch_dtype="auto",
                batch_max_size=run_cfg.batch_max_size,
                batch_max_wait_ms=run_cfg.batch_max_wait_ms,
            )
        )

        suites = load_real_benchmarks(samples_per_suite=run_cfg.samples_per_suite, seed=run_cfg.seed)
        (log_dir / "benchmarks.json").write_text(
            json.dumps({k: [it["id"] for it in v] for k, v in suites.items()}, indent=2),
            encoding="utf-8",
        )
        print("Suites:", {k: len(v) for k, v in suites.items()})

        all_rows: dict[str, dict[str, list[dict[str, Any]]]] = {"baseline": {}, "vectra": {}}

        for suite, items in suites.items():
            print(f"\nBaseline: {suite}")
            logger.event("suite_start", {"mode": "baseline", "suite": suite, "n": len(items)})
            all_rows["baseline"][suite] = await run_suite(
                mode="baseline",
                suite=suite,
                items=items,
                client=client,
                cfg=run_cfg,
                logger=logger,
            )

        for suite, items in suites.items():
            print(f"\nVECTRA: {suite}")
            logger.event("suite_start", {"mode": "vectra", "suite": suite, "n": len(items)})
            all_rows["vectra"][suite] = await run_suite(
                mode="vectra",
                suite=suite,
                items=items,
                client=client,
                cfg=run_cfg,
                logger=logger,
            )

        summary_rows: list[dict[str, Any]] = []
        for suite in suites.keys():
            summary_rows.append({"suite": suite, "mode": "baseline", **_summarize(all_rows["baseline"][suite])})
            summary_rows.append({"suite": suite, "mode": "vectra", **_summarize(all_rows["vectra"][suite])})

        baseline_flat = [r for v in all_rows["baseline"].values() for r in v]
        vectra_flat = [r for v in all_rows["vectra"].values() for r in v]
        agg = {
            "baseline": _summarize(baseline_flat),
            "vectra": _summarize(vectra_flat),
        }
        impact = {
            "accuracy_delta": float(agg["vectra"]["accuracy"] - agg["baseline"]["accuracy"]),
            "avg_latency_delta_s": float(agg["vectra"]["avg_latency_s"] - agg["baseline"]["avg_latency_s"]),
        }

        summary = {
            "per_suite": summary_rows,
            "aggregate": agg,
            "impact": impact,
        }
        (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.event("summary", summary)

        print("\nAggregate summary")
        print("-----------------")
        print("baseline:", agg["baseline"]) 
        print("vectra:", agg["vectra"]) 
        print("impact:", impact)

        print("\nPaste-ready")
        print("----------")
        print(
            f"Model: {run_cfg.model_id} (Transformers local inference)\n"
            f"Benchmarks: GSM8K, ARC-Challenge, MATH500, CommonsenseQA (sampled subsets).\n"
            f"Scope: {agg['baseline']['n']} questions total (SAMPLES_PER_SUITE={run_cfg.samples_per_suite}).\n"
            f"Baseline: accuracy={agg['baseline']['accuracy']:.3f}, avg latency={agg['baseline']['avg_latency_s']:.3f}s.\n"
            f"VECTRA: accuracy={agg['vectra']['accuracy']:.3f}, avg latency={agg['vectra']['avg_latency_s']:.3f}s.\n"
            f"Net impact: accuracy delta={impact['accuracy_delta']:+.3f}, latency delta={impact['avg_latency_delta_s']:+.3f}s.\n"
        )

        return 0
    finally:
        logger.close()


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
