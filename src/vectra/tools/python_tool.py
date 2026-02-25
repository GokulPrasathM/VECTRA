from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol


class PythonTool(Protocol):
    def execute(self, code: str, *, timeout_s: float | None = None) -> str:
        ...


@dataclass
class SubprocessPythonTool:
    timeout_s: float = 6.0
    max_code_chars: int = 50_000

    def execute(self, code: str, *, timeout_s: float | None = None) -> str:
        if not isinstance(code, str):
            raise TypeError("code must be a str")
        if len(code) > self.max_code_chars:
            raise ValueError(f"code too long ({len(code)} chars)")

        effective_timeout = float(timeout_s) if timeout_s is not None else float(self.timeout_s)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[ERROR] Python execution timed out after {effective_timeout} seconds"

        out = (proc.stdout or "").rstrip()
        err = (proc.stderr or "").rstrip()
        if out and err:
            return f"{out}\n{err}"
        if err:
            return err
        return out or "[WARN] No output. Use print() to see results."
