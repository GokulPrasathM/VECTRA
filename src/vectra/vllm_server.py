from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx


class VLLMServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class VLLMServerConfig:
    model_path: str
    served_model_name: str

    host: str = "127.0.0.1"
    port: int = 8000

    startup_timeout_s: float = 180.0
    probe_timeout_s: float = 2.0

    log_path: str = "vllm_server.log"

    seed: int | None = None
    tensor_parallel_size: int | None = None
    max_num_seqs: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    max_model_len: int | None = None
    stream_interval: int | None = None

    async_scheduling: bool = True
    disable_log_stats: bool = True
    enable_prefix_caching: bool = True


class VLLMServer:
    def __init__(self, config: VLLMServerConfig):
        self._cfg = config
        self._proc: subprocess.Popen[object] | None = None
        self._log_fp = None

    @property
    def base_url(self) -> str:
        return f"http://{self._cfg.host}:{self._cfg.port}"

    def _build_cmd(self) -> list[str]:
        cmd: list[str] = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._cfg.model_path,
            "--served-model-name",
            self._cfg.served_model_name,
            "--host",
            self._cfg.host,
            "--port",
            str(self._cfg.port),
        ]

        def opt(flag: str, value: object | None) -> None:
            if value is None:
                return
            cmd.extend([flag, str(value)])

        opt("--seed", self._cfg.seed)
        opt("--tensor-parallel-size", self._cfg.tensor_parallel_size)
        opt("--max-num-seqs", self._cfg.max_num_seqs)
        opt("--gpu-memory-utilization", self._cfg.gpu_memory_utilization)
        opt("--dtype", self._cfg.dtype)
        opt("--kv-cache-dtype", self._cfg.kv_cache_dtype)
        opt("--max-model-len", self._cfg.max_model_len)
        opt("--stream-interval", self._cfg.stream_interval)

        if self._cfg.async_scheduling:
            cmd.append("--async-scheduling")
        if self._cfg.disable_log_stats:
            cmd.append("--disable-log-stats")
        if self._cfg.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        return cmd

    def start(self) -> None:
        if self._proc is not None:
            return

        cmd = self._build_cmd()
        self._log_fp = open(self._cfg.log_path, "w", encoding="utf-8")

        kwargs: dict[str, object] = {
            "stdout": self._log_fp,
            "stderr": subprocess.STDOUT,
            "cwd": os.getcwd(),
        }

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]

    def _read_log_tail(self, max_chars: int = 8000) -> str:
        try:
            if not os.path.exists(self._cfg.log_path):
                return ""
            with open(self._cfg.log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            if len(data) <= max_chars:
                return data
            return data[-max_chars:]
        except Exception:  # noqa: BLE001
            return ""

    def wait_ready(self) -> None:
        if self._proc is None:
            raise VLLMServerError("Server not started")

        deadline = time.time() + float(self._cfg.startup_timeout_s)
        url = f"{self.base_url}/v1/models"

        with httpx.Client(timeout=httpx.Timeout(self._cfg.probe_timeout_s)) as client:
            while time.time() < deadline:
                rc = self._proc.poll()
                if rc is not None:
                    if self._log_fp:
                        self._log_fp.flush()
                    tail = self._read_log_tail()
                    raise VLLMServerError(f"vLLM server exited with code {rc}. Log tail:\n{tail}")

                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return
                except Exception:  # noqa: BLE001
                    pass

                time.sleep(1.0)

        tail = self._read_log_tail()
        raise VLLMServerError(
            f"vLLM server failed to become ready within timeout. Log tail:\n{tail}"
        )

    def stop(self, *, grace_s: float = 5.0) -> None:
        proc = self._proc
        self._proc = None

        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_s)
                except Exception:  # noqa: BLE001
                    proc.kill()
        finally:
            if self._log_fp is not None:
                try:
                    self._log_fp.flush()
                finally:
                    self._log_fp.close()
                self._log_fp = None

    def __enter__(self) -> "VLLMServer":
        self.start()
        self.wait_ready()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.stop()
