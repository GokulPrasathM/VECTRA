from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..types import ChatMessage
from .python_tool import PythonTool


class ChatClient:
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        n: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> list[str]:
        ...


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    code: str


@dataclass(frozen=True)
class ToolLoopConfig:
    max_turns: int = 32
    temperature: float = 0.7
    model: str | None = None
    python_timeout_s: float = 6.0


def _extract_final(text: str) -> str | None:
    marker = "FINAL:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    return text[idx + len(marker) :].strip() or None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(obj, dict):
        return obj
    return None


def parse_tool_request(text: str) -> ToolRequest | None:
    obj = _extract_json_object(text)
    if not obj:
        return None
    tool = obj.get("tool")
    code = obj.get("code")
    if tool == "python" and isinstance(code, str) and code.strip():
        return ToolRequest(tool="python", code=code)
    return None


class ToolLoopRunner:
    def __init__(
        self,
        *,
        client: ChatClient,
        python_tool: PythonTool,
        config: ToolLoopConfig | None = None,
    ) -> None:
        self._client = client
        self._python = python_tool
        self._config = config or ToolLoopConfig()

    async def run(self, messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
        transcript = list(messages)

        for _ in range(self._config.max_turns):
            outputs = await self._client.chat(
                transcript,
                model=self._config.model,
                temperature=self._config.temperature,
                n=1,
            )
            assistant_text = outputs[0]
            transcript.append(ChatMessage(role="assistant", content=assistant_text))

            final = _extract_final(assistant_text)
            if final is not None:
                return final, transcript

            req = parse_tool_request(assistant_text)
            if req and req.tool == "python":
                tool_out = self._python.execute(req.code, timeout_s=self._config.python_timeout_s)
                transcript.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "PYTHON_OUTPUT:\n"
                            f"{tool_out}\n\n"
                            "Continue. If you need python again, emit a new tool JSON object."
                        ),
                    )
                )

        return "", transcript

