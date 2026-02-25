from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..types import ChatMessage


@dataclass(frozen=True)
class RoleOutput:
    text: str


class Role(Protocol):
    name: str

    def build_messages(self, *, problem: str, **kwargs) -> list[ChatMessage]:
        ...

    def parse(self, text: str, **kwargs):
        ...
