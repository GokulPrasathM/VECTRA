from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Registry:
    roles: dict[str, object] = field(default_factory=dict)
    policies: dict[str, object] = field(default_factory=dict)

    def register_role(self, name: str, role: object) -> None:
        self.roles[name] = role

    def register_policy(self, name: str, policy: object) -> None:
        self.policies[name] = policy


default_registry = Registry()
