"""DeciderKit: config-driven decision guardrails for Strands agents."""

from .audit import AuditLog
from .client import DeciderClient, DeciderError, DeciderUnavailable
from .engine import Decision, DecisionEngine, EscalationContext
from .recipe import BUILTIN_CHECKS, Recipe, RecipeError, load_recipe, with_mode

__version__ = "0.1.0"

__all__ = [
    "AuditLog", "BUILTIN_CHECKS", "DeciderClient", "DeciderError", "DeciderGuard",
    "DeciderUnavailable", "Decision", "DecisionEngine", "EscalationContext",
    "Recipe", "RecipeError", "load_recipe", "with_mode",
]


def __getattr__(name: str):
    # DeciderGuard needs strands-agents; import it lazily so the CLI and engine work without it.
    if name == "DeciderGuard":
        from .guard import DeciderGuard

        return DeciderGuard
    raise AttributeError(name)
