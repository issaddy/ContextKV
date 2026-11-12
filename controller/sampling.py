"""Explicit request sampling, separate from model/cache identity and output budget."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SamplingRecipe:
    temperature: float = 0
    top_p: float = 1
    top_k: int = -1
    min_p: float = 0

    def record(self) -> dict:
        return asdict(self)


def request_sampling(recipe: SamplingRecipe | None) -> dict:
    """Emit the complete configured recipe, including greedy defaults."""
    return (recipe or SamplingRecipe()).record()
