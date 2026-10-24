"""Explicit request sampling, separate from model/cache identity and output budget."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SamplingRecipe:
    temperature: float = 0
    top_p: float = 1
    top_k: int = -1
    min_p: float = 0

    def __post_init__(self):
        for name in ("temperature", "top_p", "min_p"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"sampling.{name} must be a finite number")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1:
            raise ValueError("sampling requires temperature >= 0, 0 < top_p <= 1, 0 <= min_p <= 1")
        if type(self.top_k) is not int or self.top_k != -1 and self.top_k < 1:
            raise ValueError("sampling.top_k must be -1 or a positive integer")

    def record(self) -> dict:
        return asdict(self)


def request_sampling(recipe: SamplingRecipe | None) -> dict:
    """None preserves the historical wire, including omitted sampler defaults."""
    return {"temperature": 0} if recipe is None else recipe.record()


def bound_recipe(record: dict | None) -> SamplingRecipe | None:
    """Parse an offline binding once; missing legacy metadata remains unknown."""
    if record is None or record == {}:
        return None
    if set(record) != {"temperature", "top_p", "top_k", "min_p"} and record != {"temperature": 0}:
        raise ValueError("sampling binding needs the complete recipe or the legacy temperature: 0 wire")
    return SamplingRecipe(**record)
