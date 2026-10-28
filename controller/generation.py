"""Shared answer-generation policy for serving and offline candidate replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

from .sampling import SamplingRecipe
from .templates import SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationProtocol:
    system_prompt: str
    sampling: SamplingRecipe | None = None

    @classmethod
    def from_config(cls, config):
        # Read only request policy: archived model paths need not still exist.
        config = asdict(config) if is_dataclass(config) else config
        prompt = (config.get("context") or {}).get("system_prompt")
        sampling = config.get("sampling")
        return cls(SYSTEM_PROMPT if prompt is None else prompt,
                   SamplingRecipe(**sampling) if sampling is not None else None)

    def record(self):
        return {"version": 1, "system_prompt": self.system_prompt,
                "sampling_recipe": (self.sampling or SamplingRecipe()).record(),
                "citation_constraint": "required_reference_map_v1"}


def candidate_request(candidate, max_new_tokens):
    """Use exactly the required IDs, including an empty constrained allowlist.

    An empty tuple forbids reserved citation IDs; None disables the grammar.
    Warmup requests intentionally do not use this answer-generation interface.
    """
    return {"input_ids": candidate.input_ids, "max_new_tokens": max_new_tokens,
            "citation_ids": tuple(ref for ref, key in candidate.reference_map
                                  if key in candidate.required_blocks)}


def validate_candidate_protocol(cases, protocol):
    """The config must describe the actual frozen prompt, not just its metadata."""
    for case in cases:
        for candidate in case.candidates:
            prompts = [message.content for message in candidate.messages if message.role == "system"]
            if prompts != [protocol.system_prompt]:
                raise ValueError("frozen candidate system prompt differs from generation configuration")
