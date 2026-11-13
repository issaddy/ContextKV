"""Stop/continue values on the shared evidence representation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


STATE_FIELDS = (
    "remaining_sources", "open_needs", "required_rows", "processed_rows",
    "input_tokens", "remaining_tokens", "elapsed_seconds", "queue_depth",
    "gpu_ready_tokens", "host_ready_tokens", "prefill_token_ms", "transfer_token_ms", "answer_cost_ms",
)
SUMMARY_SIZE = 1540
VALUE_FIELDS = ("quality", "seconds", "gpu_gib_seconds", "host_gib_seconds")


@dataclass(frozen=True)
class ValuePrices:
    seconds: float = 0.0
    gpu_gib_seconds: float = 0.0
    host_gib_seconds: float = 0.0
    tail_seconds: float = 0.0

    def utility(self, values, tails):
        prices = values.new_tensor((self.seconds, self.gpu_gib_seconds, self.host_gib_seconds))
        return values[..., 0] - (values[..., 1:] * prices).sum(-1) - self.tail_seconds * tails


class StopValueHead(nn.Module):
    """Two terminal-return predictions; costs include all work after this state."""

    def __init__(self, summary_size=SUMMARY_SIZE, hidden_size=128):
        super().__init__()
        self.summary_size, self.hidden_size = summary_size, hidden_size
        self.network = nn.Sequential(
            nn.Linear(summary_size + len(STATE_FIELDS), hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 10),
        )
        self.register_buffer("mean", torch.zeros(summary_size + len(STATE_FIELDS)))
        self.register_buffer("scale", torch.ones(summary_size + len(STATE_FIELDS)))
        self.register_buffer("active", torch.ones(summary_size + len(STATE_FIELDS)))

    def forward(self, features):
        raw = self.network((features - self.mean) / self.scale * self.active).reshape(*features.shape[:-1], 2, 5)
        values = torch.cat((raw[..., :1].sigmoid(), nn.functional.softplus(raw[..., 1:4])), -1)
        return values, nn.functional.softplus(raw[..., 4])

    @staticmethod
    def features(summary, state):
        return torch.cat((summary, summary.new_tensor([state[name] for name in STATE_FIELDS])))

    @torch.inference_mode()
    def choose(self, summary, state, prices, *, can_continue):
        if not can_continue:
            return "finish"
        values, tails = self(self.features(summary, state).to(self.mean.device))
        utilities = prices.utility(values, tails)
        return "continue" if utilities[1] > utilities[0] else "finish"

    def save(self, path):
        torch.save({"summary_size": self.summary_size, "hidden_size": self.hidden_size,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        asset = torch.load(path, map_location=device, weights_only=True)
        model = cls(asset["summary_size"], asset["hidden_size"])
        model.load_state_dict(asset["state_dict"])
        return model.to(device).eval()
