"""Full-width forward Q(lambda) supervision from measured QA trajectories.

Each trajectory contains the shared feature vector, a real STOP answer score
and complete STOP cost at every state, plus the work paid to reach the next
state. Source texts and model token states are not copied into training rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from controller.value_policy import StopValueHead, ValuePrices, VALUE_FIELDS


def observed_values(row, device):
    return torch.tensor([row[name] for name in VALUE_FIELDS], dtype=torch.float32, device=device)


@torch.no_grad()
def lambda_targets(model, trajectory, prices, trace_lambda):
    """STOP ends now; CONTINUE pays one transition and follows the next decision."""
    rows = trajectory["states"]
    features = torch.tensor([row["features"] for row in rows], device=model.mean.device)
    predicted, tails = model(features)
    count = len(rows)
    observed = torch.stack([observed_values(row["stop"], model.mean.device) for row in rows])
    mask = torch.tensor([[True, index < count-1] for index in range(count)], device=model.mean.device)
    target = torch.zeros_like(predicted)
    target[:, 0] = observed
    realised_seconds = torch.zeros_like(tails)
    realised_seconds[:, 0] = observed[:, 1]
    previous, previous_tails = predicted.clone(), tails.clone()
    previous[-1, 0], previous_tails[-1, 0] = observed[-1], observed[-1, 1]
    steps = [observed_values({"quality": 0.0, **row["transition"]}, model.mean.device)
             for row in rows[:-1]]
    # Each backup depth takes its own maximum over STOP and CONTINUE before
    # the forward-view lambda mixture (Stop-RAG equations 3--11).
    for depth in range(1, count):
        current, current_tails = previous.clone(), previous_tails.clone()
        current[:, 0], current_tails[:, 0] = observed, observed[:, 1]
        utilities = prices.utility(previous, previous_tails).masked_fill(~mask, -torch.inf)
        chosen = utilities.argmax(-1)
        for index, step in enumerate(steps):
            action = chosen[index+1]
            current[index, 1] = step + previous[index+1, action]
            current_tails[index, 1] = step[1] + previous_tails[index+1, action]
            horizon = count-1-index
            if depth <= horizon:
                weight = trace_lambda ** (depth-1)
                if depth < horizon:
                    weight *= 1-trace_lambda
                target[index, 1] += weight * current[index, 1]
                if depth == horizon:
                    realised_seconds[index, 1] = current_tails[index, 1]
        previous, previous_tails = current, current_tails
    return features, target, realised_seconds, mask


def fit(trajectories, *, epochs=40, learning_rate=1e-3, trace_lambda=0.8,
        prices=ValuePrices(), device="cpu"):
    model = StopValueHead().to(device)
    features = torch.tensor([row["features"] for trajectory in trajectories
                             for row in trajectory["states"]], device=device)
    model.mean.copy_(features.mean(0))
    scale = features.std(0, unbiased=False)
    model.scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)))
    model.active.copy_((scale > 0).to(model.active.dtype))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for trajectory in trajectories:
            x, targets, seconds, mask = lambda_targets(model, trajectory, prices, trace_lambda)
            predicted, tails = model(x)
            # All decisions in a trajectory have equal weight. Cost units are
            # scaled by observed magnitude rather than by arbitrary loss prices.
            denominator = targets[..., 1:].mean((0, 1)).clamp_min(1.0)
            quality = (predicted[..., 0] - targets[..., 0]).square()
            costs = nn.functional.smooth_l1_loss(
                predicted[..., 1:] / denominator, targets[..., 1:] / denominator,
                reduction="none").mean(-1)
            residual = (seconds - tails) / seconds.mean().clamp_min(1.0)
            quantile = torch.maximum(0.95 * residual, -0.05 * residual)
            loss = (quality + costs + quantile)[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--seconds-price", type=float, default=0.0)
    parser.add_argument("--gpu-price", type=float, default=0.0)
    parser.add_argument("--host-price", type=float, default=0.0)
    parser.add_argument("--tail-price", type=float, default=0.0)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.trajectories).read_text().splitlines() if line]
    prices = ValuePrices(args.seconds_price, args.gpu_price, args.host_price, args.tail_price)
    model = fit([row for row in rows if row["split"] == "train"], epochs=args.epochs,
                trace_lambda=args.trace_lambda, prices=prices, device=args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
