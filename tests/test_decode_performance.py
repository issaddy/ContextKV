"""Equivalence to the original decode lookup, including sparse/interval support."""
from dataclasses import asdict
from itertools import product
import random

import pytest

from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.schema import CacheQuote


def legacy_interpolate(model, target, samples):
    # Frozen pre-optimization oracle. This intentionally retains the expensive
    # nearest sort so tests compare the actual former numerical policy.
    cells = {coordinates: (cost, error) for coordinates, cost, error in samples}
    if target in cells:
        return (*cells[target], True)
    axes = [sorted({coordinates[i] for coordinates in cells}) for i in range(len(target))]
    covered = all(values[0] <= value <= values[-1] for value, values in zip(target, axes))
    scales = [max(1, values[-1] - values[0]) for values in axes]
    nearby = sorted((sum(abs(a-b)/s for a, b, s in zip(coordinates, target, scales)), cost, error)
                    for coordinates, cost, error in samples)
    if not covered:
        distance, cost, error = nearby[0]
        return cost, error + cost * distance, False
    bounds = [(max(v for v in values if v <= value), min(v for v in values if v >= value))
              for value, values in zip(target, axes)]
    corners = tuple(product(*(tuple(dict.fromkeys(pair)) for pair in bounds)))
    if all(corner in cells for corner in corners):
        weights = []
        for corner in corners:
            weight = 1.0
            for value, coordinate, (low, high) in zip(target, corner, bounds):
                if high != low:
                    weight *= (high-value)/(high-low) if coordinate == low else (value-low)/(high-low)
            weights.append((weight, *cells[corner]))
        return (sum(w*v for w, v, _ in weights), sum(w*e for w, _, e in weights), True)
    selected = nearby[:min(4, len(nearby))]
    weights = [1/max(distance, 1e-9) for distance, _, _ in selected]
    cost = sum(w*sample[1] for w, sample in zip(weights, selected))/sum(weights)
    error = sum(w*(sample[2]+abs(sample[1]-cost)) for w, sample in zip(weights, selected))/sum(weights)
    return cost, error, nearby[0][0] <= model.max_normalized_distance


class LegacyCostModel(CostModel):
    def _interpolate(self, target, samples, **kwargs):
        return legacy_interpolate(self, target, samples)

    def _decode(self, context, concurrency):
        points = [p for p in self.decode_points if self._source_valid(p.source)]
        if not points:
            estimate = .01 + .00001*context
            return estimate, estimate, False
        cells = {}
        for point in sorted(points, key=lambda p: abs(p.context_tokens-context), reverse=True):
            coordinate = (context if point.context_start is not None
                          and point.context_start <= context <= point.context_end else point.context_tokens)
            cells[(coordinate, point.concurrency)] = (point.tpot_ms, point.error_ms)
        return self._interpolate((context, concurrency), [(key, *value) for key, value in cells.items()])


@pytest.mark.parametrize("intervals", [False, True])
@pytest.mark.parametrize("allow_mock", [False, True])
def test_random_sparse_dense_duplicate_and_interval_lookup_matches_legacy(intervals, allow_mock):
    rng = random.Random(410)
    for size in (0, 1, 9, 30):
        points = []
        for index in range(size):
            context = rng.randrange(0, 40)
            points.append(DecodePoint(context, rng.randrange(1, 4), rng.uniform(.1, 10), rng.random(),
                          "mock" if index % 3 == 0 else "hardware",
                          max(0, context-5) if intervals else None, context+5 if intervals else None))
        # Exact duplicate coordinates test last-point overwrite, with/without intervals.
        if points:
            last = points[-1]
            points.append(DecodePoint(last.context_tokens, last.concurrency, 99, 9, "hardware",
                                      last.context_start, last.context_end))
        model = CostModel(decode_points=points, allow_mock_calibration=allow_mock)
        legacy = LegacyCostModel(decode_points=points, allow_mock_calibration=allow_mock)
        for context in (-1, 0, 5, 10, 15, 20, 35, 40, 46):
            for concurrency in (1, 2, 2.5, 3, 5):
                actual, expected = model._decode(context, concurrency), legacy._decode(context, concurrency)
                assert actual[:2] == pytest.approx(expected[:2], rel=1e-12, abs=1e-12)
                assert actual[2] == expected[2]


def test_overlap_nearest_midpoint_and_later_point_tie_semantics():
    points = [DecodePoint(10, 1, 1, .1, context_start=5, context_end=15),
              DecodePoint(20, 1, 2, .2, context_start=15, context_end=25),
              DecodePoint(20, 1, 3, .3, context_start=15, context_end=25)]
    model = CostModel(decode_points=points)
    assert model._decode(15, 1) == (3, .3, True)
    assert model._decode(14, 1) == (1, .1, True)
    assert model._decode(26, 1)[2] is False


@pytest.mark.parametrize("intervals", [False, True])
def test_full_prediction_preserves_cost_risk_uncertainty_and_support(intervals):
    points = [DecodePoint(ctx, concurrency, 1 + ctx*.0001 + concurrency*.1, .01,
              context_start=ctx-100 if intervals else None, context_end=ctx+100 if intervals else None)
              for ctx in range(512, 4096, 512) for concurrency in (1, 2, 4, 8)]
    prefill = [LookupPoint(0, length, c, 1+length*.001) for length in (512, 1024, 2048, 4096) for c in (1, 2, 4, 8)]
    for n, concurrency, mean, p90 in ((1024, 1, 1, 1), (1024, 1, 128, 256),
                                     (513, 3, 16, 32), (4096, 1, 3, 4)):
        settings = dict(default_output_tokens=mean, p90_output_tokens=p90)
        model, legacy = CostModel(prefill, points, **settings), LegacyCostModel(prefill, points, **settings)
        from controller.schema import PromptCandidate
        candidate = PromptCandidate("prediction", "REBUILD", 0, (), (1,)*n, "unused",
                                    frozenset(), frozenset())
        quote = CacheQuote("prediction", "unused", n, n-1)
        actual = asdict(model.predict_native(candidate, quote, concurrency=concurrency, control_ms=1.5, queue_ms=2))
        expected = asdict(legacy.predict_native(candidate, quote, concurrency=concurrency, control_ms=1.5, queue_ms=2))
        for name in actual:
            if isinstance(actual[name], float):
                assert actual[name] == pytest.approx(expected[name], rel=1e-12, abs=1e-12)
            else:
                assert actual[name] == expected[name]
