"""Measured lookup costs with explicit calibration and historical output priors.

The analytic fallback is diagnostic, not a hardware performance claim. Planners
must use a fixed policy whenever either compared estimate is uncalibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from bisect import bisect_left
from itertools import product
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping

from .schema import CacheQuote, CacheQuoteBatch, CostEstimate, PromptCandidate, SessionSnapshot


@dataclass(frozen=True)
class LookupPoint:
    prefix_tokens: int
    suffix_tokens: int
    concurrency: int
    elapsed_ms: float
    error_ms: float = 0.0
    host_load_tokens: int = 0
    source: str = "hardware"
    scope: str = "restore_prefill"


@dataclass(frozen=True)
class DecodePoint:
    context_tokens: int
    concurrency: int
    tpot_ms: float
    error_ms: float = 0.0
    source: str = "hardware"
    context_start: int | None = None
    context_end: int | None = None


@dataclass(frozen=True)
class OutputPrior:
    mean_tokens: int
    p90_tokens: int


def positive_prefill_features(prefix, suffix, load, concurrency):
    """Eight nonnegative runtime terms; fitting dependencies stay offline."""
    base = (1.0, float(suffix), float(suffix*prefix + suffix*suffix/2), float(load))
    pressure = max(0, concurrency-1)
    return base + tuple(pressure*value for value in base)


def output_length_prior(session: SessionSnapshot | None, *, global_mean: int = 128,
                        global_p90: int = 256, max_new_tokens: int = 512,
                        window: int = 4, shrinkage: float = 2.0) -> OutputPrior:
    """Use completed, committed outputs only; never accept current target length."""
    values = ([] if session is None else
              [len(t.assistant_output_ids) for t in session.raw_history[-window:]
               if t.assistant_output_ids])
    values = [min(max_new_tokens, max(1, length)) for length in values]
    mean = ((sum(values) + shrinkage * global_mean) / (len(values) + shrinkage)
            if values or shrinkage else global_mean)
    p90 = max(global_p90, sorted(values)[math.ceil(.9 * len(values)) - 1] if values else 1)
    return OutputPrior(min(max_new_tokens, max(1, round(mean))), min(max_new_tokens, max(1, p90)))


class CostModel:
    def __init__(self, prefill_points: Iterable[LookupPoint] = (),
                 decode_points: Iterable[DecodePoint] = (), *,
                 default_output_tokens: int = 128, p90_output_tokens: int = 256,
                 max_new_tokens: int = 512, paired_error_ms: float = 0.0,
                 allow_mock_calibration: bool = False,
                 max_normalized_distance: float = .5,
                 include_host_cost: bool = True, include_decode_cost: bool = True,
                 include_control_cost: bool = True, residual_alpha: float = .1,
                 prefill_estimator: str = "lookup", positive_prefill: Mapping | None = None) -> None:
        self.prefill_points = tuple(prefill_points)
        self.decode_points = tuple(decode_points)
        self.default_output_tokens = default_output_tokens
        self.p90_output_tokens = p90_output_tokens
        self.max_new_tokens = max_new_tokens
        self.paired_error_ms = paired_error_ms
        self.allow_mock_calibration = allow_mock_calibration
        self.max_normalized_distance = max_normalized_distance
        self.include_host_cost = include_host_cost
        self.include_decode_cost = include_decode_cost
        self.include_control_cost = include_control_cost
        self.residual_alpha = residual_alpha
        if prefill_estimator not in {"lookup", "positive_basis"}:
            raise ValueError("prefill_estimator must be lookup or positive_basis")
        self.prefill_estimator = prefill_estimator
        self.positive_prefill = dict(positive_prefill or {})
        self._bias_ms = 0.0
        self._past_residuals: list[float] = []
        self.calibration_metadata: dict = {}
        self.calibration_scope_compatible: bool | None = None
        self.calibration_scope_report: dict | None = None
        for p in self.prefill_points:
            if min(p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.elapsed_ms, p.error_ms) < 0:
                raise ValueError("cost measurements must be nonnegative")
        for p in self.decode_points:
            if min(p.context_tokens, p.tpot_ms, p.error_ms) < 0:
                raise ValueError("decode measurements must be nonnegative")
        valid_prefill = [p for p in self.prefill_points if self._source_valid(p.source)]
        prefill_scope = "first_token" if any(p.scope == "first_token" for p in valid_prefill) else "restore_prefill"
        scoped_prefill = [p for p in valid_prefill if p.scope == prefill_scope]
        lengths = [p.prefix_tokens+p.suffix_tokens for p in scoped_prefill]
        self._prefill_input_bounds = (min(lengths), max(lengths)) if lengths else None
        self._positive_geometry = None
        if prefill_estimator == "positive_basis":
            coefficients = self.positive_prefill.get("coefficients", ())
            if len(coefficients) != 8 or any(not math.isfinite(x) or x < 0 for x in coefficients):
                raise ValueError("positive prefill requires eight finite nonnegative fitted coefficients")
            for name in ("error_p95_ms", "error_p99_ms"):
                value = self.positive_prefill.get(name)
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError("positive prefill errors must be nonnegative finite values")
            cells = {(p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.concurrency) for p in scoped_prefill}
            if cells:
                axes = [sorted({coordinate[i] for coordinate in cells}) for i in range(4)]
                scales = [max(1, values[-1]-values[0]) for values in axes]
                self._positive_geometry = (cells, axes, scales)
        # Calibration points are fixed for this model's lifetime. Compile their
        # geometry once instead of rebuilding/sorting it for each output token.
        self._decode_valid_points = tuple(p for p in self.decode_points if self._source_valid(p.source))
        self._decode_by_concurrency: dict[int, list[DecodePoint]] = {}
        for point in self._decode_valid_points:
            self._decode_by_concurrency.setdefault(point.concurrency, []).append(point)
        self._decode_static = None
        if self._decode_valid_points and all(p.context_start is None for p in self._decode_valid_points):
            cells = {(p.context_tokens, p.concurrency): (p.tpot_ms, p.error_ms) for p in self._decode_valid_points}
            samples = tuple((key, *value) for key, value in cells.items())
            axes = [sorted({coordinate[i] for coordinate in cells}) for i in range(2)]
            scales = [max(1, values[-1] - values[0]) for values in axes]
            self._decode_static = (samples, (cells, axes, scales))

    @classmethod
    def from_measurements(cls, rows: Iterable[Mapping | object], *, allow_mock: bool = False,
                          **kwargs) -> "CostModel":
        groups: dict[tuple, list[float]] = {}
        for raw in rows:
            row = asdict(raw) if is_dataclass(raw) else dict(raw)
            source = row.get("source")
            if source not in {"hardware", "mock"}:
                continue
            if source != "hardware" and not allow_mock:
                continue
            phase = row.get("phase")
            elapsed = row.get("elapsed_ms")
            if elapsed is None or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
                continue
            concurrency = int(row.get("concurrency", 1))
            prompt = int(row.get("prompt_tokens", 0))
            prefix = int(row.get("prefix_tokens", 0))
            if phase in {"prefill", "restore_prefill", "first_token"}:
                load = int(row.get("host_load_tokens", 0))
                if int(row.get("transfer_bytes", 0)) and not load:
                    # A transferred-byte row without its KV-token conversion
                    # cannot calibrate a joint prefix/restore lookup safely.
                    continue
                scope = "first_token" if phase == "first_token" else "restore_prefill"
                key = ("prefill", prefix, max(0, prompt-prefix), concurrency, load, source, scope)
                groups.setdefault(key, []).append(float(elapsed))
            elif phase == "decode" and int(row.get("output_tokens", 0)) > 1:
                outputs = int(row["output_tokens"])
                context = prompt + outputs // 2
                key = ("decode", context, concurrency, source, prompt+1, prompt+outputs-1)
                groups.setdefault(key, []).append(float(elapsed)/(outputs-1))
        prefill, decode = [], []
        for key, values in groups.items():
            mean = statistics.mean(values)
            errors = sorted(abs(v-mean) for v in values)
            # Single samples cannot support a confident prediction.
            error = errors[math.ceil(.95*len(errors))-1] if len(values) > 1 else mean
            if key[0] == "prefill":
                _, prefix, suffix, concurrency, load, source, scope = key
                prefill.append(LookupPoint(prefix, suffix, concurrency, mean, error, load, source, scope))
            else:
                _, context, concurrency, source, start, end = key
                decode.append(DecodePoint(context, concurrency, mean, error, source, start, end))
        return cls(prefill, decode, allow_mock_calibration=allow_mock, **kwargs)

    def _source_valid(self, source: str) -> bool:
        return source == "hardware" or self.allow_mock_calibration

    def _interpolate(self, target, samples, *, prepared=None):
        """Multilinear cells when available; bounded local interpolation otherwise.

        Samples are (coordinates, cost, uncertainty). Outside measured support the
        returned estimate is diagnostic only and the planner uses its fixed policy.
        """
        if prepared is None:
            cells = {coordinates: (cost, error) for coordinates, cost, error in samples}
            if target in cells:
                return (*cells[target], True)
            axes = [sorted({coordinates[i] for coordinates in cells}) for i in range(len(target))]
            scales = [max(1, values[-1] - values[0]) for values in axes]
        else:
            cells, axes, scales = prepared
            if target in cells:
                return (*cells[target], True)
        covered = all(values[0] <= value <= values[-1] for value, values in zip(target, axes))
        if covered:
            bounds = []
            for value, values in zip(target, axes):
                index = bisect_left(values, value)
                bounds.append((values[index] if values[index] == value else values[index-1], values[index]))
            corners = tuple(product(*(tuple(dict.fromkeys(pair)) for pair in bounds)))
            if all(corner in cells for corner in corners):
                weights = []
                for corner in corners:
                    weight = 1.0
                    for value, coordinate, (low, high) in zip(target, corner, bounds):
                        if high != low:
                            weight *= (high-value)/(high-low) if coordinate == low else (value-low)/(high-low)
                    weights.append((weight, *cells[corner]))
                cost = sum(weight*value for weight, value, _ in weights)
                error = sum(weight*error for weight, _, error in weights)
                return cost, error, True
        # Nearest neighbors are only needed outside support or in a sparse cell.
        # Computing them for a full interpolation cell was the hot-path bottleneck.
        nearby = sorted((sum(abs(a-b)/s for a, b, s in zip(coordinates, target, scales)), cost, error)
                        for coordinates, cost, error in samples)
        if not covered:
            distance, cost, error = nearby[0]
            return cost, error + cost * distance, False
        selected = nearby[:min(4, len(nearby))]
        weights = [1/max(distance, 1e-9) for distance, _, _ in selected]
        cost = sum(w*sample[1] for w, sample in zip(weights, selected))/sum(weights)
        error = sum(w*(sample[2]+abs(sample[1]-cost)) for w, sample in zip(weights, selected))/sum(weights)
        return cost, error, nearby[0][0] <= self.max_normalized_distance

    def _prefill(self, prefix: int, suffix: int, load: int,
                 concurrency: int) -> tuple[float, float, bool]:
        # Separate P/U axis bounds can admit unmeasured total lengths. Both
        # estimators keep diagnostic values but require the observed n range.
        bounds = self._prefill_input_bounds
        length_supported = bounds is not None and bounds[0] <= prefix+suffix <= bounds[1]
        if self.prefill_estimator == "positive_basis":
            estimate = sum(a*b for a, b in zip(positive_prefill_features(prefix, suffix, load, concurrency),
                                               self.positive_prefill["coefficients"]))
            sufficient = self.positive_prefill.get("uncertainty_sufficient") is True
            error = self.positive_prefill.get("error_p99_ms")
            if not sufficient or error is None:
                return estimate, max(estimate, error or 0), False
            sources = self.positive_prefill.get("fit_sources")
            if sources is None:
                # Older positive artifacts may carry source provenance only in
                # the enclosing fit metadata. Missing provenance is not hardware.
                sources = self.calibration_metadata.get("measurement_sources")
            allowed = {"hardware", "mock"} if self.allow_mock_calibration else {"hardware"}
            sources_supported = bool(sources) and set(sources) <= allowed
            return estimate, error, (sources_supported and length_supported
                                     and self._positive_supported((prefix, suffix, load, concurrency)))
        points = [p for p in self.prefill_points if self._source_valid(p.source)]
        if not points:
            # Positive-coefficient structure captures long-prefix attention.
            estimate = .05 + .004*suffix + .0000002*(suffix*prefix+suffix*(suffix+1)/2) + .003*load
            return estimate, max(estimate, 1.0), False
        # Never mix a whole server TTFT sample with a queue-excluding phase.
        scope = "first_token" if any(p.scope == "first_token" for p in points) else "restore_prefill"
        points = [p for p in points if p.scope == scope]
        estimate, error, supported = self._interpolate((prefix, suffix, load, concurrency),
            [((p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.concurrency), p.elapsed_ms, p.error_ms)
             for p in points])
        return estimate, error, supported and length_supported

    def _positive_supported(self, target):
        """Preserve lookup geometry; _prefill also enforces the common n range.

        Only fixed geometry is consulted. Cost prediction is the eight-term dot
        product; this guard needs no neighbor cost interpolation or sorting.
        """
        if self._positive_geometry is None:
            return False
        cells, axes, scales = self._positive_geometry
        if target in cells:
            return True
        if not all(values[0] <= value <= values[-1] for value, values in zip(target, axes)):
            return False
        bounds = []
        for value, values in zip(target, axes):
            index = bisect_left(values, value)
            bounds.append((values[index] if values[index] == value else values[index-1], values[index]))
        corners = product(*(tuple(dict.fromkeys(pair)) for pair in bounds))
        if all(corner in cells for corner in corners):
            return True
        return min(sum(abs(a-b)/s for a, b, s in zip(coordinates, target, scales))
                   for coordinates in cells) <= self.max_normalized_distance

    def _decode(self, context: int, concurrency: int) -> tuple[float,float,bool]:
        points = self._decode_valid_points
        if not points:
            estimate = .01 + .00001*context
            return estimate, estimate, False
        if self._decode_static is not None:
            samples, prepared = self._decode_static
            return self._interpolate((context, concurrency), samples, prepared=prepared)
        # A measured interval creates an exact cell at the requested context.
        # Match the original reverse-distance overwrite order: nearest midpoint
        # wins, and equal-distance ties go to the later calibration point.
        exact = None
        distance = math.inf
        for point in self._decode_by_concurrency.get(concurrency, ()):
            if (point.context_tokens == context or (point.context_start is not None
                    and point.context_start <= context <= point.context_end)):
                current_distance = abs(point.context_tokens-context)
                if current_distance <= distance:
                    exact, distance = point, current_distance
        if exact is not None:
            return exact.tpot_ms, exact.error_ms, True
        # An observed mean TPOT covers the measured decode interval, not just
        # its midpoint. Preserve that support without extrapolating past it.
        cells = {}
        for point in sorted(points, key=lambda p: abs(p.context_tokens-context), reverse=True):
            coordinate = (context if point.context_start is not None
                          and point.context_start <= context <= point.context_end else point.context_tokens)
            cells[(coordinate, point.concurrency)] = (point.tpot_ms, point.error_ms)
        return self._interpolate((context, concurrency),
                                 [(key, *value) for key, value in cells.items()])

    def predict_native(self, candidate: PromptCandidate, quote: CacheQuote,
                       session: SessionSnapshot | None = None, *,
                       batch: CacheQuoteBatch | None = None, concurrency: int = 1,
                       control_ms: float = 0.0, queue_ms: float = 0.0) -> CostEstimate:
        # Candidate/quote identity is checked by the planner at the batch boundary.
        prior = output_length_prior(session, global_mean=self.default_output_tokens,
            global_p90=self.p90_output_tokens, max_new_tokens=self.max_new_tokens)
        n = len(candidate.input_ids)
        loaded = max(0, min(quote.host_only_load_tokens, quote.native_load_estimate))
        # A GPU-resident segment can follow a host-only gap. Once every missing
        # host segment is restored the usable boundary is the full contiguous
        # recoverable prefix, not gpu_prefix + bytes newly loaded.
        full_restore = quote.host_only_load_tokens > 0 and loaded == quote.host_only_load_tokens
        partial_restore = 0 < loaded < quote.host_only_load_tokens
        prefix = min(quote.match_limit, quote.recoverable_prefix_tokens if full_restore
                     else quote.ready_gpu_prefix_tokens)
        joint, joint_error, calibrated = self._prefill(prefix, max(0,n-prefix),
                            loaded if self.include_host_cost else 0, concurrency)
        control_ms = control_ms if self.include_control_cost else 0.0
        # first_token observations already include the server-side queue.
        if any(p.scope == "first_token" and self._source_valid(p.source) for p in self.prefill_points):
            queue_ms = 0.0
        decode = decode_error = risk_decode = 0.0
        for step in range(1, max(prior.mean_tokens,prior.p90_tokens) if self.include_decode_cost else 1):
            tpot, error, step_calibrated = self._decode(n+step, concurrency)
            if step < prior.mean_tokens:
                decode += tpot
                decode_error += error
            if step < prior.p90_tokens:
                risk_decode += tpot
            calibrated = calibrated and step_calibrated
        calibrated = calibrated and not quote.native_estimate_uncertain and not partial_restore
        calibrated = calibrated and self.calibration_scope_compatible is not False
        if batch is not None and not batch.host_quote_supported:
            calibrated = False
        cost = max(0.0, control_ms + queue_ms + joint + decode + self._bias_ms)
        capacity_risk = batch is not None and (n-prefix+loaded+prior.p90_tokens >
                            batch.free_gpu_tokens+batch.evictable_gpu_tokens)
        return CostEstimate(candidate.candidate_id, cost, control_ms, queue_ms, joint,
            decode, joint_error+decode_error, prior.mean_tokens, prior.p90_tokens,
            max(0.0, control_ms+queue_ms+joint+risk_decode+self._bias_ms), calibrated, capacity_risk)

    def paired_error(self, append: CostEstimate, rebuild: CostEstimate) -> float:
        # Online residuals are from completed prior requests. Without paired dev
        # residuals use a conservative sum, not a confidence claim.
        measured = (statistics.pstdev(self._past_residuals)
                    if len(self._past_residuals) > 1 else 0.0)
        return max(self.paired_error_ms, measured, append.error_ms+rebuild.error_ms)

    def observe(self, predicted: CostEstimate, actual_elapsed_ms: float) -> None:
        if not math.isfinite(actual_elapsed_ms) or actual_elapsed_ms < 0:
            raise ValueError("actual latency must be a finite nonnegative measurement")
        self._past_residuals.append(actual_elapsed_ms-predicted.cost)
        self._past_residuals = self._past_residuals[-128:]
        self._bias_ms += self.residual_alpha * (actual_elapsed_ms-predicted.cost)

    def check_calibration_scope(self, info) -> dict:
        """Compare a bound engine configuration once at startup/restart.

        Matching this identity does not prove the same GPU, drivers or topology.
        Legacy artifacts remain usable with an explicit unbound report. A known
        mismatch keeps diagnostic predictions but disables calibrated selection.
        """
        actual = asdict(info) if is_dataclass(info) else dict(info)
        manifest = self.calibration_metadata.get("profile_manifest") or {}
        manifests = self.calibration_metadata.get("profile_manifests") or ([manifest] if manifest else [])
        expected = {"fingerprint": manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint"),
                    "cache_config_hash": manifest.get("engine", {}).get("cache_config_hash")}
        compared = [key for key, value in expected.items() if value is not None]
        mismatches = [{"field": key, "expected": expected[key], "actual": actual.get(key)}
                      for key in compared if actual.get(key) != expected[key]]
        status = "mismatch" if mismatches else ("matching_scope" if len(compared) == 2 else "legacy_unbound")
        self.calibration_scope_compatible = False if mismatches else (True if status == "matching_scope" else None)
        self.calibration_scope_report = {"status": status,
            "matching_scope": "engine_configuration_only" if status == "matching_scope" else None,
            "compared_fields": compared, "mismatches": mismatches,
            "profile_run_id": manifest.get("run_id"),
            "profile_run_ids": [entry["run_id"] for entry in manifests if entry.get("run_id")],
            "gpu_hardware_verified": False,
            "measurement_binding": self.calibration_metadata.get("measurement_provenance", {}).get("binding_status", "not_provided")}
        return self.calibration_scope_report

    def save(self, path: str | Path, *, metadata: Mapping | None = None) -> None:
        """Freeze fitted points and priors. Runtime residual adaptation is not persisted."""
        settings = {name: getattr(self, name) for name in (
            "default_output_tokens", "p90_output_tokens", "max_new_tokens", "paired_error_ms",
            "max_normalized_distance", "include_host_cost", "include_decode_cost", "include_control_cost", "residual_alpha", "prefill_estimator")}
        payload = {"version": 1, "settings": settings, "prefill_points": [asdict(p) for p in self.prefill_points],
                   "decode_points": [asdict(p) for p in self.decode_points],
                   "metadata": dict(self.calibration_metadata if metadata is None else metadata),
                   "positive_prefill": self.positive_prefill or None}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "CostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        settings = {**payload["settings"], "positive_prefill": payload.get("positive_prefill"), **overrides}
        model = cls((LookupPoint(**p) for p in payload["prefill_points"]),
                    (DecodePoint(**p) for p in payload["decode_points"]), **settings)
        model.calibration_metadata = payload.get("metadata", {})
        return model


PiecewiseCostModel = CostModel
