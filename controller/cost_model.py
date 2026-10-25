"""Measured lookup costs with explicit calibration and historical output priors.

The default planner requires calibrated costs. An explicitly selected native
workload estimator can compare measured execution with analytical waiting;
those estimates retain a separate eligibility flag and are not called calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from collections import deque
from bisect import bisect_left
from functools import lru_cache
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
                 prefill_estimator: str = "lookup", positive_prefill: Mapping | None = None,
                 uncertainty_mode: str | None = None, output_length_model=None,
                 stage_model: Mapping | None = None, admission_estimator: str = "strict") -> None:
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
        self.uncertainty_mode = (self.positive_prefill.get("uncertainty_mode", "global")
                                 if uncertainty_mode is None else uncertainty_mode)
        if self.uncertainty_mode not in {"global", "conditional"}:
            raise ValueError("uncertainty_mode must be global or conditional")
        self.output_length_model = output_length_model
        self.stage_model = dict(stage_model or {})
        self.admission_estimator = admission_estimator
        if admission_estimator not in {"strict", "native_workload", "native_budget"}:
            raise ValueError("unknown admission estimator")
        if self.stage_model and admission_estimator not in {"native_workload", "native_budget"}:
            raise ValueError("forward-stage artifact needs explicit admission_estimator=native_workload or native_budget")
        # An unfitted model is constructed before loading a configured artifact.
        if admission_estimator in {"native_workload", "native_budget"} and self.prefill_points and not self.stage_model:
            raise ValueError("native workload estimators need a forward-stage artifact; total TTFT would double-count waiting")
        if not self.stage_model and any(p.scope == "forward_first" for p in self.prefill_points):
            raise ValueError("forward_first points require a stage model for pre-forward waiting")
        self._idle_pre_forward_samples = ()
        if self.stage_model:
            if (self.stage_model.get("version") != 1 or
                    any(p.scope != "forward_first" for p in self.prefill_points)):
                raise ValueError("stage artifact must contain only forward_first points")
            idle = [cell for cell in self.stage_model["pre_forward"]["cells"] if cell["features"][3] == 1]
            if not idle:
                raise ValueError("stage artifact needs observed single-request pre-forward overhead")
            # Idle pre-forward includes native preparation and any host restore.
            # Pool identity is checked at startup. Output budgets do not affect
            # this single-request setup; aggregate them at the same P/U/L cell.
            groups = {}
            for cell in idle:
                groups.setdefault(tuple(cell["features"][:3]), []).append(cell)
            self._idle_pre_forward_samples = tuple((key,
                sum(c["mean_ms"]*c["requests"] for c in cells)/sum(c["requests"] for c in cells), 0.)
                for key, cells in groups.items())
        self._bias_ms = 0.0
        self._past_residuals: list[float] = []
        self.sampling_scope_compatible = True
        self.sampling_scope_report = None
        self._sampling_prior_override = None
        self._development_lengths = self.stage_model.get("development_output_lengths", {}).get("lengths", ())
        self._observed_output_lengths = deque(maxlen=128)
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
        prefill_scope = self._prefill_scope(valid_prefill)
        scoped_prefill = [p for p in valid_prefill if p.scope == prefill_scope]
        self._prefill_samples = tuple(((p.prefix_tokens, p.suffix_tokens, p.host_load_tokens, p.concurrency),
                                      p.elapsed_ms, p.error_ms) for p in scoped_prefill)
        self._prefill_geometry = self._geometry(self._prefill_samples)
        self._idle_pre_forward_geometry = self._geometry(self._idle_pre_forward_samples)
        self._prefill_includes_queue = any(p.scope == "first_token" for p in valid_prefill)
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
        self._decode_static = self._decode_midpoints = None
        self._decode_intervals = tuple(p for p in self._decode_valid_points if p.context_start is not None)
        if self._decode_valid_points:
            cells = {(p.context_tokens, p.concurrency): (p.tpot_ms, p.error_ms) for p in self._decode_valid_points}
            samples = tuple((key, *value) for key, value in cells.items())
            axes = [sorted({coordinate[i] for coordinate in cells}) for i in range(2)]
            scales = [max(1, values[-1] - values[0]) for values in axes]
            self._decode_midpoints = (samples, (cells, axes, scales))
            if not self._decode_intervals:
                self._decode_static = self._decode_midpoints
        # Fixed calibration makes (context length, concurrency) a pure lookup.
        # Mean/P90 workload scenarios and the two candidates repeatedly traverse
        # the same decode intervals. Cache derived values, not request contents.
        self._decode_cached = lru_cache(maxsize=8192)(self._decode_uncached)

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

    @staticmethod
    def _geometry(samples):
        if not samples:
            return None
        cells = {coordinates: (cost, error) for coordinates, cost, error in samples}
        axes = [sorted({coordinate[i] for coordinate in cells}) for i in range(len(samples[0][0]))]
        return cells, axes, [max(1, values[-1] - values[0]) for values in axes]

    def _interpolate(self, target, samples, *, prepared=None, max_distance=None):
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
        limit = self.max_normalized_distance if max_distance is None else max_distance
        return cost, error, nearby[0][0] <= limit

    def _positive_error(self, concurrency, load):
        error = self.positive_prefill.get("error_p99_ms")
        if self.uncertainty_mode == "conditional":
            table = self.positive_prefill.get("conditional_uncertainty", {})
            minimum = table.get("minimum_feature_cells", 20)
            host_load = load > 0
            for level in (host_load, None):
                for row in table.get("strata", ()):
                    if (row["concurrency"] == concurrency and row["host_load"] is level
                            and host_load in row["host_load_states"] and row["feature_cells"] >= minimum):
                        return row["error_p99_ms"]
        return error

    def _prefill(self, prefix: int, suffix: int, load: int,
                 concurrency: int, *, uncertainty_load: int | None = None) -> tuple[float, float, bool]:
        # Separate P/U axis bounds can admit unmeasured total lengths. Both
        # estimators keep diagnostic values but require the observed n range.
        bounds = self._prefill_input_bounds
        length_supported = bounds is not None and bounds[0] <= prefix+suffix <= bounds[1]
        if self.prefill_estimator == "positive_basis":
            estimate = sum(a*b for a, b in zip(positive_prefill_features(prefix, suffix, load, concurrency),
                                               self.positive_prefill["coefficients"]))
            sufficient = self.positive_prefill.get("uncertainty_sufficient") is True
            error = self._positive_error(concurrency, load if uncertainty_load is None else uncertainty_load)
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
        if not self._prefill_samples:
            # Positive-coefficient structure captures long-prefix attention.
            estimate = .05 + .004*suffix + .0000002*(suffix*prefix+suffix*(suffix+1)/2) + .003*load
            return estimate, max(estimate, 1.0), False
        estimate, error, supported = self._interpolate((prefix, suffix, load, concurrency),
            self._prefill_samples, prepared=self._prefill_geometry)
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
        return self._decode_cached(context, concurrency, self.max_normalized_distance)

    def _decode_uncached(self, context: int, concurrency: int, max_distance: float) -> tuple[float,float,bool]:
        points = self._decode_valid_points
        if not points:
            estimate = .01 + .00001*context
            return estimate, estimate, False
        if self._decode_static is not None:
            samples, prepared = self._decode_static
            return self._interpolate((context, concurrency), samples, prepared=prepared, max_distance=max_distance)
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
        if not any(point.context_start <= context <= point.context_end for point in self._decode_intervals):
            # Between measured intervals every coordinate is its fixed midpoint.
            # Reuse that geometry instead of sorting/rebuilding it per token.
            samples, prepared = self._decode_midpoints
            return self._interpolate((context, concurrency), samples, prepared=prepared, max_distance=max_distance)
        # An observed mean TPOT covers the measured decode interval, not just
        # its midpoint. Preserve that support without extrapolating past it.
        cells = {}
        for point in sorted(points, key=lambda p: abs(p.context_tokens-context), reverse=True):
            coordinate = (context if point.context_start is not None
                          and point.context_start <= context <= point.context_end else point.context_tokens)
            cells[(coordinate, point.concurrency)] = (point.tpot_ms, point.error_ms)
        return self._interpolate((context, concurrency),
                                 [(key, *value) for key, value in cells.items()], max_distance=max_distance)

    @staticmethod
    def _prefill_scope(points):
        for scope in ("forward_first", "first_token", "restore_prefill"):
            if any(p.scope == scope for p in points):
                return scope
        return "restore_prefill"

    def _workload_forward(self, input_tokens, resident_gpu_tokens, concurrency):
        prefix = min(max(0, input_tokens-1), resident_gpu_tokens)
        value, _, supported = self._prefill(prefix, input_tokens-prefix, 0, concurrency)
        return value, supported

    def _workload_decode(self, context_tokens, output_tokens, concurrency):
        total, supported = 0.0, True
        for step in range(output_tokens):
            value, _, valid = self._decode(context_tokens+step, concurrency)
            total += value
            supported = supported and valid
        return total, supported

    def _queued_workload_forward(self, input_tokens, device_tokens, host_tokens, concurrency):
        prefix = min(max(0, input_tokens-1), device_tokens+host_tokens)
        loaded = host_tokens if self.include_host_cost else 0
        forward, _, supported = self._prefill(prefix, input_tokens-prefix, loaded, concurrency,
                                              uncertainty_load=host_tokens)
        setup, _, setup_supported = self._interpolate(
            (prefix, input_tokens-prefix, loaded), self._idle_pre_forward_samples,
            prepared=self._idle_pre_forward_geometry)
        return setup+forward, supported and setup_supported

    def predict_native(self, candidate: PromptCandidate, quote: CacheQuote,
                       session: SessionSnapshot | None = None, *,
                       batch: CacheQuoteBatch | None = None, concurrency: int = 1,
                       control_ms: float = 0.0, queue_ms: float = 0.0) -> CostEstimate:
        # Candidate/quote identity is checked by the planner at the batch boundary.
        mean, p90 = self.effective_output_prior
        prior = output_length_prior(session, global_mean=mean,
            global_p90=p90, max_new_tokens=self.max_new_tokens)
        length_prediction = None
        if self.output_length_model is not None:
            length_prediction = self.output_length_model.predict_prior(
                candidate, session, self.max_new_tokens, prior)
            prior = OutputPrior(length_prediction.mean_tokens, length_prediction.p90_tokens)
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
                            loaded if self.include_host_cost else 0, concurrency, uncertainty_load=loaded)
        control_ms = control_ms if self.include_control_cost else 0.0
        # first_token observations already include the server-side queue.
        if self._prefill_includes_queue:
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
        components_supported = (calibrated and self.calibration_scope_compatible is not False
                                and self.sampling_scope_compatible)
        calibrated = components_supported and not quote.native_estimate_uncertain and not partial_restore
        if batch is not None and not batch.host_quote_supported:
            calibrated = False
        wait, selection_eligible = None, None
        if self.stage_model:
            from .admission import estimate_native_budget_wait, estimate_wait
            setup_ms, _, setup_supported = self._interpolate(
                (prefix, max(0,n-prefix), loaded if self.include_host_cost else 0),
                self._idle_pre_forward_samples, prepared=self._idle_pre_forward_geometry)
            wait_args = dict(candidate_input_tokens=n, resident_gpu_tokens=max(0, prefix-loaded),
                batch=batch, forward_fn=self._workload_forward, decode_fn=self._workload_decode,
                development_lengths=tuple(self._observed_output_lengths) or self._development_lengths,
                idle_ms=setup_ms)
            if self.admission_estimator == "native_budget":
                profile = self.calibration_metadata.get("profile_manifest") or {}
                # This observation describes only the current contiguous GPU
                # match, not resident tails after a host gap or future unlocks.
                initial_pin = (quote.evictable_gpu_prefix_tokens
                    if not quote.host_only_load_tokens
                    and not quote.inflight_h2d_tokens and not quote.inflight_d2h_tokens
                    and wait_args["resident_gpu_tokens"] == quote.ready_gpu_prefix_tokens
                    else None)
                wait = estimate_native_budget_wait(**wait_args, max_new_tokens=self.max_new_tokens,
                    page_size=(profile.get("cache_config") or {}).get("page_size"),
                    queued_forward_fn=self._queued_workload_forward, initial_pinned_tokens=initial_pin)
            else:
                wait = estimate_wait(**wait_args, output_mean=prior.mean_tokens, output_p90=prior.p90_tokens)
            queue_ms = wait.mean_ms
            reasons = set(quote.uncertainty_reasons)
            known_native_state = (not quote.inflight_h2d_tokens and not quote.inflight_d2h_tokens
                and reasons <= {"capacity_pressure"}
                and (not quote.native_estimate_uncertain or reasons == {"capacity_pressure"}))
            selection_eligible = bool(components_supported and self.calibration_scope_compatible is True
                and setup_supported and wait.supported and known_native_state and not partial_restore
                and batch is not None and batch.host_quote_supported)
            # Empirical forward/decode support is not calibration of the new
            # heterogeneous workload forecast. Selection is explicit and opt-in.
            calibrated = False
        cost = max(0.0, control_ms + queue_ms + joint + decode + self._bias_ms)
        capacity_risk = batch is not None and (n-prefix+loaded+prior.p90_tokens >
                            batch.free_gpu_tokens+batch.evictable_gpu_tokens)
        return CostEstimate(candidate.candidate_id, cost, control_ms, queue_ms, joint,
            decode, joint_error+decode_error, prior.mean_tokens, prior.p90_tokens,
            max(0.0, control_ms+(wait.upper_ms if wait else queue_ms)+joint+risk_decode+self._bias_ms), calibrated, capacity_risk,
            # Length residuals have a different meaning from timing errors.
            # Keep them diagnostic; mean/p90 affect expected/risk decode cost.
            output_length_error_tokens=length_prediction.error_tokens if length_prediction else None,
            output_length_model=length_prediction.model_kind if length_prediction else "shared",
            output_length_in_range=length_prediction.within_training_range if length_prediction else None,
            selection_eligible=selection_eligible, waiting_estimate_kind=wait.model_kind if wait else None,
            waiting_upper_ms=wait.upper_ms if wait else None,
            waiting_reason=wait.reason if wait else None,
            timing_components_supported=components_supported if self.stage_model else None)

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

    def observe_output(self, output_tokens: int) -> None:
        """Completed serving lengths for native wait forecasts, not timing fitting.

        Keep capped completions too: these are served lengths, not estimates of
        uncensored natural EOS length. Native admission still uses its requested
        output reservation, independently of the empirical service distribution.
        """
        if output_tokens > 0:
            self._observed_output_lengths.append(output_tokens)

    @property
    def effective_output_prior(self) -> tuple[int, int]:
        return self._sampling_prior_override or (self.default_output_tokens, self.p90_output_tokens)

    def configure_sampling(self, sampling, *, fallback_mean: int, fallback_p90: int,
                           system_prompt: str | None = None) -> dict:
        """Bind request policy once, without changing measured execution points.

        Server wall intervals include sampler work. Cross-recipe timing reuse is
        diagnostic, not validated calibration. The output distribution has its
        own source: a profiler's fixed-output requests do not fit EOS lengths.
        """
        from .sampling import SamplingRecipe, bound_recipe, request_sampling
        recipe = sampling or SamplingRecipe()
        manifest = self.calibration_metadata.get("profile_manifest") or {}
        manifests = self.calibration_metadata.get("profile_manifests") or [manifest]
        timing_recipes = [bound_recipe(entry.get("sampling_recipe")) for entry in manifests]
        # Preserve old standard-greedy loading without inventing source proof.
        legacy_compatible = recipe == SamplingRecipe()
        self.sampling_scope_compatible = all(
            value == recipe or value is None and legacy_compatible for value in timing_recipes)
        prior_recipe = bound_recipe(self.calibration_metadata.get("output_prior_sampling"))
        prior_matches = recipe == prior_recipe or prior_recipe is None and legacy_compatible
        prior_prompt = self.calibration_metadata.get("output_prior_system_prompt")
        prompt_matches = system_prompt is None or prior_prompt == system_prompt
        self._sampling_prior_override = None if prior_matches and prompt_matches else (fallback_mean, fallback_p90)
        lengths = self.stage_model.get("development_output_lengths", {})
        length_recipe = bound_recipe(lengths.get("sampling_recipe", self.calibration_metadata.get("output_prior_sampling")))
        lengths_match = recipe == length_recipe or length_recipe is None and legacy_compatible
        length_prompt_matches = system_prompt is None or lengths.get("system_prompt", prior_prompt) == system_prompt
        self._development_lengths = lengths.get("lengths", ()) if lengths_match and length_prompt_matches else ()
        self._observed_output_lengths.clear()
        self.sampling_scope_report = {
            "requested_sampling": request_sampling(sampling),
            "timing_recipes": [value.record() if value is not None else None for value in timing_recipes],
            "timing_binding": "legacy_unbound" if any(value is None for value in timing_recipes) else "explicit",
            "timing_compatible": self.sampling_scope_compatible,
            "timing_scope": "measured_server_wall_including_sampler; cross_recipe_unvalidated",
            "output_prior_recipe": prior_recipe.record() if prior_recipe is not None else None,
            "output_prior_status": ("legacy_unbound_compatibility" if prior_recipe is None else "matching_recipe")
                if prior_matches and prompt_matches else "configured_prior_fallback",
            "output_prior_prompt_status": ("not_checked" if system_prompt is None else
                "matching_prompt" if prompt_matches else "unbound_or_different_prompt"),
            "development_lengths_status": ("legacy_unbound_compatibility" if length_recipe is None else "matching_recipe")
                if lengths_match and length_prompt_matches else
                "disabled_recipe_mismatch" if not lengths_match else "disabled_prompt_mismatch",
            "legacy_missing_binding": "unknown; compatibility exception only for standard greedy serving",
        }
        return self.sampling_scope_report

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
        if self.sampling_scope_report is not None:
            self.calibration_scope_report["sampling"] = self.sampling_scope_report
        return self.calibration_scope_report

    def save(self, path: str | Path, *, metadata: Mapping | None = None) -> None:
        """Freeze fitted points and priors. Runtime residual adaptation is not persisted."""
        settings = {name: getattr(self, name) for name in (
            "default_output_tokens", "p90_output_tokens", "max_new_tokens", "paired_error_ms",
            "max_normalized_distance", "include_host_cost", "include_decode_cost", "include_control_cost", "residual_alpha", "prefill_estimator", "uncertainty_mode", "admission_estimator")}
        payload = {"version": 1, "settings": settings, "prefill_points": [asdict(p) for p in self.prefill_points],
                   "decode_points": [asdict(p) for p in self.decode_points],
                   "metadata": dict(self.calibration_metadata if metadata is None else metadata),
                   "positive_prefill": self.positive_prefill or None,
                   "stage_model": self.stage_model or None}
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "CostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if overrides.get("uncertainty_mode") is None:
            overrides.pop("uncertainty_mode", None)
        settings = {**payload["settings"], "positive_prefill": payload.get("positive_prefill"),
                    "stage_model": payload.get("stage_model"), **overrides}
        model = cls((LookupPoint(**p) for p in payload["prefill_points"]),
                    (DecodePoint(**p) for p in payload["decode_points"]), **settings)
        model.calibration_metadata = payload.get("metadata", {})
        return model


PiecewiseCostModel = CostModel
