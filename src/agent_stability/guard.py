"""Record per-turn token use and trip when growth becomes unstable."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from typing import Any

RATIO_THRESHOLD_ENV = "AGENT_STABILITY_RATIO_THRESHOLD"
BUDGET_GATE_ENV = "AGENT_STABILITY_BUDGET_GATE"

PATTERN_CONTEXT_SPIRAL = "context_spiral"
PATTERN_OUTPUT_RAMBLE = "output_ramble"
PATTERN_RETRY_STORM = "retry_storm"
PATTERN_POLICY_DRIFT = "policy_drift"
PATTERN_EARLY_EXPLOSION = "early_explosion"
PATTERN_BUDGET_EXHAUSTION = "budget_exhaustion"
PATTERN_SEARCH_ABORTED = "search_aborted"
PATTERN_BRANCH_PRUNED = "branch_pruned"
PATTERN_STABLE = "stable"

ACTION_ALLOW = "allow"
ACTION_WARN = "warn"
ACTION_TRIP = "trip"

_DEFAULT_DRIFT_THRESHOLD = 0.5


class StabilityError(RuntimeError):
    """Base for a stopped branch and a stopped search."""

    def __init__(self, pattern: str, step: int, message: str, edit: Any = None):
        super().__init__(message)
        self.pattern = pattern
        self.step = step
        self.edit = edit


class StabilityViolation(StabilityError):
    """This branch should stop. A search that owns it can continue."""


class SearchAborted(StabilityError):
    """The shared search ceiling is spent. Stop every branch."""


@dataclass
class Step:
    tokens: int
    input_tokens: int
    output_tokens: int
    split: bool
    errors: int
    tool_name: str | None
    arguments_hash: str | None
    similarity: float | None
    cumulative: int
    ratio: float
    input_ratio: float
    output_ratio: float
    energy: float


@dataclass(frozen=True)
class Decision:
    """What to do before spending tokens on the next model call."""

    action: str
    pattern: str | None
    message: str
    edit: Any = None

    @property
    def allowed(self) -> bool:
        return self.action == ACTION_ALLOW


class StabilityGuard:
    """Trip when turn size stays above a multiple of the warmup baseline.

    Call :meth:`preview` with the prompt size before the model call. It
    returns ``allow``, ``warn``, or ``trip`` and does not record anything.
    ``warn`` means one more escalating turn will trip.

    Pass ``input_tokens`` and ``output_tokens`` when you have them. A climbing
    prompt is a context spiral. A climbing completion with a flat prompt is
    an output ramble. ``tokens_used`` still records a single total.

    The first ``warmup_turns`` set the baseline and are not ratio-checked.
    Ratio, retry, and drift trips are impossible before step
    ``warmup_turns + window``. See :func:`earliest_trip_step`.

    Environment variables override the constructor when they are set:
    ``AGENT_STABILITY_RATIO_THRESHOLD`` and ``AGENT_STABILITY_BUDGET_GATE``.
    """

    def __init__(
        self,
        token_budget: int = 100_000,
        ratio_threshold: float = 2.0,
        window: int = 3,
        warmup_turns: int = 3,
        budget_gate: int = 8_000,
        lambda_: float = 1.0,
        drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
        price_per_million: float | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
    ):
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1:
            raise ValueError("token_budget must be an int >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        if warmup_turns < 1:
            raise ValueError("warmup_turns must be >= 1")
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")
        if not 0 <= drift_threshold <= 1:
            raise ValueError("drift_threshold must be between 0 and 1")

        self.token_budget = token_budget
        self.ratio_threshold = _env_float(RATIO_THRESHOLD_ENV, ratio_threshold)
        self.window = window
        self.warmup_turns = warmup_turns
        self.budget_gate = _env_int(BUDGET_GATE_ENV, budget_gate)
        self.lambda_ = lambda_
        self.drift_threshold = drift_threshold
        self.input_price_per_million = _validated_price(
            "input_price_per_million",
            input_price_per_million if input_price_per_million is not None else price_per_million,
        )
        self.output_price_per_million = _validated_price(
            "output_price_per_million",
            output_price_per_million if output_price_per_million is not None else price_per_million,
        )
        self.price_per_million = price_per_million

        if self.ratio_threshold <= 0:
            raise ValueError("ratio_threshold must be > 0")
        if self.budget_gate < 0:
            raise ValueError("budget_gate must be >= 0")

        self._steps: list[Step] = []
        self._baseline: float | None = None
        self._input_baseline: float | None = None
        self._output_baseline: float | None = None
        self.tripped = False
        self.trip_pattern: str | None = None
        self.tripped_at: int | None = None
        self.pruned = False
        self.branch_name: str | None = None
        self._search: Any = None

    def __enter__(self) -> StabilityGuard:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def attach_search(self, budget: Any, name: str) -> None:
        self._search = budget
        self.branch_name = name

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(self._steps)

    @property
    def total_tokens(self) -> int:
        if not self._steps:
            return 0
        return self._steps[-1].cumulative

    @property
    def total_input_tokens(self) -> int:
        return sum(step.input_tokens for step in self._steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(step.output_tokens for step in self._steps)

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def input_baseline(self) -> float | None:
        return self._input_baseline

    @property
    def output_baseline(self) -> float | None:
        return self._output_baseline

    @property
    def current_ratio(self) -> float:
        if not self._steps:
            return 0.0
        return self._steps[-1].ratio

    @property
    def peak_ratio(self) -> float:
        if not self._steps:
            return 0.0
        return max(step.ratio for step in self._steps)

    @property
    def peak_output_ratio(self) -> float:
        if not self._steps:
            return 0.0
        return max(step.output_ratio for step in self._steps)

    @property
    def earliest_trip_step(self) -> int:
        """First step on which a ratio, retry, or drift trip is possible."""
        return earliest_trip_step(self.warmup_turns, self.window)

    @property
    def status(self) -> str:
        if self.tripped or self.pruned:
            return ACTION_TRIP
        if self._search is not None and self._search.aborted:
            return ACTION_TRIP
        return self._allow_or_warn(self._steps)

    def preview(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        *,
        errors: int = 0,
        tool_name: str | None = None,
        tool_arguments: Any = None,
        similarity: float | None = None,
    ) -> Decision:
        """Judge the next call from its prompt size. Does not record a step."""
        if self.tripped or self.pruned:
            return self._build_decision(
                ACTION_TRIP,
                self.trip_pattern or PATTERN_BRANCH_PRUNED,
                f"guard already tripped ({self.trip_pattern}) at step {self.tripped_at}",
                self._steps,
            )
        if self._search is not None and self._search.aborted:
            return self._build_decision(
                ACTION_TRIP,
                PATTERN_SEARCH_ABORTED,
                "search is already aborted",
                self._steps,
            )

        input_tokens = _non_negative_int("input_tokens", input_tokens)
        output_tokens = _non_negative_int("output_tokens", output_tokens)
        projected = [_copy_step(step) for step in self._steps]
        projected.append(
            self._make_step(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                split=True,
                errors=errors,
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                similarity=similarity,
                cumulative=self.total_tokens + input_tokens + output_tokens,
            )
        )
        self._assign_growth(projected)

        shared_spent = self._search.spent if self._search is not None else 0
        incoming = input_tokens + output_tokens
        if self._search is not None and shared_spent + incoming >= self._search.token_budget:
            return self._build_decision(
                ACTION_TRIP,
                PATTERN_SEARCH_ABORTED,
                (
                    f"prompt of {incoming} tokens would spend the shared ceiling "
                    f"({shared_spent + incoming} of {self._search.token_budget})"
                ),
                projected,
            )

        pattern = self._matched_pattern(projected)
        if pattern is not None:
            step = projected[-1]
            return self._build_decision(
                ACTION_TRIP,
                pattern,
                f"{pattern} if sent: ratio {step.ratio:.2f}x, {step.cumulative} tokens",
                projected,
            )
        action = self._allow_or_warn(projected)
        if action == ACTION_WARN:
            message = "One more escalating turn will trip."
        else:
            message = "Prompt is inside the growth ratio."
        return self._build_decision(action, None, message, projected)

    def record_step(
        self,
        tokens_used: int | None = None,
        errors: int = 0,
        tool_name: str | None = None,
        similarity: float | None = None,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tool_arguments: Any = None,
    ) -> Decision:
        """Append one turn. Raises when the guard trips or the search aborts."""
        self._require_active()
        input_tokens, output_tokens, split = _resolve_tokens(tokens_used, input_tokens, output_tokens)
        errors = _non_negative_int("errors", errors)
        if similarity is not None and not 0 <= similarity <= 1:
            raise ValueError("similarity must be between 0 and 1")

        total = input_tokens + output_tokens
        step = self._make_step(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            split=split,
            errors=errors,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            similarity=similarity,
            cumulative=self.total_tokens + total,
        )
        self._steps.append(step)
        self._assign_growth(self._steps)

        if self._search is not None and self._search.spend(total):
            self.pruned = True
            self._raise_trip(PATTERN_SEARCH_ABORTED, SearchAborted)

        pattern = self._matched_pattern(self._steps)
        if pattern is not None:
            self.pruned = True
            self._raise_trip(pattern, StabilityViolation)

        action = self._allow_or_warn(self._steps)
        if action == ACTION_WARN:
            message = "One more escalating turn will trip."
        else:
            message = "Turn recorded."
        return self._build_decision(action, None, message, self._steps)

    def prune(self) -> None:
        """Stop this branch without aborting the search."""
        self.pruned = True

    def proposed_edit(self):
        from agent_stability.edit import propose_edit

        return propose_edit(
            pattern=self.trip_pattern,
            steps=self.steps,
            baseline=self.baseline,
            ratio_threshold=self.ratio_threshold,
            window=self.window,
            drift_threshold=self.drift_threshold,
            token_budget=self.token_budget,
            total_tokens=self.total_tokens,
        )

    def _require_active(self) -> None:
        if self._search is not None and self._search.aborted:
            raise SearchAborted(
                PATTERN_SEARCH_ABORTED,
                self.tripped_at or len(self._steps),
                "search is already aborted",
            )
        if self.pruned or self.tripped:
            raise StabilityViolation(
                self.trip_pattern or PATTERN_BRANCH_PRUNED,
                self.tripped_at or len(self._steps),
                f"guard already tripped ({self.trip_pattern}) at step {self.tripped_at}",
            )

    def _make_step(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        split: bool,
        errors: int,
        tool_name: str | None,
        tool_arguments: Any,
        similarity: float | None,
        cumulative: int,
    ) -> Step:
        return Step(
            tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            split=split,
            errors=errors,
            tool_name=tool_name,
            arguments_hash=_arguments_hash(tool_name, tool_arguments),
            similarity=similarity,
            cumulative=cumulative,
            ratio=1.0,
            input_ratio=1.0,
            output_ratio=1.0,
            energy=1.0 + self.lambda_ * errors,
        )

    def _assign_growth(self, steps: list[Step]) -> None:
        baseline, input_baseline, output_baseline = _baselines(steps, self.warmup_turns)
        own = steps is self._steps
        if own:
            self._baseline = baseline
            self._input_baseline = input_baseline
            self._output_baseline = output_baseline
        if baseline is None or input_baseline is None or output_baseline is None:
            return
        for step in steps:
            step.ratio = _ratio(step.tokens, baseline)
            step.input_ratio = _ratio(step.input_tokens, input_baseline)
            step.output_ratio = _ratio(step.output_tokens, output_baseline)
            step.energy = step.ratio + self.lambda_ * step.errors

    def _matched_pattern(self, steps: list[Step]) -> str | None:
        if len(steps) == self.warmup_turns and _early_explosion(
            steps, self.warmup_turns, self.ratio_threshold, self.budget_gate
        ):
            return PATTERN_EARLY_EXPLOSION
        past_warmup = len(steps) > self.warmup_turns
        spent = steps[-1].cumulative if steps else 0
        if past_warmup and spent >= self.budget_gate:
            if _policy_drift(steps, self.warmup_turns, self.window, self.drift_threshold):
                return PATTERN_POLICY_DRIFT
            if _retry_storm(steps, self.warmup_turns, self.window):
                return PATTERN_RETRY_STORM
            if _output_ramble(steps, self.warmup_turns, self.window, self.ratio_threshold):
                return PATTERN_OUTPUT_RAMBLE
            if _context_spiral(steps, self.warmup_turns, self.window, self.ratio_threshold):
                return PATTERN_CONTEXT_SPIRAL
        if spent >= self.token_budget:
            return PATTERN_BUDGET_EXHAUSTION
        return None

    def _allow_or_warn(self, steps: list[Step]) -> str:
        if self.window <= 1 or len(steps) <= self.warmup_turns:
            return ACTION_ALLOW
        spent = steps[-1].cumulative
        if spent < self.budget_gate:
            return ACTION_ALLOW
        streak = _escalating_streak(steps, self.warmup_turns, self.ratio_threshold)
        if streak == self.window - 1:
            return ACTION_WARN
        return ACTION_ALLOW

    def _build_decision(self, action: str, pattern: str | None, message: str, steps: list[Step]) -> Decision:
        from agent_stability.edit import propose_edit

        edit = None
        if action in {ACTION_WARN, ACTION_TRIP}:
            edit = propose_edit(
                pattern=pattern or PATTERN_CONTEXT_SPIRAL,
                steps=tuple(steps),
                baseline=self._baseline if steps is self._steps else _baselines(steps, self.warmup_turns)[0],
                ratio_threshold=self.ratio_threshold,
                window=self.window,
                drift_threshold=self.drift_threshold,
                token_budget=self.token_budget,
                total_tokens=steps[-1].cumulative if steps else 0,
            )
        return Decision(action=action, pattern=pattern, message=message, edit=edit)

    def _raise_trip(self, pattern: str, error_type: type[StabilityError]) -> None:
        self.tripped = True
        self.trip_pattern = pattern
        self.tripped_at = len(self._steps)
        step = self._steps[-1]
        edit = self.proposed_edit()
        raise error_type(
            pattern,
            self.tripped_at,
            (
                f"{pattern} at step {self.tripped_at}: "
                f"ratio {step.ratio:.2f}x, {step.cumulative} tokens"
            ),
            edit,
        )


def earliest_trip_step(warmup_turns: int = 3, window: int = 3) -> int:
    """First 1-based step on which ratio, retry, and drift are able to trip.

    Early explosion can trip at the end of warmup. A hard ceiling can trip
    on any step. A loop shorter than this cannot trip on growth ratio.
    """
    if warmup_turns < 1 or window < 1:
        raise ValueError("warmup_turns and window must be >= 1")
    return warmup_turns + window


def _baselines(steps: list[Step], warmup_turns: int) -> tuple[float | None, float | None, float | None]:
    if len(steps) < warmup_turns:
        return None, None, None
    warmup = steps[:warmup_turns]
    count = float(warmup_turns)
    return (
        sum(step.tokens for step in warmup) / count,
        sum(step.input_tokens for step in warmup) / count,
        sum(step.output_tokens for step in warmup) / count,
    )


def _trailing_turns(steps: list[Step], warmup_turns: int, window: int) -> list[Step] | None:
    if len(steps) < warmup_turns + window:
        return None
    return steps[-window:]


def _early_explosion(
    steps: list[Step],
    warmup_turns: int,
    ratio_threshold: float,
    budget_gate: int,
) -> bool:
    if warmup_turns < 2:
        return False
    tokens = [step.tokens for step in steps[:warmup_turns]]
    peak = max(tokens)
    if budget_gate > 0 and peak < budget_gate:
        return False
    rest = list(tokens)
    rest.remove(peak)
    if not rest:
        return False
    rest_mean = sum(rest) / float(len(rest))
    if rest_mean <= 0:
        return peak > 0
    return peak > ratio_threshold * rest_mean


def _context_spiral(steps: list[Step], warmup_turns: int, window: int, ratio_threshold: float) -> bool:
    recent = _trailing_turns(steps, warmup_turns, window)
    if recent is None:
        return False
    if all(step.split for step in recent):
        return all(step.input_ratio > ratio_threshold for step in recent)
    return all(step.ratio > ratio_threshold for step in recent)


def _output_ramble(steps: list[Step], warmup_turns: int, window: int, ratio_threshold: float) -> bool:
    recent = _trailing_turns(steps, warmup_turns, window)
    if recent is None or not all(step.split for step in recent):
        return False
    return all(step.output_ratio > ratio_threshold for step in recent) and all(
        step.input_ratio <= ratio_threshold for step in recent
    )


def _retry_storm(steps: list[Step], warmup_turns: int, window: int) -> bool:
    recent = _trailing_turns(steps, warmup_turns, window)
    if recent is None:
        return False
    hashes = [step.arguments_hash for step in recent]
    if any(value is None for value in hashes):
        return False
    if len(set(hashes)) != 1:
        return False
    return all(step.errors > 0 for step in recent)


def _policy_drift(
    steps: list[Step],
    warmup_turns: int,
    window: int,
    drift_threshold: float,
) -> bool:
    recent = _trailing_turns(steps, warmup_turns, window)
    if recent is None:
        return False
    scores = [step.similarity for step in recent]
    if any(score is None for score in scores):
        return False
    return all(score < drift_threshold for score in scores)  # type: ignore[operator]


def _escalating_streak(steps: list[Step], warmup_turns: int, ratio_threshold: float) -> int:
    streak = 0
    for step in reversed(steps[warmup_turns:]):
        if _is_escalating(step, ratio_threshold):
            streak += 1
        else:
            break
    return streak


def _is_escalating(step: Step, ratio_threshold: float) -> bool:
    if step.split and step.output_ratio > ratio_threshold and step.input_ratio <= ratio_threshold:
        return True
    ratio = step.input_ratio if step.split else step.ratio
    return ratio > ratio_threshold


def _resolve_tokens(
    tokens_used: int | None,
    input_tokens: int | None,
    output_tokens: int | None,
) -> tuple[int, int, bool]:
    if input_tokens is None and output_tokens is None:
        if tokens_used is None:
            raise ValueError("pass tokens_used or input_tokens and output_tokens")
        return _non_negative_int("tokens_used", tokens_used), 0, False
    input_count = _non_negative_int("input_tokens", 0 if input_tokens is None else input_tokens)
    output_count = _non_negative_int("output_tokens", 0 if output_tokens is None else output_tokens)
    if tokens_used is not None and tokens_used != input_count + output_count:
        raise ValueError("tokens_used must equal input_tokens + output_tokens")
    return input_count, output_count, True


def _arguments_hash(tool_name: str | None, tool_arguments: Any) -> str | None:
    if tool_name is None or tool_name == "":
        return None
    payload = json.dumps(_canonicalize(tool_arguments), sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{tool_name}\n{payload}".encode()).hexdigest()
    return digest[:16]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "[{":
            try:
                return _canonicalize(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped
        return stripped
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _copy_step(step: Step) -> Step:
    return replace(step)


def _ratio(tokens: int, baseline: float) -> float:
    if baseline <= 0:
        return float(tokens) if tokens else 1.0
    return tokens / baseline


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validated_price(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return float(value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc
