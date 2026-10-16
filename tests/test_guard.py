import pytest

from agent_stability import FailureReport, StabilityGuard, StabilityViolation
from agent_stability.guard import (
    PATTERN_BUDGET_EXHAUSTION,
    PATTERN_CONTEXT_SPIRAL,
    PATTERN_EARLY_EXPLOSION,
    PATTERN_POLICY_DRIFT,
    PATTERN_RETRY_STORM,
    PATTERN_STABLE,
)


def _guard(**kwargs) -> StabilityGuard:
    defaults = dict(token_budget=100_000, budget_gate=0, warmup_turns=3, window=3)
    defaults.update(kwargs)
    return StabilityGuard(**defaults)


def _record_until_trip(guard: StabilityGuard, turns: list[dict]) -> StabilityViolation | None:
    for turn in turns:
        try:
            guard.record_step(**turn)
        except StabilityViolation as exc:
            return exc
    return None


def test_linear_growth_does_not_trip():
    guard = _guard()
    with guard:
        for tokens in (1000, 1100, 1200, 1300, 1400, 1500):
            guard.record_step(tokens_used=tokens)

    assert guard.tripped is False
    assert guard.baseline == pytest.approx(1100)
    assert guard.current_ratio == pytest.approx(1500 / 1100)
    report = FailureReport.from_guard(guard)
    assert report.pattern == PATTERN_STABLE
    assert report.tripped is False


def test_context_spiral_trips_after_window():
    guard = _guard()
    exc = _record_until_trip(
        guard,
        [{"tokens_used": tokens} for tokens in (1000, 1000, 1000, 2500, 4000, 8000)],
    )

    assert exc is not None
    assert exc.pattern == PATTERN_CONTEXT_SPIRAL
    assert exc.step == 6
    assert guard.peak_ratio == pytest.approx(8.0)
    assert len(guard.steps) == 6

    with pytest.raises(StabilityViolation):
        guard.record_step(tokens_used=9000)
    assert len(guard.steps) == 6


def test_budget_gate_holds_a_spiral_that_has_not_spent_enough():
    guard = _guard(budget_gate=1_000)
    for tokens in (10, 10, 10, 40, 50, 80):
        guard.record_step(tokens_used=tokens)

    assert guard.tripped is False
    assert guard.total_tokens == 200


def test_flat_run_that_hits_the_ceiling_is_budget_exhaustion():
    guard = _guard(token_budget=650)
    exc = _record_until_trip(
        guard,
        [{"tokens_used": tokens} for tokens in (100, 100, 100, 110, 120, 130)],
    )

    assert exc is not None
    assert exc.pattern == PATTERN_BUDGET_EXHAUSTION
    assert guard.total_tokens == 660


def test_early_explosion_on_an_oversized_opening_turn():
    guard = _guard(budget_gate=8_000)
    exc = _record_until_trip(
        guard,
        [{"tokens_used": tokens} for tokens in (12_000, 200, 200)],
    )

    assert exc is not None
    assert exc.pattern == PATTERN_EARLY_EXPLOSION
    assert exc.step == 3


def test_small_warmup_spike_stays_under_the_gate():
    guard = _guard(budget_gate=8_000)
    for tokens in (100, 100, 5_000):
        guard.record_step(tokens_used=tokens)

    assert guard.tripped is False


def test_retry_storm_wins_over_a_high_ratio():
    guard = _guard()
    turns = [{"tokens_used": 100}] * 3
    turns += [
        {"tokens_used": 250, "errors": 1, "tool_name": "search"},
        {"tokens_used": 250, "errors": 1, "tool_name": "search"},
        {"tokens_used": 250, "errors": 1, "tool_name": "search"},
    ]
    exc = _record_until_trip(guard, turns)

    assert exc is not None
    assert exc.pattern == PATTERN_RETRY_STORM


def test_varied_arguments_are_not_a_storm():
    guard = _guard()
    turns = [{"tokens_used": 100}] * 3
    turns += [
        {"tokens_used": 250, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "a"}},
        {"tokens_used": 400, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "b"}},
        {"tokens_used": 800, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "c"}},
    ]
    exc = _record_until_trip(guard, turns)

    assert exc is not None
    assert exc.pattern == PATTERN_CONTEXT_SPIRAL


def test_policy_drift_from_caller_supplied_similarity():
    guard = _guard(drift_threshold=0.5)
    turns = [{"tokens_used": 100, "similarity": 0.9}] * 3
    turns += [{"tokens_used": 120, "similarity": 0.2}] * 3
    exc = _record_until_trip(guard, turns)

    assert exc is not None
    assert exc.pattern == PATTERN_POLICY_DRIFT


def test_environment_overrides_constructor_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_STABILITY_RATIO_THRESHOLD", "4")
    monkeypatch.setenv("AGENT_STABILITY_BUDGET_GATE", "50")
    guard = StabilityGuard(ratio_threshold=2.0, budget_gate=0)

    assert guard.ratio_threshold == 4.0
    assert guard.budget_gate == 50


def test_rejects_bad_inputs():
    guard = _guard()
    with pytest.raises(ValueError):
        guard.record_step(tokens_used=-1)
    with pytest.raises(ValueError):
        guard.record_step(tokens_used=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        guard.record_step(tokens_used=10, similarity=1.5)
    with pytest.raises(ValueError):
        StabilityGuard(token_budget=0)


def test_zero_baseline_does_not_divide_by_zero():
    guard = _guard(ratio_threshold=2.0)
    exc = _record_until_trip(
        guard,
        [{"tokens_used": tokens} for tokens in (0, 0, 0, 5, 5, 5)],
    )

    assert exc is not None
    assert exc.pattern == PATTERN_CONTEXT_SPIRAL
