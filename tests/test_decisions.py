import pytest

from agent_stability import (
    CORPUS,
    FailureReport,
    StabilityGuard,
    SearchAborted,
    SearchBudget,
    StabilityViolation,
    earliest_trip_step,
    run_example,
)
from agent_stability.cli import main
from agent_stability.guard import (
    PATTERN_CONTEXT_SPIRAL,
    PATTERN_OUTPUT_RAMBLE,
    PATTERN_RETRY_STORM,
    PATTERN_SEARCH_ABORTED,
)


def _guard(**kwargs) -> StabilityGuard:
    defaults = dict(token_budget=100_000, budget_gate=0, warmup_turns=3, window=3)
    defaults.update(kwargs)
    return StabilityGuard(**defaults)


def test_preview_trips_before_the_call_is_recorded():
    guard = _guard()
    for tokens in (1000, 1000, 1000, 2500, 4000):
        decision = guard.record_step(tokens_used=tokens)

    assert decision.action == "warn"
    assert guard.status == "warn"
    assert len(guard.steps) == 5

    preview = guard.preview(8000)
    assert preview.action == "trip"
    assert preview.pattern == PATTERN_CONTEXT_SPIRAL
    assert preview.edit is not None
    assert len(guard.steps) == 5


def test_preview_allows_a_flat_prompt():
    guard = _guard()
    for tokens in (1000, 1000, 1000):
        guard.record_step(tokens_used=tokens)

    decision = guard.preview(1100)
    assert decision.action == "allow"
    assert decision.allowed is True


def test_output_ramble_is_not_a_context_spiral():
    guard = _guard()
    turns = [{"input_tokens": 100, "output_tokens": 20}] * 3
    turns += [
        {"input_tokens": 100, "output_tokens": 80},
        {"input_tokens": 110, "output_tokens": 90},
        {"input_tokens": 100, "output_tokens": 120},
    ]
    with pytest.raises(StabilityViolation) as caught:
        for turn in turns:
            guard.record_step(**turn)

    assert caught.value.pattern == PATTERN_OUTPUT_RAMBLE
    report = FailureReport.from_guard(guard)
    assert "completion" in report.suggestions[0]


def test_climbing_prompt_stays_a_context_spiral_when_split():
    guard = _guard()
    with pytest.raises(StabilityViolation) as caught:
        for _ in range(3):
            guard.record_step(input_tokens=100, output_tokens=10)
        for amount in (400, 500, 900):
            guard.record_step(input_tokens=amount, output_tokens=10)

    assert caught.value.pattern == PATTERN_CONTEXT_SPIRAL


def test_input_and_output_prices_are_separate():
    guard = _guard(
        token_budget=10_000_000,
        input_price_per_million=1.0,
        output_price_per_million=4.0,
    )
    guard.record_step(input_tokens=1_000_000, output_tokens=500_000)
    report = FailureReport.from_guard(guard)

    assert report.cost == pytest.approx(1.0 + 2.0)
    assert report.total_input_tokens == 1_000_000
    assert report.total_output_tokens == 500_000


def test_argument_key_order_does_not_hide_a_retry():
    guard = _guard()
    for _ in range(3):
        guard.record_step(tokens_used=100)
    calls = [
        {"limit": 1, "q": "tide"},
        {"q": "tide", "limit": 1},
        '{"q": "tide", "limit": 1}',
    ]
    with pytest.raises(StabilityViolation) as caught:
        for arguments in calls:
            guard.record_step(
                tokens_used=150,
                errors=1,
                tool_name="search",
                tool_arguments=arguments,
            )

    assert caught.value.pattern == PATTERN_RETRY_STORM
    assert caught.value.edit is not None
    assert caught.value.edit.block_tool == "search"


def test_same_size_calls_with_different_arguments_stay_up():
    guard = _guard()
    for _ in range(3):
        guard.record_step(tokens_used=100)
    for query in ("a", "b", "c"):
        guard.record_step(
            tokens_used=110,
            errors=1,
            tool_name="search",
            tool_arguments={"q": query},
        )

    assert guard.tripped is False


def test_overshoot_confidence_grows_with_the_peak():
    mild = _guard()
    severe = _guard()
    mild_error = None
    severe_error = None
    try:
        for tokens in (1000, 1000, 1000, 2100, 2100, 2100):
            mild.record_step(tokens_used=tokens)
    except StabilityViolation as exc:
        mild_error = exc
    try:
        for tokens in (1000, 1000, 1000, 2500, 4000, 8000):
            severe.record_step(tokens_used=tokens)
    except StabilityViolation as exc:
        severe_error = exc

    assert mild_error is not None and severe_error is not None
    mild_report = FailureReport.from_guard(mild)
    severe_report = FailureReport.from_guard(severe)
    assert mild_report.confidence == pytest.approx(0.05)
    assert severe_report.confidence == pytest.approx(0.99)
    assert mild_report.confidence < severe_report.confidence


def test_ratio_trip_prunes_one_branch_and_leaves_the_search_running():
    budget = SearchBudget(50_000)
    first = budget.branch("first", budget_gate=0, warmup_turns=3, window=3)
    with pytest.raises(StabilityViolation) as caught:
        for tokens in (100, 100, 100, 400, 500, 800):
            first.record_step(tokens_used=tokens)

    assert not isinstance(caught.value, SearchAborted)
    assert first.pruned is True
    assert budget.aborted is False

    second = budget.branch("second", budget_gate=0, warmup_turns=3, window=3)
    second.record_step(tokens_used=100)
    assert second.tripped is False
    assert budget.spent == 100 + 100 + 100 + 400 + 500 + 800 + 100


def test_shared_ceiling_aborts_the_search():
    budget = SearchBudget(500)
    first = budget.branch("first", budget_gate=0, warmup_turns=1, window=5, ratio_threshold=100)
    first.record_step(tokens_used=300)
    second = budget.branch("second", budget_gate=0, warmup_turns=1, window=5, ratio_threshold=100)

    with pytest.raises(SearchAborted) as caught:
        second.record_step(tokens_used=300)

    assert caught.value.pattern == PATTERN_SEARCH_ABORTED
    assert budget.aborted is True
    with pytest.raises(SearchAborted):
        second.record_step(tokens_used=10)


def test_labeled_corpus_matches_and_publishes_the_trip_region():
    assert earliest_trip_step() == 6
    assert len(CORPUS) >= 5
    for example in CORPUS:
        assert run_example(example) == example.label, example.name

    short = next(example for example in CORPUS if example.name == "short_loop")
    assert len(short.turns) < earliest_trip_step(short.warmup_turns, short.window)


def test_region_command(capsys):
    assert main(["region", "--warmup", "3", "--window", "3"]) == 0
    output = capsys.readouterr().out
    assert "step 6" in output
    assert "shorter than 6" in output
