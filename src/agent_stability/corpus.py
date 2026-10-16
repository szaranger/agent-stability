"""Labeled trajectories and the step on which a ratio trip becomes possible."""

from __future__ import annotations

from dataclasses import dataclass

from agent_stability.guard import (
    PATTERN_BUDGET_EXHAUSTION,
    PATTERN_CONTEXT_SPIRAL,
    PATTERN_OUTPUT_RAMBLE,
    PATTERN_RETRY_STORM,
    PATTERN_STABLE,
    StabilityGuard,
    StabilityViolation,
    earliest_trip_step,
)


@dataclass(frozen=True)
class Example:
    """One hand-labeled run. ``turns`` are ``record_step`` keyword arguments."""

    name: str
    label: str
    turns: tuple[dict, ...]
    note: str
    warmup_turns: int = 3
    window: int = 3
    token_budget: int = 100_000
    budget_gate: int = 0
    ratio_threshold: float = 2.0


def run_example(example: Example) -> str:
    """Replay an example and return the pattern the guard actually assigned."""
    guard = StabilityGuard(
        token_budget=example.token_budget,
        ratio_threshold=example.ratio_threshold,
        window=example.window,
        warmup_turns=example.warmup_turns,
        budget_gate=example.budget_gate,
    )
    for turn in example.turns:
        try:
            guard.record_step(**turn)
        except StabilityViolation:
            break
    return guard.trip_pattern or PATTERN_STABLE


CORPUS: tuple[Example, ...] = (
    Example(
        name="healthy_growth",
        label=PATTERN_STABLE,
        note="Linear prompt growth stays under the ratio.",
        turns=tuple({"tokens_used": tokens} for tokens in (1000, 1100, 1200, 1300, 1400, 1500)),
    ),
    Example(
        name="context_spiral",
        label=PATTERN_CONTEXT_SPIRAL,
        note="Three post-warmup turns each clear 2× the warmup mean.",
        turns=tuple({"tokens_used": tokens} for tokens in (1000, 1000, 1000, 2500, 4000, 8000)),
    ),
    Example(
        name="retry_storm",
        label=PATTERN_RETRY_STORM,
        note="The same tool arguments fail on every turn in the window.",
        turns=(
            {"tokens_used": 100},
            {"tokens_used": 100},
            {"tokens_used": 100},
            {"tokens_used": 120, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "tide"}},
            {"tokens_used": 180, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "tide"}},
            {"tokens_used": 900, "errors": 1, "tool_name": "search", "tool_arguments": {"q": "tide"}},
        ),
    ),
    Example(
        name="expensive_flat",
        label=PATTERN_BUDGET_EXHAUSTION,
        note="The ceiling hits while the ratio stays near 1.",
        token_budget=650,
        turns=tuple({"tokens_used": tokens} for tokens in (100, 100, 100, 110, 120, 130)),
    ),
    Example(
        name="short_loop",
        label=PATTERN_STABLE,
        note=(
            "Five turns cannot trip a 3-warmup, 3-window ratio check. "
            f"The first possible ratio step is {earliest_trip_step()}."
        ),
        turns=tuple({"tokens_used": tokens} for tokens in (1000, 1000, 1000, 9000, 9000)),
    ),
    Example(
        name="output_ramble",
        label=PATTERN_OUTPUT_RAMBLE,
        note="Prompt size stays flat and the completion climbs.",
        turns=(
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 100, "output_tokens": 80},
            {"input_tokens": 110, "output_tokens": 90},
            {"input_tokens": 100, "output_tokens": 120},
        ),
    ),
)
