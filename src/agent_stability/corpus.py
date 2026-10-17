"""Labeled trajectories and the step on which a ratio trip becomes possible."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def run_example(
    example: Example,
    *,
    ratio_threshold: float | None = None,
    max_turns: int | None = None,
) -> str:
    """Replay an example and return the pattern the guard actually assigned.

    ``ratio_threshold`` replaces the example's cutoff. ``max_turns`` keeps
    only the first turns of the trajectory.
    """
    turns = example.turns if max_turns is None else example.turns[:max_turns]
    guard = StabilityGuard(
        token_budget=example.token_budget,
        ratio_threshold=example.ratio_threshold if ratio_threshold is None else ratio_threshold,
        window=example.window,
        warmup_turns=example.warmup_turns,
        budget_gate=example.budget_gate,
    )
    for turn in turns:
        try:
            guard.record_step(**turn)
        except StabilityViolation:
            break
    return guard.trip_pattern or PATTERN_STABLE


def load_corpus(path: str | Path) -> tuple[Example, ...]:
    """Load ``*.json`` trajectories from a directory."""
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    files = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no JSON trajectories in {directory}")
    return tuple(_example_from_json(file) for file in files)


def _example_from_json(path: Path) -> Example:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be an object")
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{path.name}: turns must be a non-empty list")
    normalized: list[dict] = []
    for turn in turns:
        if isinstance(turn, int) and not isinstance(turn, bool):
            normalized.append({"tokens_used": turn})
        elif isinstance(turn, dict):
            normalized.append(turn)
        else:
            raise ValueError(f"{path.name}: each turn must be an int or an object")
    return Example(
        name=str(payload.get("name") or path.stem),
        label=str(payload.get("label", "")),
        turns=tuple(normalized),
        note=str(payload.get("note", "")),
        warmup_turns=int(payload.get("warmup_turns", 3)),
        window=int(payload.get("window", 3)),
        token_budget=int(payload.get("token_budget", 100_000)),
        budget_gate=int(payload.get("budget_gate", 0)),
        ratio_threshold=float(payload.get("ratio_threshold", 2.0)),
    )


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
