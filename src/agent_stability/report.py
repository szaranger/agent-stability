"""Classify a finished guard run without calling a model."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent_stability.edit import ContextEdit
from agent_stability.guard import (
    ACTION_WARN,
    PATTERN_BUDGET_EXHAUSTION,
    PATTERN_CONTEXT_SPIRAL,
    PATTERN_EARLY_EXPLOSION,
    PATTERN_OUTPUT_RAMBLE,
    PATTERN_POLICY_DRIFT,
    PATTERN_RETRY_STORM,
    PATTERN_SEARCH_ABORTED,
    PATTERN_STABLE,
    StabilityGuard,
)

_BLOCKS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class FailureReport:
    """Pattern, evidence, and a concrete edit derived from the trajectory.

    ``confidence`` is how far the firing signal sits past its threshold,
    clipped to 0.99. It is not a probability.
    """

    pattern: str
    confidence: float
    tripped: bool
    tripped_at: int | None
    evidence: tuple[str, ...]
    suggestions: tuple[str, ...]
    baseline: float | None
    peak_ratio: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    token_budget: int
    energy: tuple[float, ...]
    sparkline: str
    cost: float | None
    projected_cost: float | None
    model: str | None
    step_count: int
    edit: ContextEdit | None
    status: str

    @classmethod
    def from_guard(
        cls,
        guard: StabilityGuard,
        model: str | None = None,
    ) -> FailureReport:
        pattern = guard.trip_pattern or PATTERN_STABLE
        steps = guard.steps
        edit = guard.proposed_edit()
        if edit is None and guard.status == ACTION_WARN:
            from agent_stability.edit import propose_edit

            edit = propose_edit(
                pattern=PATTERN_CONTEXT_SPIRAL,
                steps=steps,
                baseline=guard.baseline,
                ratio_threshold=guard.ratio_threshold,
                window=guard.window,
                drift_threshold=guard.drift_threshold,
                token_budget=guard.token_budget,
                total_tokens=guard.total_tokens,
            )
        cost, projected = _costs(guard)
        energy = tuple(step.energy for step in steps)
        suggestions = (edit.message,) if edit is not None else ()
        return cls(
            pattern=pattern,
            confidence=_confidence(guard, pattern),
            tripped=guard.tripped,
            tripped_at=guard.tripped_at,
            evidence=_evidence(guard, pattern),
            suggestions=suggestions,
            baseline=guard.baseline,
            peak_ratio=guard.peak_ratio,
            total_tokens=guard.total_tokens,
            total_input_tokens=guard.total_input_tokens,
            total_output_tokens=guard.total_output_tokens,
            token_budget=guard.token_budget,
            energy=energy,
            sparkline=sparkline(energy),
            cost=cost,
            projected_cost=projected,
            model=model,
            step_count=len(steps),
            edit=edit,
            status=guard.status,
        )

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "confidence": self.confidence,
            "tripped": self.tripped,
            "tripped_at": self.tripped_at,
            "status": self.status,
            "evidence": list(self.evidence),
            "suggestions": list(self.suggestions),
            "baseline": self.baseline,
            "peak_ratio": self.peak_ratio,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "token_budget": self.token_budget,
            "energy": list(self.energy),
            "sparkline": self.sparkline,
            "cost": self.cost,
            "projected_cost": self.projected_cost,
            "model": self.model,
            "step_count": self.step_count,
            "edit": None if self.edit is None else self.edit.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def csv_header() -> str:
        return (
            "pattern,confidence,tripped,total_tokens,baseline,"
            "peak_ratio,cost,step_count,input_tokens,output_tokens"
        )

    def to_csv_row(self) -> str:
        baseline = "" if self.baseline is None else f"{self.baseline:.4f}"
        cost = "" if self.cost is None else f"{self.cost:.6f}"
        values = [
            self.pattern,
            f"{self.confidence:.4f}",
            "1" if self.tripped else "0",
            str(self.total_tokens),
            baseline,
            f"{self.peak_ratio:.4f}",
            cost,
            str(self.step_count),
            str(self.total_input_tokens),
            str(self.total_output_tokens),
        ]
        return ",".join(values)

    def __str__(self) -> str:
        if self.tripped:
            header = f"Stability tripped at step {self.tripped_at}"
        elif self.status == ACTION_WARN:
            header = "Stability warning"
        else:
            header = "Stability held"
        lines = [
            header,
            f"Pattern: {self.pattern} (confidence {self.confidence:.0%})",
        ]
        if self.model:
            lines.append(f"Model: {self.model}")
        for item in self.evidence:
            lines.append(f"  - {item}")
        if self.sparkline:
            lines.append(f"Energy: {self.sparkline}")
        baseline = "n/a" if self.baseline is None else f"{self.baseline:.0f} tokens/turn"
        lines.append(f"Baseline: {baseline}")
        lines.append(f"Peak ratio: {self.peak_ratio:.2f}x")
        lines.append(
            f"Tokens: {self.total_tokens} of {self.token_budget} "
            f"(input {self.total_input_tokens}, output {self.total_output_tokens})"
        )
        if self.cost is not None:
            lines.append(f"Cost: ${self.cost:.4f}")
            if self.projected_cost is not None and self.tripped:
                saved = max(0.0, self.projected_cost - self.cost)
                lines.append(
                    f"Projected if run to budget: ${self.projected_cost:.4f} (saved ${saved:.4f})"
                )
        if self.suggestions:
            lines.append("Suggested actions:")
            for index, suggestion in enumerate(self.suggestions, start=1):
                lines.append(f"  {index}. {suggestion}")
        return "\n".join(lines)


def sparkline(values: tuple[float, ...] | list[float]) -> str:
    if not values:
        return ""
    low = min(values)
    high = max(values)
    if high == low:
        return _BLOCKS[0] * len(values)
    span = high - low
    chars = []
    last = len(_BLOCKS) - 1
    for value in values:
        slot = int(round((value - low) / span * last))
        chars.append(_BLOCKS[max(0, min(last, slot))])
    return "".join(chars)


def _evidence(guard: StabilityGuard, pattern: str) -> tuple[str, ...]:
    steps = guard.steps
    if not steps:
        return ("No turns recorded.",)
    items = [
        f"Peak growth ratio: {guard.peak_ratio:.2f}x baseline.",
        (
            f"Spent {guard.total_tokens} tokens of a {guard.token_budget} budget "
            f"(input {guard.total_input_tokens}, output {guard.total_output_tokens})."
        ),
    ]
    if pattern == PATTERN_CONTEXT_SPIRAL:
        recent = steps[-guard.window :]
        climbed = sum(
            1
            for index in range(1, len(recent))
            if recent[index].ratio > recent[index - 1].ratio
        )
        items.insert(
            0,
            (
                f"Last {len(recent)} turns exceeded {guard.ratio_threshold:.1f}x baseline "
                f"({climbed} of {len(recent) - 1} steps were still climbing)."
            ),
        )
    elif pattern == PATTERN_OUTPUT_RAMBLE:
        items.insert(
            0,
            (
                f"Completions exceeded {guard.ratio_threshold:.1f}x the output baseline "
                "while prompts stayed at or under it."
            ),
        )
    elif pattern == PATTERN_RETRY_STORM:
        name = steps[-1].tool_name or "tool"
        items.insert(
            0,
            f"Last {guard.window} calls repeated {name} with the same arguments and an error.",
        )
    elif pattern == PATTERN_POLICY_DRIFT:
        items.insert(
            0,
            f"Similarity stayed below {guard.drift_threshold:.2f} for {guard.window} turns.",
        )
    elif pattern == PATTERN_EARLY_EXPLOSION:
        warmup = steps[: guard.warmup_turns]
        peak = max(step.tokens for step in warmup)
        items.insert(0, f"A warmup turn used {peak} tokens, far above the other opening turns.")
    elif pattern == PATTERN_BUDGET_EXHAUSTION:
        items.insert(0, "The run hit the token ceiling. The growth-ratio rule did not fire.")
    elif pattern == PATTERN_SEARCH_ABORTED:
        items.insert(0, "The shared search ceiling was spent.")
    return tuple(items)


def _confidence(guard: StabilityGuard, pattern: str) -> float:
    threshold = guard.ratio_threshold
    if pattern == PATTERN_STABLE:
        if guard.baseline is None:
            return 0.0
        return _headroom(guard.current_ratio, threshold)
    if pattern == PATTERN_CONTEXT_SPIRAL:
        return _overshoot(guard.peak_ratio, threshold)
    if pattern == PATTERN_OUTPUT_RAMBLE:
        return _overshoot(guard.peak_output_ratio, threshold)
    if pattern == PATTERN_RETRY_STORM:
        return 0.99
    if pattern == PATTERN_POLICY_DRIFT:
        recent = guard.steps[-guard.window :]
        scores = [step.similarity for step in recent if step.similarity is not None]
        if not scores or guard.drift_threshold <= 0:
            return 0.0
        gaps = [(guard.drift_threshold - score) / guard.drift_threshold for score in scores]
        return min(0.99, max(0.0, sum(gaps) / len(gaps)))
    if pattern == PATTERN_EARLY_EXPLOSION:
        return _overshoot(guard.peak_ratio, threshold)
    if pattern in {PATTERN_BUDGET_EXHAUSTION, PATTERN_SEARCH_ABORTED}:
        return _headroom(guard.peak_ratio, threshold)
    return 0.0


def _overshoot(peak: float, threshold: float) -> float:
    """Fraction of the threshold by which the peak cleared the line, clipped to 0.99."""
    if threshold <= 0:
        return 0.0
    return min(0.99, max(0.0, (peak - threshold) / threshold))


def _headroom(ratio: float, threshold: float) -> float:
    """Fraction of the threshold still unused, clipped to 0.99."""
    if threshold <= 0:
        return 0.0
    return min(0.99, max(0.0, (threshold - ratio) / threshold))


def _costs(guard: StabilityGuard) -> tuple[float | None, float | None]:
    input_price = guard.input_price_per_million
    output_price = guard.output_price_per_million
    if input_price is None and output_price is None:
        return None, None
    input_price = 0.0 if input_price is None else input_price
    output_price = 0.0 if output_price is None else output_price
    cost = (
        guard.total_input_tokens / 1_000_000 * input_price
        + guard.total_output_tokens / 1_000_000 * output_price
    )
    total = guard.total_tokens
    if total <= 0:
        blend = input_price
    else:
        blend = (
            guard.total_input_tokens * input_price + guard.total_output_tokens * output_price
        ) / total
    projected = guard.token_budget / 1_000_000 * blend
    return cost, projected
