"""Concrete next-step edits computed from a recorded trajectory."""

from __future__ import annotations

from dataclasses import dataclass

from agent_stability.guard import (
    PATTERN_BUDGET_EXHAUSTION,
    PATTERN_CONTEXT_SPIRAL,
    PATTERN_EARLY_EXPLOSION,
    PATTERN_OUTPUT_RAMBLE,
    PATTERN_POLICY_DRIFT,
    PATTERN_RETRY_STORM,
    PATTERN_SEARCH_ABORTED,
    Step,
)


@dataclass(frozen=True)
class ContextEdit:
    """A change the caller can apply before the next model call.

    ``drop_steps`` are 1-based turn numbers whose added prompt tokens should
    be left out. The package does not rewrite the prompt.
    """

    message: str
    drop_steps: tuple[int, ...] = ()
    estimated_prompt_tokens: int | None = None
    estimated_ratio: float | None = None
    block_tool: str | None = None
    block_arguments_hash: str | None = None

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "drop_steps": list(self.drop_steps),
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "estimated_ratio": self.estimated_ratio,
            "block_tool": self.block_tool,
            "block_arguments_hash": self.block_arguments_hash,
        }


def propose_edit(
    *,
    pattern: str | None,
    steps: tuple[Step, ...],
    baseline: float | None,
    ratio_threshold: float,
    window: int,
    drift_threshold: float,
    token_budget: int,
    total_tokens: int,
) -> ContextEdit | None:
    """Describe the edit implied by the rule that fired."""
    if not steps or pattern is None:
        return None
    if pattern == PATTERN_CONTEXT_SPIRAL:
        return _spiral_edit(steps, baseline, ratio_threshold)
    if pattern == PATTERN_OUTPUT_RAMBLE:
        return _ramble_edit(steps, ratio_threshold)
    if pattern == PATTERN_RETRY_STORM:
        return _retry_edit(steps, window)
    if pattern == PATTERN_POLICY_DRIFT:
        return _drift_edit(steps, window, drift_threshold)
    if pattern == PATTERN_EARLY_EXPLOSION:
        return _early_explosion_edit(steps)
    if pattern in {PATTERN_BUDGET_EXHAUSTION, PATTERN_SEARCH_ABORTED}:
        return _budget_edit(steps, baseline, ratio_threshold, token_budget, total_tokens, pattern)
    return None


def _spiral_edit(
    steps: tuple[Step, ...],
    baseline: float | None,
    ratio_threshold: float,
) -> ContextEdit:
    current = steps[-1].input_tokens
    increments = _prompt_increments(steps)
    target = None if baseline is None or baseline <= 0 else ratio_threshold * baseline
    droppable = [index for index in range(1, len(steps)) if increments[index] > 0]
    droppable.sort(key=lambda index: (steps[index].tool_name is None, index))

    dropped: list[int] = []
    cut = 0
    if target is not None:
        need = max(0, current - int(target))
        for index in droppable:
            if cut >= need:
                break
            dropped.append(index)
            cut += increments[index]

    estimated = max(0, current - cut)
    ratio = None if baseline is None or baseline <= 0 else estimated / baseline
    worst = max(range(len(steps)), key=lambda index: steps[index].ratio)
    worst_step = steps[worst]
    turn = worst + 1
    kind = f"{worst_step.tool_name} result" if worst_step.tool_name else "prompt"
    if dropped and ratio is not None and baseline is not None:
        relation = "under" if ratio <= ratio_threshold else "still above"
        dropped_labels = ", ".join(str(index + 1) for index in sorted(dropped))
        message = (
            f"Turn {turn}'s {kind} is {worst_step.ratio:.1f}× the baseline. "
            f"Dropping the prompt growth from turns {dropped_labels} leaves about "
            f"{estimated} prompt tokens ({ratio:.1f}× baseline), {relation} {ratio_threshold:.1f}×."
        )
    else:
        message = f"Turn {turn}'s {kind} is {worst_step.ratio:.1f}× the baseline."
    return ContextEdit(
        message=message,
        drop_steps=tuple(index + 1 for index in sorted(dropped)),
        estimated_prompt_tokens=estimated,
        estimated_ratio=ratio,
    )


def _ramble_edit(steps: tuple[Step, ...], ratio_threshold: float) -> ContextEdit:
    worst = max(range(len(steps)), key=lambda index: steps[index].output_ratio)
    step = steps[worst]
    return ContextEdit(
        message=(
            f"Turn {worst + 1} completion is {step.output_ratio:.1f}× the output baseline "
            f"({step.output_tokens} tokens) while the prompt is {step.input_ratio:.1f}× "
            f"the input baseline. Cap the completion; the prompt is within {ratio_threshold:.1f}×."
        )
    )


def _retry_edit(steps: tuple[Step, ...], window: int) -> ContextEdit:
    last = steps[-1]
    name = last.tool_name or "tool"
    start = len(steps) - window + 1
    return ContextEdit(
        message=(
            f"Stop calling {name}. Turns {start}-{len(steps)} used the same arguments and returned an error."
        ),
        drop_steps=tuple(range(start, len(steps) + 1)),
        block_tool=last.tool_name,
        block_arguments_hash=last.arguments_hash,
    )


def _drift_edit(steps: tuple[Step, ...], window: int, drift_threshold: float) -> ContextEdit:
    start = len(steps) - window + 1
    return ContextEdit(
        message=(
            f"Turns {start}-{len(steps)} stayed below similarity {drift_threshold:.2f}. "
            "Restate the original task before the next call."
        ),
        drop_steps=tuple(range(start, len(steps) + 1)),
    )


def _early_explosion_edit(steps: tuple[Step, ...]) -> ContextEdit:
    worst = max(range(len(steps)), key=lambda index: steps[index].tokens)
    step = steps[worst]
    where = "prompt" if not step.split or step.input_tokens >= step.output_tokens else "completion"
    size = step.input_tokens if where == "prompt" else step.output_tokens
    return ContextEdit(
        message=(
            f"Turn {worst + 1} {where} is {size} tokens, above the other opening turns. "
            "Shrink that payload before sending it again."
        ),
        drop_steps=(worst + 1,),
        estimated_prompt_tokens=step.input_tokens,
    )


def _budget_edit(
    steps: tuple[Step, ...],
    baseline: float | None,
    ratio_threshold: float,
    token_budget: int,
    total_tokens: int,
    pattern: str,
) -> ContextEdit:
    peak = max(step.ratio for step in steps)
    ceiling = "shared search ceiling" if pattern == PATTERN_SEARCH_ABORTED else "token ceiling"
    baseline_text = f" Baseline {baseline:.0f}." if baseline else ""
    return ContextEdit(
        message=(
            f"Spent {total_tokens} tokens against a {ceiling} of {token_budget}. "
            f"Peak growth was {peak:.2f}×.{baseline_text} "
            f"The growth-ratio rule (above {ratio_threshold:.1f}×) did not fire, so raise the ceiling or split the work."
        )
    )


def _prompt_increments(steps: tuple[Step, ...]) -> list[int]:
    """New prompt tokens at each turn, assuming ``input_tokens`` is the full prompt size."""
    previous = 0
    increments: list[int] = []
    for step in steps:
        increments.append(max(0, step.input_tokens - previous))
        previous = max(previous, step.input_tokens)
    return increments
