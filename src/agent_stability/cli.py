"""Simulate a token trajectory and print the stability report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_stability.corpus import CORPUS, load_corpus, run_example
from agent_stability.guard import StabilityGuard, StabilityViolation, earliest_trip_step
from agent_stability.report import FailureReport

# The default growth-ratio cutoff is 2× the warmup baseline. --safety-k scales it:
# 0.90 trips at 1.8×, and 1 keeps the 2× line.
_DEFAULT_RATIO = 2.0


def _safety_k(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("safety-k must be a number") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("safety-k must be > 0")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-stability")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Directory of JSON trajectories to replay",
    )
    parser.add_argument(
        "--safety-k",
        type=_safety_k,
        default=None,
        help="Scale the 2× growth-ratio cutoff (0.90 trips at 1.8× the warmup baseline)",
    )
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=None,
        help="Replay at most this many turns from each trajectory",
    )
    sub = parser.add_subparsers(dest="command")

    simulate = sub.add_parser("simulate", help="Replay token counts through the guard")
    simulate.add_argument("tokens", nargs="+", type=int, help="Token count for each turn")
    simulate.add_argument("--budget", type=int, default=100_000)
    simulate.add_argument("--threshold", type=float, default=2.0)
    simulate.add_argument("--window", type=int, default=3)
    simulate.add_argument("--warmup", type=int, default=3)
    simulate.add_argument("--gate", type=int, default=8_000)
    simulate.add_argument("--price-per-million", type=float, default=None)
    simulate.add_argument("--input-price-per-million", type=float, default=None)
    simulate.add_argument("--output-price-per-million", type=float, default=None)

    region = sub.add_parser("region", help="Print the first step on which a ratio trip is possible")
    region.add_argument("--warmup", type=int, default=3)
    region.add_argument("--window", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "region":
        return _region(args)
    if args.command is not None:
        parser.error(f"unknown command {args.command}")
    if args.corpus is None and args.max_turns is None:
        parser.error("choose simulate or region, or pass --corpus / --max-turns")
    return _run_corpus(args)


def _simulate(args: argparse.Namespace) -> int:
    guard = StabilityGuard(
        token_budget=args.budget,
        ratio_threshold=args.threshold,
        window=args.window,
        warmup_turns=args.warmup,
        budget_gate=args.gate,
        price_per_million=args.price_per_million,
        input_price_per_million=args.input_price_per_million,
        output_price_per_million=args.output_price_per_million,
    )
    for tokens in args.tokens:
        try:
            guard.record_step(tokens_used=tokens)
        except StabilityViolation:
            break
    print(FailureReport.from_guard(guard))
    return 0


def _run_corpus(args: argparse.Namespace) -> int:
    cutoff = None if args.safety_k is None else _DEFAULT_RATIO * args.safety_k
    try:
        examples = load_corpus(args.corpus) if args.corpus is not None else CORPUS
    except (FileNotFoundError, ValueError) as exc:
        print(f"agent-stability: {exc}", file=sys.stderr)
        return 2

    header: list[str] = []
    if args.corpus is not None:
        header.append(str(args.corpus))
    if args.safety_k is not None and cutoff is not None:
        header.append(f"safety-k {args.safety_k:g}  ratio cutoff {cutoff:.2f}")
    if args.max_turns is not None:
        header.append(f"max turns {args.max_turns}")
    if header:
        print("  ".join(header))

    for example in examples:
        pattern = run_example(example, ratio_threshold=cutoff, max_turns=args.max_turns)
        line = f"{example.name}  {pattern}"
        if example.label and example.label != pattern:
            line += f"  expected {example.label}"
        print(line)
    return 0


def _region(args: argparse.Namespace) -> int:
    first = earliest_trip_step(args.warmup, args.window)
    print(f"Ratio, retry, and drift can trip from step {first}.")
    print(f"Early explosion can trip at step {args.warmup}.")
    print(f"A loop shorter than {first} turns cannot trip on growth ratio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
