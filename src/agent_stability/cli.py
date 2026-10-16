"""Simulate a token trajectory and print the stability report."""

from __future__ import annotations

import argparse
import sys

from agent_stability.guard import StabilityGuard, StabilityViolation, earliest_trip_step
from agent_stability.report import FailureReport


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-stability")
    sub = parser.add_subparsers(dest="command", required=True)

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
    parser.error(f"unknown command {args.command}")
    return 2


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


def _region(args: argparse.Namespace) -> int:
    first = earliest_trip_step(args.warmup, args.window)
    print(f"Ratio, retry, and drift can trip from step {first}.")
    print(f"Early explosion can trip at step {args.warmup}.")
    print(f"A loop shorter than {first} turns cannot trip on growth ratio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
