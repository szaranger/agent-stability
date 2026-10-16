import json

from agent_stability import FailureReport, StabilityGuard, StabilityViolation
from agent_stability.cli import main
from agent_stability.report import sparkline


def _tripped_guard() -> StabilityGuard:
    guard = StabilityGuard(
        token_budget=50_000,
        budget_gate=0,
        price_per_million=2.0,
    )
    for tokens in (1000, 1000, 1000, 2500, 4000, 8000):
        try:
            guard.record_step(tokens_used=tokens)
        except StabilityViolation:
            break
    return guard


def test_report_renders_spiral_details():
    report = FailureReport.from_guard(_tripped_guard(), model="example-model")

    assert report.pattern == "context_spiral"
    assert report.confidence > 0.8
    assert report.model == "example-model"
    assert report.baseline == 1000
    assert report.cost == (1000 * 3 + 2500 + 4000 + 8000) / 1_000_000 * 2.0
    assert report.projected_cost == 50_000 / 1_000_000 * 2.0
    text = str(report)
    assert "Stability tripped at step 6" in text
    assert "example-model" in text
    assert "Suggested actions:" in text
    assert report.edit is not None
    assert "Turn 6" in report.edit.message
    assert report.edit.estimated_ratio is not None
    assert report.edit.estimated_ratio <= 2.0
    assert len(report.sparkline) == report.step_count


def test_json_and_csv_round_trip_columns():
    report = FailureReport.from_guard(_tripped_guard())
    payload = json.loads(report.to_json())

    assert payload["pattern"] == "context_spiral"
    assert payload["tripped"] is True
    header = FailureReport.csv_header().split(",")
    row = report.to_csv_row().split(",")
    assert len(header) == len(row)
    assert row[0] == "context_spiral"


def test_sparkline_uses_block_characters():
    assert sparkline(()) == ""
    assert set(sparkline((1.0, 2.0, 4.0))) <= set("▁▂▃▄▅▆▇█")
    assert sparkline((3.0, 3.0)) == "▁▁"


def test_simulate_command(capsys):
    code = main(
        [
            "simulate",
            "1000",
            "1200",
            "1500",
            "2000",
            "3000",
            "5000",
            "8000",
            "--budget",
            "50000",
            "--gate",
            "8000",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "context_spiral" in output
    assert "Stability tripped at step 7" in output
