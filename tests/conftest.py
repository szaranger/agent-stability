import pytest


@pytest.fixture(autouse=True)
def _clear_overrides(monkeypatch):
    monkeypatch.delenv("AGENT_STABILITY_RATIO_THRESHOLD", raising=False)
    monkeypatch.delenv("AGENT_STABILITY_BUDGET_GATE", raising=False)
