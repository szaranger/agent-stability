"""Shared search ceiling with an independent growth ratio on each branch."""

from __future__ import annotations

from agent_stability.guard import StabilityGuard, SearchAborted


class SearchBudget:
    """Token ceiling shared by every branch of a search.

    A branch that trips on growth is pruned. The search continues until a
    branch spends the shared ceiling, which aborts every remaining branch.
    """

    def __init__(self, token_budget: int):
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1:
            raise ValueError("token_budget must be an int >= 1")
        self.token_budget = token_budget
        self.spent = 0
        self.aborted = False

    def branch(self, name: str, **kwargs) -> StabilityGuard:
        """Start a branch. Ratio settings belong to the branch; the ceiling is shared."""
        if self.aborted:
            raise SearchAborted(
                "search_aborted",
                0,
                f"search already aborted at {self.spent} tokens",
            )
        kwargs["token_budget"] = self.token_budget
        guard = StabilityGuard(**kwargs)
        guard.attach_search(self, name)
        return guard

    def spend(self, tokens: int) -> bool:
        """Add tokens. Return True when the shared ceiling is now spent."""
        self.spent += tokens
        if self.spent >= self.token_budget:
            self.aborted = True
            return True
        return False
