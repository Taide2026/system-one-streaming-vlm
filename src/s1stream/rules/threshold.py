"""Decision rule "threshold": alarm when the score is at or above a threshold."""
from __future__ import annotations

from s1stream.registry import DECISION_RULES


@DECISION_RULES.register("threshold")
class Threshold:
    """Thresholds start, start + step, ... up to and including stop (rounded to 10 decimals)."""

    def __init__(self, start: float = 0.0, stop: float = 1.0, step: float = 0.01) -> None:
        self.start, self.stop, self.step = start, stop, step

    def settings(self) -> list[float]:
        raise NotImplementedError

    def alarms(self, scores: list[float]) -> list[list[bool]]:
        """One sequence per threshold t: [score >= t for each score]."""
        raise NotImplementedError
