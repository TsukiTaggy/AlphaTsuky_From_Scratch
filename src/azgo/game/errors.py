"""Exceptions raised by the Go rules engine."""


class GoEngineError(Exception):
    """Base class for engine-specific errors."""


class IllegalMoveError(GoEngineError, ValueError):
    """Raised when an action is not legal in the current state."""


class InvalidActionError(IllegalMoveError):
    """Raised when an action is outside the configured action space."""


class OccupiedPointError(IllegalMoveError):
    """Raised when attempting to play on an occupied intersection."""


class SuicideError(IllegalMoveError):
    """Raised when a stone placement would have no liberties after captures."""


class SuperkoError(IllegalMoveError):
    """Raised when a stone placement would repeat a prior board position."""


class GameOverError(IllegalMoveError):
    """Raised when attempting an action after the game has ended."""


class GameNotFinishedError(GoEngineError, RuntimeError):
    """Raised when requesting a final outcome before game termination."""
