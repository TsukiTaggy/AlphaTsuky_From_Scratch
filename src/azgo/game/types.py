"""Core value types for the Go rules engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


SUPPORTED_BOARD_SIZES = frozenset({5, 9, 13, 19})


class Color(IntEnum):
    """Contents of an intersection and, for non-empty values, a player color."""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> Color:
        """Return the opposing player color."""

        if self is Color.BLACK:
            return Color.WHITE
        if self is Color.WHITE:
            return Color.BLACK
        raise ValueError("the empty color has no opponent")


# A semantic alias useful when interpreting values read from ``GameState.board``.
Intersection = Color
Stone = Color


class Ruleset(StrEnum):
    """Rulesets implemented by this milestone."""

    TROMP_TAYLOR = "tromp_taylor"


@dataclass(frozen=True, slots=True)
class Rules:
    """Validated rules that determine the behavior of a game.

    The baseline uses area scoring, positional superko, illegal suicide, and
    repetition-exempt passes. Those semantics are deliberately not switches:
    supporting a named ruleset means supporting it completely.
    """

    board_size: int = 5
    komi: float = 7.5
    ruleset: Ruleset = Ruleset.TROMP_TAYLOR

    def __post_init__(self) -> None:
        if (
            isinstance(self.board_size, bool)
            or not isinstance(self.board_size, int)
            or self.board_size not in SUPPORTED_BOARD_SIZES
        ):
            supported = ", ".join(str(size) for size in sorted(SUPPORTED_BOARD_SIZES))
            raise ValueError(f"board_size must be one of {{{supported}}}")

        try:
            komi = float(self.komi)
        except (TypeError, ValueError) as exc:
            raise ValueError("komi must be a finite number") from exc
        if not isfinite(komi):
            raise ValueError("komi must be a finite number")
        object.__setattr__(self, "komi", komi)

        try:
            ruleset = Ruleset(self.ruleset)
        except ValueError as exc:
            raise ValueError(f"unsupported ruleset: {self.ruleset!r}") from exc
        object.__setattr__(self, "ruleset", ruleset)

    @property
    def action_size(self) -> int:
        """Number of policy actions, including pass."""

        return self.board_size * self.board_size + 1

    @property
    def pass_action(self) -> int:
        """Integer action used for pass."""

        return self.board_size * self.board_size


GameRules = Rules


@dataclass(frozen=True, slots=True)
class Group:
    """A connected same-color group and its current liberties."""

    color: Color
    stones: frozenset[int]
    liberties: frozenset[int]

    @property
    def size(self) -> int:
        return len(self.stones)


@dataclass(frozen=True, slots=True)
class Score:
    """Tromp--Taylor-style area score, with komi applied to White."""

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    neutral_points: int
    komi: float

    @property
    def black(self) -> float:
        return float(self.black_stones + self.black_territory)

    @property
    def white(self) -> float:
        return float(self.white_stones + self.white_territory) + self.komi

    @property
    def black_score(self) -> float:
        return self.black

    @property
    def white_score(self) -> float:
        return self.white

    @property
    def winner(self) -> Color | None:
        if self.black > self.white:
            return Color.BLACK
        if self.white > self.black:
            return Color.WHITE
        return None

    @property
    def black_margin(self) -> float:
        """Black's signed margin; negative values favor White."""

        return self.black - self.white

    def for_player(self, color: Color) -> float:
        color = require_player(color)
        return self.black if color is Color.BLACK else self.white

    def outcome(self, perspective: Color) -> int:
        """Return ``+1``, ``0``, or ``-1`` from ``perspective``."""

        perspective = require_player(perspective)
        winner = self.winner
        if winner is None:
            return 0
        return 1 if winner is perspective else -1

    def __iter__(self) -> Iterator[float]:
        """Allow ``black, white = score`` without losing named fields."""

        yield self.black
        yield self.white


def require_player(color: Color | int) -> Color:
    """Normalize and validate a non-empty player color."""

    try:
        normalized = Color(color)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid player color: {color!r}") from exc
    if normalized is Color.EMPTY:
        raise ValueError("a player color must be BLACK or WHITE")
    return normalized
