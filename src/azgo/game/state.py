"""Immutable Go game state and rule application."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from .coordinates import action_to_coord, coord_to_action
from .errors import (
    GameNotFinishedError,
    GameOverError,
    IllegalMoveError,
    InvalidActionError,
    OccupiedPointError,
    SuicideError,
    SuperkoError,
)
from .types import Color, Group, Rules, Score, require_player
from .zobrist import ZobristTable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


@cache
def _neighbor_table(board_size: int) -> tuple[tuple[int, ...], ...]:
    """Build the deterministic orthogonal-neighbor table for a board size."""

    table: list[tuple[int, ...]] = []
    for point in range(board_size * board_size):
        row, col = divmod(point, board_size)
        neighbors: list[int] = []
        if row > 0:
            neighbors.append(point - board_size)
        if row + 1 < board_size:
            neighbors.append(point + board_size)
        if col > 0:
            neighbors.append(point - 1)
        if col + 1 < board_size:
            neighbors.append(point + 1)
        table.append(tuple(neighbors))
    return tuple(table)


def _collect_group(
    board: Sequence[int],
    neighbor_table: tuple[tuple[int, ...], ...],
    start: int,
) -> Group:
    raw_color = board[start]
    if raw_color == Color.EMPTY:
        raise ValueError("cannot collect a group from an empty intersection")
    color = Color(raw_color)
    stones: set[int] = set()
    liberties: set[int] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        if point in stones:
            continue
        stones.add(point)
        for neighbor in neighbor_table[point]:
            neighbor_color = board[neighbor]
            if neighbor_color == Color.EMPTY:
                liberties.add(neighbor)
            elif neighbor_color == color and neighbor not in stones:
                pending.append(neighbor)
    return Group(color=color, stones=frozenset(stones), liberties=frozenset(liberties))


def _collect_empty_region(
    board: Sequence[int],
    neighbor_table: tuple[tuple[int, ...], ...],
    start: int,
) -> tuple[frozenset[int], frozenset[Color]]:
    region: set[int] = set()
    border_colors: set[Color] = set()
    pending = [start]
    while pending:
        point = pending.pop()
        if point in region:
            continue
        region.add(point)
        for neighbor in neighbor_table[point]:
            raw_color = board[neighbor]
            if raw_color == Color.EMPTY:
                if neighbor not in region:
                    pending.append(neighbor)
            else:
                border_colors.add(Color(raw_color))
    return frozenset(region), frozenset(border_colors)


def _normalize_board(board: bytes | bytearray | Sequence[Color | int], point_count: int) -> bytes:
    if len(board) != point_count:
        raise ValueError(f"board must contain exactly {point_count} intersections")
    normalized: list[int] = []
    for point, raw_color in enumerate(board):
        try:
            color = Color(raw_color)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid intersection value at point {point}: {raw_color!r}") from exc
        normalized.append(int(color))
    return bytes(normalized)


@dataclass(frozen=True, slots=True)
class _Placement:
    board: bytes
    board_hash: int


@dataclass(frozen=True, slots=True, init=False)
class GameState:
    """An immutable position plus all history needed to enforce positional superko.

    Use :meth:`new` for an empty game or :meth:`from_board` for an explicitly
    constructed analysis position. Every field reachable through the public
    properties is itself immutable.
    """

    rules: Rules
    zobrist: ZobristTable
    _board: bytes
    to_play: Color
    consecutive_passes: int
    move_number: int
    _board_hash: int
    _history: tuple[bytes, ...]
    _hash_history: tuple[int, ...]
    last_action: int | None

    def __init__(self) -> None:
        raise TypeError("construct a GameState with GameState.new() or GameState.from_board()")

    @classmethod
    def _build(
        cls,
        *,
        rules: Rules,
        zobrist: ZobristTable,
        board: bytes,
        to_play: Color,
        consecutive_passes: int,
        move_number: int,
        board_hash: int,
        history: tuple[bytes, ...],
        hash_history: tuple[int, ...],
        last_action: int | None,
    ) -> GameState:
        state = object.__new__(cls)
        object.__setattr__(state, "rules", rules)
        object.__setattr__(state, "zobrist", zobrist)
        object.__setattr__(state, "_board", board)
        object.__setattr__(state, "to_play", to_play)
        object.__setattr__(state, "consecutive_passes", consecutive_passes)
        object.__setattr__(state, "move_number", move_number)
        object.__setattr__(state, "_board_hash", board_hash)
        object.__setattr__(state, "_history", history)
        object.__setattr__(state, "_hash_history", hash_history)
        object.__setattr__(state, "last_action", last_action)
        return state

    @classmethod
    def new(
        cls,
        rules: Rules | int | None = None,
        zobrist: ZobristTable | None = None,
        *,
        board_size: int | None = None,
        komi: float | None = None,
        zobrist_seed: int | None = None,
    ) -> GameState:
        """Create an empty game.

        ``rules`` may be a :class:`Rules` object or a board-size integer. The
        explicit ``board_size`` form is also supported for convenient scripts.
        """

        if isinstance(rules, Rules):
            if board_size is not None or komi is not None:
                raise ValueError("board_size and komi cannot override a Rules object")
            normalized_rules = rules
        elif rules is None:
            normalized_rules = Rules(
                board_size=5 if board_size is None else board_size,
                komi=7.5 if komi is None else komi,
            )
        elif isinstance(rules, int) and not isinstance(rules, bool):
            if board_size is not None:
                raise ValueError("board size was supplied twice")
            normalized_rules = Rules(
                board_size=rules,
                komi=7.5 if komi is None else komi,
            )
        else:
            raise TypeError("rules must be a Rules object, a board-size integer, or None")

        if zobrist is None:
            normalized_zobrist = ZobristTable(
                normalized_rules.board_size,
                0 if zobrist_seed is None else zobrist_seed,
            )
        else:
            if zobrist.board_size != normalized_rules.board_size:
                raise ValueError("Zobrist table size does not match the rules board size")
            if zobrist_seed is not None and zobrist.seed != zobrist_seed:
                raise ValueError("zobrist_seed does not match the supplied Zobrist table")
            normalized_zobrist = zobrist

        board = bytes(normalized_rules.board_size * normalized_rules.board_size)
        board_hash = normalized_zobrist.hash_board(board)
        return cls._build(
            rules=normalized_rules,
            zobrist=normalized_zobrist,
            board=board,
            to_play=Color.BLACK,
            consecutive_passes=0,
            move_number=0,
            board_hash=board_hash,
            history=(board,),
            hash_history=(board_hash,),
            last_action=None,
        )

    @classmethod
    def from_board(
        cls,
        board: bytes | bytearray | Sequence[Color | int],
        rules: Rules | int,
        zobrist: ZobristTable | None = None,
        *,
        to_play: Color | int = Color.BLACK,
        history: Sequence[bytes | bytearray | Sequence[Color | int]] | None = None,
        consecutive_passes: int = 0,
        move_number: int | None = None,
        last_action: int | None = None,
        zobrist_seed: int | None = None,
    ) -> GameState:
        """Construct a validated immutable analysis state from explicit stones.

        When ``history`` is omitted, the supplied board is the sole known
        position. If provided, history must be chronological and end at board.
        """

        normalized_rules = rules if isinstance(rules, Rules) else Rules(board_size=rules)
        point_count = normalized_rules.board_size * normalized_rules.board_size
        normalized_board = _normalize_board(board, point_count)
        normalized_player = require_player(to_play)

        if zobrist is None:
            normalized_zobrist = ZobristTable(
                normalized_rules.board_size,
                0 if zobrist_seed is None else zobrist_seed,
            )
        else:
            if zobrist.board_size != normalized_rules.board_size:
                raise ValueError("Zobrist table size does not match the rules board size")
            if zobrist_seed is not None and zobrist.seed != zobrist_seed:
                raise ValueError("zobrist_seed does not match the supplied Zobrist table")
            normalized_zobrist = zobrist

        normalized_history: tuple[bytes, ...]
        if history is None:
            normalized_history = (normalized_board,)
        else:
            if not history:
                raise ValueError("history must contain at least the current board")
            normalized_history = tuple(_normalize_board(item, point_count) for item in history)
            if normalized_history[-1] != normalized_board:
                raise ValueError("history must end with the supplied current board")
        hash_history = tuple(normalized_zobrist.hash_board(item) for item in normalized_history)

        if (
            isinstance(consecutive_passes, bool)
            or not isinstance(consecutive_passes, int)
            or not 0 <= consecutive_passes <= 2
        ):
            raise ValueError("consecutive_passes must be 0, 1, or 2")
        inferred_move_number = len(normalized_history) - 1
        normalized_move_number = inferred_move_number if move_number is None else move_number
        if (
            isinstance(normalized_move_number, bool)
            or not isinstance(normalized_move_number, int)
            or normalized_move_number < 0
        ):
            raise ValueError("move_number must be a non-negative integer")
        if last_action is not None:
            normalized_last_action = _coerce_action(last_action, normalized_rules.pass_action)
        else:
            normalized_last_action = None

        return cls._build(
            rules=normalized_rules,
            zobrist=normalized_zobrist,
            board=normalized_board,
            to_play=normalized_player,
            consecutive_passes=consecutive_passes,
            move_number=normalized_move_number,
            board_hash=hash_history[-1],
            history=normalized_history,
            hash_history=hash_history,
            last_action=normalized_last_action,
        )

    @property
    def board_size(self) -> int:
        return self.rules.board_size

    @property
    def action_size(self) -> int:
        return self.rules.action_size

    @property
    def pass_action(self) -> int:
        return self.rules.pass_action

    @property
    def board(self) -> bytes:
        """Flat row-major immutable board bytes (0 empty, 1 black, 2 white)."""

        return self._board

    @property
    def history(self) -> tuple[bytes, ...]:
        """Initial board followed by every post-action board, including passes."""

        return self._history

    @property
    def board_history(self) -> tuple[bytes, ...]:
        return self._history

    @property
    def hash_history(self) -> tuple[int, ...]:
        return self._hash_history

    @property
    def position_hash(self) -> int:
        """Zobrist hash of stones only, as used for positional superko."""

        return self._board_hash

    @property
    def board_hash(self) -> int:
        return self._board_hash

    @property
    def zobrist_hash(self) -> int:
        return self._board_hash

    @property
    def state_hash(self) -> int:
        """Hash including side to play, useful for future inference caches."""

        if self.to_play is Color.WHITE:
            return self._board_hash ^ self.zobrist.side_to_play_key
        return self._board_hash

    @property
    def is_terminal(self) -> bool:
        return self.consecutive_passes >= 2

    @property
    def terminal(self) -> bool:
        return self.is_terminal

    def neighbors(self, point: int) -> tuple[int, ...]:
        normalized = _coerce_point(point, self.pass_action)
        return _neighbor_table(self.board_size)[normalized]

    def stone_at(self, point: int) -> Color:
        normalized = _coerce_point(point, self.pass_action)
        return Color(self._board[normalized])

    def at(self, row: int, col: int) -> Color:
        return self.stone_at(self.coord_to_action(row, col))

    def group_at(self, point: int) -> Group | None:
        """Return the group at a point action, or ``None`` when it is empty."""

        normalized = _coerce_point(point, self.pass_action)
        if self._board[normalized] == Color.EMPTY:
            return None
        return _collect_group(self._board, _neighbor_table(self.board_size), normalized)

    def group_at_coord(self, row: int, col: int) -> Group | None:
        return self.group_at(self.coord_to_action(row, col))

    def coord_to_action(self, row: int, col: int) -> int:
        return coord_to_action(row, col, self.board_size)

    def action_to_coord(self, action: int) -> tuple[int, int] | None:
        return action_to_coord(action, self.board_size)

    def _placement(self, action: int) -> _Placement:
        if self._board[action] != Color.EMPTY:
            raise OccupiedPointError(f"intersection {action} is occupied")

        board = bytearray(self._board)
        board[action] = self.to_play
        neighbors = _neighbor_table(self.board_size)
        opponent = self.to_play.opponent
        checked_opponent_stones: set[int] = set()
        for neighbor in neighbors[action]:
            if board[neighbor] != opponent or neighbor in checked_opponent_stones:
                continue
            group = _collect_group(board, neighbors, neighbor)
            checked_opponent_stones.update(group.stones)
            if not group.liberties:
                for captured in group.stones:
                    board[captured] = Color.EMPTY

        own_group = _collect_group(board, neighbors, action)
        if not own_group.liberties:
            raise SuicideError(f"action {action} is suicide under {self.rules.ruleset.value} rules")

        result = bytes(board)
        result_hash = self.zobrist.hash_board(result)
        for previous_hash, previous_board in zip(
            self._hash_history,
            self._history,
            strict=True,
        ):
            if previous_hash == result_hash and previous_board == result:
                raise SuperkoError(f"action {action} violates positional superko")
        return _Placement(board=result, board_hash=result_hash)

    def is_legal(self, action: int) -> bool:
        """Return whether ``action`` is legal without changing this state."""

        if self.is_terminal:
            return False
        try:
            normalized = _coerce_action(action, self.pass_action)
            if normalized != self.pass_action:
                self._placement(normalized)
        except IllegalMoveError:
            return False
        return True

    def legal_actions(self) -> tuple[int, ...]:
        """Return legal actions in ascending integer order (pass is last)."""

        if self.is_terminal:
            return ()
        return tuple(action for action in range(self.action_size) if self.is_legal(action))

    def legal_action_mask(self) -> NDArray[np.bool_]:
        """Return a fresh ``[N*N+1]`` boolean array; the final entry is pass."""

        mask = np.zeros(self.action_size, dtype=np.bool_)
        if not self.is_terminal:
            for action in range(self.action_size):
                mask[action] = self.is_legal(action)
        return mask

    def legal_mask(self) -> NDArray[np.bool_]:
        """Alias for :meth:`legal_action_mask`."""

        return self.legal_action_mask()

    def apply(self, action: int) -> GameState:
        """Apply a legal action and return a new immutable state."""

        if self.is_terminal:
            raise GameOverError("cannot apply an action after two consecutive passes")
        normalized = _coerce_action(action, self.pass_action)
        if normalized == self.pass_action:
            next_board = self._board
            next_hash = self._board_hash
            next_passes = self.consecutive_passes + 1
        else:
            placement = self._placement(normalized)
            next_board = placement.board
            next_hash = placement.board_hash
            next_passes = 0

        return self._build(
            rules=self.rules,
            zobrist=self.zobrist,
            board=next_board,
            to_play=self.to_play.opponent,
            consecutive_passes=next_passes,
            move_number=self.move_number + 1,
            board_hash=next_hash,
            history=(*self._history, next_board),
            hash_history=(*self._hash_history, next_hash),
            last_action=normalized,
        )

    def score(self) -> Score:
        """Compute current Tromp--Taylor-style area, with komi for White."""

        black_stones = self._board.count(Color.BLACK)
        white_stones = self._board.count(Color.WHITE)
        black_territory = 0
        white_territory = 0
        neutral_points = 0
        visited_empty: set[int] = set()
        neighbors = _neighbor_table(self.board_size)

        for point, raw_color in enumerate(self._board):
            if raw_color != Color.EMPTY or point in visited_empty:
                continue
            region, border_colors = _collect_empty_region(self._board, neighbors, point)
            visited_empty.update(region)
            if border_colors == frozenset({Color.BLACK}):
                black_territory += len(region)
            elif border_colors == frozenset({Color.WHITE}):
                white_territory += len(region)
            else:
                neutral_points += len(region)

        return Score(
            black_stones=black_stones,
            white_stones=white_stones,
            black_territory=black_territory,
            white_territory=white_territory,
            neutral_points=neutral_points,
            komi=self.rules.komi,
        )

    def outcome(self, perspective: Color | int) -> int:
        """Return final ``+1/0/-1`` outcome from a player perspective."""

        if not self.is_terminal:
            raise GameNotFinishedError("outcome is defined only after two consecutive passes")
        return self.score().outcome(require_player(perspective))


def _coerce_action(action: int, pass_action_value: int) -> int:
    if isinstance(action, bool):
        raise InvalidActionError("action must be an integer, not bool")
    try:
        normalized = operator.index(action)
    except TypeError as exc:
        raise InvalidActionError("action must be an integer") from exc
    if not 0 <= normalized <= pass_action_value:
        raise InvalidActionError(f"action must be in [0, {pass_action_value}]")
    return normalized


def _coerce_point(point: int, pass_action_value: int) -> int:
    normalized = _coerce_action(point, pass_action_value)
    if normalized == pass_action_value:
        raise InvalidActionError("pass does not identify a board intersection")
    return normalized
