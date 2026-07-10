"""Deterministic, immutable Zobrist tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .coordinates import _validate_board_size
from .types import Color

if TYPE_CHECKING:
    from collections.abc import Sequence

_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + _GOLDEN_GAMMA) & _MASK_64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return state, (value ^ (value >> 31)) & _MASK_64


def _validate_keys(keys: Sequence[int], point_count: int, name: str) -> tuple[int, ...]:
    if len(keys) != point_count:
        raise ValueError(f"{name} must contain exactly {point_count} keys")
    normalized: list[int] = []
    for key in keys:
        if isinstance(key, bool) or not isinstance(key, int) or not 0 <= key <= _MASK_64:
            raise ValueError(f"every {name} key must be an unsigned 64-bit integer")
        normalized.append(key)
    return tuple(normalized)


@dataclass(frozen=True, slots=True, init=False)
class ZobristTable:
    """Per-intersection keys generated reproducibly from an explicit seed.

    Custom key arrays are accepted to make hash-collision behavior directly
    testable. Game legality never relies on a hash match alone.
    """

    board_size: int
    seed: int
    black_keys: tuple[int, ...]
    white_keys: tuple[int, ...]
    side_to_play_key: int

    def __init__(
        self,
        board_size: int,
        seed: int = 0,
        *,
        black_keys: Sequence[int] | None = None,
        white_keys: Sequence[int] | None = None,
        side_to_play_key: int | None = None,
    ) -> None:
        size = _validate_board_size(board_size)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MASK_64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        point_count = size * size

        if (black_keys is None) != (white_keys is None):
            raise ValueError("black_keys and white_keys must be supplied together")

        state = seed ^ ((size * 0xD6E8FEB86659FD93) & _MASK_64)
        if black_keys is None:
            generated_black: list[int] = []
            generated_white: list[int] = []
            for _ in range(point_count):
                state, black_key = _splitmix64(state)
                state, white_key = _splitmix64(state)
                generated_black.append(black_key)
                generated_white.append(white_key)
            normalized_black = tuple(generated_black)
            normalized_white = tuple(generated_white)
        else:
            assert white_keys is not None
            normalized_black = _validate_keys(black_keys, point_count, "black_keys")
            normalized_white = _validate_keys(white_keys, point_count, "white_keys")

        if side_to_play_key is None:
            state, normalized_side_key = _splitmix64(state)
        elif (
            isinstance(side_to_play_key, bool)
            or not isinstance(side_to_play_key, int)
            or not 0 <= side_to_play_key <= _MASK_64
        ):
            raise ValueError("side_to_play_key must be an unsigned 64-bit integer")
        else:
            normalized_side_key = side_to_play_key

        object.__setattr__(self, "board_size", size)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "black_keys", normalized_black)
        object.__setattr__(self, "white_keys", normalized_white)
        object.__setattr__(self, "side_to_play_key", normalized_side_key)

    @classmethod
    def create(cls, board_size: int, seed: int = 0) -> ZobristTable:
        """Create a deterministic table (constructor-compatible convenience)."""

        return cls(board_size, seed)

    def key(self, point: int, color: Color) -> int:
        if not 0 <= point < self.board_size * self.board_size:
            raise ValueError("point is outside the board")
        if color is Color.BLACK:
            return self.black_keys[point]
        if color is Color.WHITE:
            return self.white_keys[point]
        raise ValueError("empty intersections do not have Zobrist keys")

    def hash_board(self, board: bytes | bytearray | Sequence[int]) -> int:
        """Hash stone placement only (the value used by positional superko)."""

        point_count = self.board_size * self.board_size
        if len(board) != point_count:
            raise ValueError(f"board must contain exactly {point_count} intersections")
        result = 0
        for point, raw_color in enumerate(board):
            if raw_color == Color.BLACK:
                result ^= self.black_keys[point]
            elif raw_color == Color.WHITE:
                result ^= self.white_keys[point]
            elif raw_color != Color.EMPTY:
                raise ValueError(f"invalid intersection value at point {point}: {raw_color!r}")
        return result
