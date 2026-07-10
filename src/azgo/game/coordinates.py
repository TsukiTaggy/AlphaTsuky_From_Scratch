"""Row-major action and coordinate conversion helpers."""

from __future__ import annotations

import operator

from .types import SUPPORTED_BOARD_SIZES


def _validate_board_size(board_size: int) -> int:
    if isinstance(board_size, bool) or board_size not in SUPPORTED_BOARD_SIZES:
        supported = ", ".join(str(size) for size in sorted(SUPPORTED_BOARD_SIZES))
        raise ValueError(f"board_size must be one of {{{supported}}}")
    return board_size


def pass_action(board_size: int) -> int:
    """Return the pass action for ``board_size``."""

    size = _validate_board_size(board_size)
    return size * size


def coord_to_action(row: int, col: int, board_size: int) -> int:
    """Convert a zero-based ``(row, col)`` coordinate to a row-major action."""

    size = _validate_board_size(board_size)
    if isinstance(row, bool) or isinstance(col, bool):
        raise ValueError("row and col must be integer coordinates")
    try:
        normalized_row = operator.index(row)
        normalized_col = operator.index(col)
    except TypeError as exc:
        raise ValueError("row and col must be integer coordinates") from exc
    if not 0 <= normalized_row < size or not 0 <= normalized_col < size:
        raise ValueError(
            f"coordinate ({normalized_row}, {normalized_col}) is outside {size}x{size}"
        )
    return normalized_row * size + normalized_col


def action_to_coord(action: int, board_size: int) -> tuple[int, int] | None:
    """Convert an action to ``(row, col)``; pass converts to ``None``."""

    size = _validate_board_size(board_size)
    if isinstance(action, bool):
        raise ValueError("action must be an integer")
    try:
        normalized = operator.index(action)
    except TypeError as exc:
        raise ValueError("action must be an integer") from exc
    maximum = size * size
    if not 0 <= normalized <= maximum:
        raise ValueError(f"action must be in [0, {maximum}]")
    if normalized == maximum:
        return None
    return divmod(normalized, size)
