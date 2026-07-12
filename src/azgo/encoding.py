"""Neural-network feature encoding for immutable Go game states."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from azgo.game import Color, GameState

if TYPE_CHECKING:
    from numpy.typing import NDArray


def encode_state(state: GameState, history_length: int = 8) -> NDArray[np.float32]:
    """Encode ``state`` as current-player-relative ``[C, N, N]`` features.

    The first ``2 * history_length`` planes contain position pairs ordered from
    newest to oldest. Within each pair, the first plane marks stones belonging
    to the player to move in ``state`` and the second marks their opponent's
    stones. Unknown history is represented by all-zero pairs. The final plane
    is one everywhere when Black is to play and zero everywhere when White is
    to play.

    Because :class:`GameState` records a board after every action, passes are
    represented naturally as consecutive duplicate position pairs.
    """

    if (
        isinstance(history_length, bool)
        or not isinstance(history_length, int)
        or history_length <= 0
    ):
        raise ValueError("history_length must be a positive integer")

    board_size = state.board_size
    features = np.zeros(
        (2 * history_length + 1, board_size, board_size),
        dtype=np.float32,
    )
    current_player = state.to_play
    opponent = current_player.opponent

    for time_step, board in enumerate(reversed(state.history[-history_length:])):
        position = np.frombuffer(board, dtype=np.uint8).reshape(board_size, board_size)
        features[2 * time_step] = position == current_player
        features[2 * time_step + 1] = position == opponent

    if current_player is Color.BLACK:
        features[-1].fill(1.0)
    return features
