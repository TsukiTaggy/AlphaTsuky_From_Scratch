"""Current-player-relative state encoding tests."""

import numpy as np
import pytest

from azgo.encoding import encode_state
from azgo.game import Color, GameState, Rules


def _board(*rows: str) -> bytes:
    values = {".": Color.EMPTY, "B": Color.BLACK, "W": Color.WHITE}
    return bytes(values[cell] for row in rows for cell in row)


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_encoding_has_expected_shape_and_dtype(board_size: int) -> None:
    features = encode_state(GameState.new(board_size))

    assert features.shape == (17, board_size, board_size)
    assert features.dtype == np.float32
    assert not features[:-1].any()
    assert features[-1].all()


def test_encoding_is_newest_first_and_relative_to_current_player() -> None:
    empty = bytes(25)
    after_black = _board("B....", ".....", ".....", ".....", ".....")
    current = _board("B....", ".....", "..W..", ".....", ".....")
    state = GameState.from_board(
        current,
        Rules(board_size=5),
        to_play=Color.BLACK,
        history=(empty, after_black, current),
    )

    features = encode_state(state, history_length=4)

    assert features.shape == (9, 5, 5)
    assert features[0, 0, 0] == 1.0
    assert features[1, 2, 2] == 1.0
    assert features[2, 0, 0] == 1.0
    assert not features[3].any()
    assert not features[4].any()
    assert not features[5].any()
    assert not features[6].any()
    assert not features[7].any()
    assert features[8].all()


def test_white_to_play_swaps_stone_planes_and_clears_color_plane() -> None:
    board = _board("B....", ".W...", ".....", ".....", ".....")
    state = GameState.from_board(board, Rules(board_size=5), to_play=Color.WHITE)

    features = encode_state(state, history_length=1)

    assert features[0, 1, 1] == 1.0
    assert features[1, 0, 0] == 1.0
    assert not features[-1].any()


def test_pass_is_represented_as_a_duplicate_position() -> None:
    state = GameState.new(5).apply(0)
    after_pass = state.apply(state.pass_action)

    features = encode_state(after_pass, history_length=2)

    np.testing.assert_array_equal(features[0], features[2])
    np.testing.assert_array_equal(features[1], features[3])


@pytest.mark.parametrize("history_length", [True, False, 0, -1, 1.5, "8"])
def test_encoding_rejects_invalid_history_length(history_length: object) -> None:
    with pytest.raises(ValueError, match="history_length"):
        encode_state(GameState.new(5), history_length)  # type: ignore[arg-type]
