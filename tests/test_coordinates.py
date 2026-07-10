"""Action-space and coordinate conversion tests."""

import pytest

from azgo.game import action_to_coord, coord_to_action, pass_action


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_coordinate_actions_are_row_major(board_size: int) -> None:
    for row in range(board_size):
        for col in range(board_size):
            action = coord_to_action(row, col, board_size)
            assert action == row * board_size + col
            assert action_to_coord(action, board_size) == (row, col)


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_pass_is_final_action(board_size: int) -> None:
    action = pass_action(board_size)

    assert action == board_size * board_size
    assert action_to_coord(action, board_size) is None


@pytest.mark.parametrize(
    ("row", "col"),
    [(-1, 0), (0, -1), (5, 0), (0, 5)],
)
def test_invalid_coordinates_are_rejected(row: int, col: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        coord_to_action(row, col, 5)


@pytest.mark.parametrize("action", [-1, 26])
def test_invalid_actions_are_rejected(action: int) -> None:
    with pytest.raises(ValueError, match="action must be"):
        action_to_coord(action, 5)


@pytest.mark.parametrize("board_size", [0, 4, 6, 8, 10, 18, 20])
def test_unsupported_board_sizes_are_rejected(board_size: int) -> None:
    with pytest.raises(ValueError, match="board_size must be"):
        pass_action(board_size)
