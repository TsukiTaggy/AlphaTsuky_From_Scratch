"""Tromp-Taylor area scoring and terminal outcome tests."""

from azgo.game import Color, GameState, Rules


def _board(*rows: str) -> bytes:
    values = {".": Color.EMPTY, "B": Color.BLACK, "W": Color.WHITE}
    return bytes(values[cell] for row in rows for cell in row)


def test_single_color_border_owns_the_empty_region() -> None:
    state = GameState.from_board(
        _board("BBBBB", "B...B", "B.B.B", "B...B", "BBBBB"),
        Rules(board_size=5, komi=0.0),
    )
    score = state.score()

    assert score.black_stones == 17
    assert score.black_territory == 8
    assert score.white == 0.0
    assert score.black == 25.0


def test_mixed_border_makes_empty_region_neutral() -> None:
    state = GameState.from_board(
        _board("BBBBB", "B...B", "B.W.B", "B...B", "BBBBB"),
        Rules(board_size=5, komi=0.0),
    )
    score = state.score()

    assert score.black_stones == 16
    assert score.white_stones == 1
    assert score.black_territory == 0
    assert score.white_territory == 0
    assert score.neutral_points == 8


def test_komi_is_added_to_white_and_changes_outcome() -> None:
    state = GameState.from_board(
        bytes(25),
        Rules(board_size=5, komi=0.5),
        consecutive_passes=2,
    )
    score = state.score()

    assert score.black == 0.0
    assert score.white == 0.5
    assert score.winner is Color.WHITE
    assert state.outcome(Color.BLACK) == -1
    assert state.outcome(Color.WHITE) == 1

