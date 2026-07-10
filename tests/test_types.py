"""Rule and score value-type tests."""

import math

import pytest

from azgo.game import Color, Rules, Score


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_rules_accept_supported_board_sizes(board_size: int) -> None:
    rules = Rules(board_size=board_size, komi=6.5)

    assert rules.action_size == board_size * board_size + 1
    assert rules.pass_action == board_size * board_size


@pytest.mark.parametrize("board_size", [True, 0, 4, 6, 8, 10, 18, 20])
def test_rules_reject_unsupported_board_sizes(board_size: int) -> None:
    with pytest.raises(ValueError, match="board_size must be"):
        Rules(board_size=board_size)


def test_rules_reject_numeric_non_integer_board_size() -> None:
    with pytest.raises(ValueError, match="board_size must be"):
        Rules(board_size=5.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("komi", [math.inf, -math.inf, math.nan, "not-a-number"])
def test_rules_reject_non_finite_komi(komi: object) -> None:
    with pytest.raises(ValueError, match="komi must be"):
        Rules(komi=komi)  # type: ignore[arg-type]


def test_empty_color_has_no_opponent() -> None:
    with pytest.raises(ValueError, match="empty color"):
        _ = Color.EMPTY.opponent


def test_score_reports_margin_winner_and_perspective() -> None:
    score = Score(
        black_stones=7,
        white_stones=5,
        black_territory=3,
        white_territory=1,
        neutral_points=9,
        komi=2.5,
    )

    assert score.black == 10.0
    assert score.white == 8.5
    assert score.black_margin == 1.5
    assert score.winner is Color.BLACK
    assert score.outcome(Color.BLACK) == 1
    assert score.outcome(Color.WHITE) == -1


def test_score_can_be_a_draw() -> None:
    score = Score(1, 1, 2, 2, 19, 0.0)

    assert score.winner is None
    assert score.outcome(Color.BLACK) == 0
    assert score.outcome(Color.WHITE) == 0
