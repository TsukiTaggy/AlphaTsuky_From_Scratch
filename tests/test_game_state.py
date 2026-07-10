"""Core immutable Go state and legality tests."""

from collections.abc import Iterable

import numpy as np
import pytest

from azgo.game import (
    Color,
    GameNotFinishedError,
    GameOverError,
    GameState,
    InvalidActionError,
    OccupiedPointError,
    Rules,
    SuicideError,
    SuperkoError,
    ZobristTable,
)

Coord = tuple[int, int]


def _action(row: int, col: int, board_size: int = 5) -> int:
    return row * board_size + col


def _play(state: GameState, moves: Iterable[Coord | None]) -> GameState:
    for move in moves:
        action = state.pass_action if move is None else state.coord_to_action(*move)
        state = state.apply(action)
    return state


def _board(*rows: str) -> bytes:
    values = {".": Color.EMPTY, "B": Color.BLACK, "W": Color.WHITE}
    return bytes(values[cell] for row in rows for cell in row)


def test_new_game_has_expected_state_and_legal_mask() -> None:
    state = GameState.new(Rules(board_size=5, komi=5.5), zobrist_seed=19)

    assert state.to_play is Color.BLACK
    assert state.move_number == 0
    assert state.consecutive_passes == 0
    assert state.board == bytes(25)
    assert state.history == (bytes(25),)
    assert state.position_hash == state.zobrist.hash_board(state.board)
    assert state.legal_actions() == tuple(range(26))
    mask = state.legal_action_mask()
    assert mask.shape == (26,)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_legal_mask_is_fresh_and_cannot_mutate_state() -> None:
    state = GameState.new(5)
    first = state.legal_action_mask()
    first[:] = False

    assert state.legal_action_mask().all()


def test_group_stones_and_liberties_are_exact() -> None:
    state = _play(GameState.new(5), [(0, 0), None, (0, 1)])
    group = state.group_at(_action(0, 0))

    assert group is not None
    assert group.color is Color.BLACK
    assert group.stones == frozenset({_action(0, 0), _action(0, 1)})
    assert group.liberties == frozenset({_action(1, 0), _action(1, 1), _action(0, 2)})
    assert state.group_at(_action(4, 4)) is None


def test_single_stone_capture_removes_the_stone() -> None:
    state = _play(
        GameState.new(5),
        [(0, 1), (1, 1), (1, 0), (4, 4), (1, 2), (4, 3), (2, 1)],
    )

    assert state.stone_at(_action(1, 1)) is Color.EMPTY
    assert state.group_at(_action(2, 1)) is not None


def test_multi_stone_capture_removes_the_entire_group() -> None:
    state = _play(
        GameState.new(5),
        [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, 2),
            (0, 2),
            (4, 4),
            (1, 3),
            (4, 3),
            (2, 1),
            (3, 4),
            (2, 2),
        ],
    )

    assert state.stone_at(_action(1, 1)) is Color.EMPTY
    assert state.stone_at(_action(1, 2)) is Color.EMPTY


def test_suicide_is_illegal_and_capture_into_atari_is_legal() -> None:
    suicide_state = _play(
        GameState.new(5),
        [(0, 1), (4, 4), (1, 0), (4, 3), (1, 2), (3, 4), (2, 1)],
    )
    center = _action(1, 1)

    assert not suicide_state.is_legal(center)
    with pytest.raises(SuicideError, match="suicide"):
        suicide_state.apply(center)

    capture_state = GameState.from_board(
        _board("WBW..", "B.B..", ".B...", ".....", "....."),
        Rules(board_size=5),
        to_play=Color.WHITE,
    )
    assert capture_state.is_legal(center)
    assert capture_state.apply(center).stone_at(_action(0, 1)) is Color.EMPTY


def test_simple_ko_recapture_is_forbidden() -> None:
    before_capture = _play(
        GameState.new(5),
        [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (4, 4),
            (1, 3),
        ],
    )
    after_capture = before_capture.apply(_action(1, 2))
    recapture = _action(1, 1)

    assert after_capture.stone_at(recapture) is Color.EMPTY
    assert not after_capture.is_legal(recapture)
    with pytest.raises(SuperkoError, match="superko"):
        after_capture.apply(recapture)


def test_positional_superko_checks_non_immediate_history() -> None:
    repeated_result = _board("B....", ".....", ".....", ".....", ".....")
    intervening = _board(".W...", ".....", ".....", ".....", ".....")
    current = bytes(25)
    state = GameState.from_board(
        current,
        Rules(board_size=5),
        to_play=Color.BLACK,
        history=(repeated_result, intervening, current),
    )

    assert not state.is_legal(0)
    with pytest.raises(SuperkoError, match="superko"):
        state.apply(0)


def test_hash_collision_does_not_cause_false_superko() -> None:
    zeroes = (0,) * 25
    zobrist = ZobristTable(
        5,
        black_keys=zeroes,
        white_keys=zeroes,
        side_to_play_key=0,
    )
    state = GameState.new(Rules(board_size=5), zobrist=zobrist)

    assert state.position_hash == 0
    assert state.is_legal(0)
    assert state.apply(0).position_hash == 0


def test_pass_is_superko_exempt_and_two_passes_end_game() -> None:
    state = GameState.new(Rules(board_size=5, komi=0.0))
    after_one = state.apply(state.pass_action)
    terminal = after_one.apply(after_one.pass_action)

    assert after_one.board == state.board
    assert after_one.history == (state.board, state.board)
    assert not after_one.is_terminal
    assert terminal.is_terminal
    assert terminal.history == (state.board, state.board, state.board)
    assert terminal.legal_actions() == ()
    assert not terminal.legal_action_mask().any()
    assert not terminal.is_legal(terminal.pass_action)
    assert terminal.outcome(Color.BLACK) == 0
    assert terminal.outcome(Color.WHITE) == 0
    with pytest.raises(GameOverError, match="after two consecutive passes"):
        terminal.apply(terminal.pass_action)


def test_placement_resets_consecutive_pass_count() -> None:
    state = GameState.new(5)
    after_pass = state.apply(state.pass_action)
    after_move = after_pass.apply(0)
    after_another_pass = after_move.apply(after_move.pass_action)

    assert after_pass.consecutive_passes == 1
    assert after_move.consecutive_passes == 0
    assert after_another_pass.consecutive_passes == 1
    assert not after_another_pass.is_terminal


def test_outcome_requires_terminal_state() -> None:
    with pytest.raises(GameNotFinishedError, match="only after"):
        GameState.new(5).outcome(Color.BLACK)


def test_parent_state_and_all_reachable_storage_remain_unchanged() -> None:
    parent = _play(GameState.new(5), [(0, 0), (4, 4)])
    board_before = parent.board
    history_before = parent.history
    hashes_before = parent.hash_history
    child = parent.apply(_action(0, 1))

    assert parent.board == board_before
    assert parent.history == history_before
    assert parent.hash_history == hashes_before
    assert child.board != parent.board
    assert child.history[:-1] == parent.history
    with pytest.raises(TypeError):
        parent.board[0] = Color.WHITE  # type: ignore[index]


def test_illegal_attempts_do_not_mutate_parent() -> None:
    state = GameState.new(5).apply(0)
    snapshot = (state.board, state.history, state.hash_history, state.to_play)

    with pytest.raises(OccupiedPointError, match="occupied"):
        state.apply(0)
    assert (state.board, state.history, state.hash_history, state.to_play) == snapshot


@pytest.mark.parametrize("action", [-1, 26, True, 1.5, "0"])
def test_invalid_actions_are_never_legal(action: object) -> None:
    state = GameState.new(5)

    assert not state.is_legal(action)  # type: ignore[arg-type]
    with pytest.raises(InvalidActionError, match="action"):
        state.apply(action)  # type: ignore[arg-type]
