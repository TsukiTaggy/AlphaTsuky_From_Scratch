"""Property-based and randomized legal-game invariants."""

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from azgo.game import Color, GameOverError, GameState, Rules


def _assert_all_groups_have_liberties(state: GameState) -> None:
    seen: set[int] = set()
    for point, raw_color in enumerate(state.board):
        if raw_color == Color.EMPTY or point in seen:
            continue
        group = state.group_at(point)
        assert group is not None
        assert group.liberties
        seen.update(group.stones)


def _groups_captured_by(state: GameState, action: int) -> frozenset[int]:
    if action == state.pass_action:
        return frozenset()
    captured: set[int] = set()
    for neighbor in state.neighbors(action):
        group = state.group_at(neighbor)
        if (
            group is not None
            and group.color is state.to_play.opponent
            and group.liberties == frozenset({action})
        ):
            captured.update(group.stones)
    return frozenset(captured)


@settings(max_examples=32, deadline=None)
@given(
    board_size=st.sampled_from([5, 9, 13, 19]),
    seed=st.integers(min_value=0, max_value=(2**32) - 1),
)
def test_random_legal_sequences_preserve_state_invariants(board_size: int, seed: int) -> None:
    rng = random.Random(seed)
    state = GameState.new(Rules(board_size=board_size), zobrist_seed=seed)
    steps = min(board_size * board_size, 32)

    for _ in range(steps):
        mask = state.legal_action_mask()
        legal_from_mask = tuple(int(action) for action in np.flatnonzero(mask))
        assert legal_from_mask == state.legal_actions()
        assert all(state.is_legal(action) for action in legal_from_mask)

        placements = [action for action in legal_from_mask if action != state.pass_action]
        action = rng.choice(placements) if placements and rng.random() < 0.9 else state.pass_action
        board_before = state.board
        history_before = state.history
        hashes_before = state.hash_history
        captured = _groups_captured_by(state, action)

        child = state.apply(action)

        assert state.board == board_before
        assert state.history == history_before
        assert state.hash_history == hashes_before
        assert child.history[:-1] == state.history
        assert child.hash_history[:-1] == state.hash_history
        for point in captured:
            assert child.stone_at(point) is Color.EMPTY
        _assert_all_groups_have_liberties(child)
        state = child
        if state.is_terminal:
            break

    while not state.is_terminal:
        state = state.apply(state.pass_action)
    assert not state.legal_action_mask().any()
    with pytest.raises(GameOverError, match="after two consecutive passes"):
        state.apply(0)


def test_one_hundred_seeded_random_games_never_play_illegal_actions() -> None:
    for game_seed in range(100):
        rng = random.Random(game_seed)
        state = GameState.new(Rules(board_size=5, komi=5.5), zobrist_seed=991)

        for _ in range(60):
            if state.is_terminal:
                break
            legal = state.legal_actions()
            placements = [action for action in legal if action != state.pass_action]
            action = (
                rng.choice(placements)
                if placements and rng.random() < 0.9
                else state.pass_action
            )
            assert state.is_legal(action), (game_seed, state.move_number, action)
            state = state.apply(action)

        while not state.is_terminal:
            assert state.is_legal(state.pass_action)
            state = state.apply(state.pass_action)

        assert state.is_terminal
        _assert_all_groups_have_liberties(state)
