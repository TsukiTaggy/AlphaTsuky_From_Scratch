"""Tests for deterministic paired arena evaluation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from azgo.arena import (
    ArenaError,
    ArenaGameError,
    ArenaGameResult,
    ArenaLimitError,
    ArenaResult,
    ArenaRunner,
    CandidateOutcome,
)
from azgo.config import AppConfig, load_config
from azgo.evaluator import EvaluationBatch
from azgo.game import Color, GameState, Rules, Score
from azgo.search import SearchResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class PassEvaluator:
    """Prefer pass strongly enough for a one-simulation search to select it."""

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        logits = np.full((len(states), states[0].action_size), -100.0, dtype=np.float32)
        for row, state in enumerate(states):
            logits[row, state.pass_action] = 100.0
        return EvaluationBatch(logits, np.zeros(len(states), dtype=np.float32))


class ExplodingEvaluator:
    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        del states
        raise RuntimeError("evaluation exploded")


def _config(
    *,
    games: int = 2,
    opening_moves: int = 1,
    max_moves: int = 8,
    threshold: float = 0.55,
    simulations: int = 1,
) -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "engine" / "go5.yaml")
    return config.model_copy(
        update={
            "search": config.search.model_copy(update={"simulations": simulations}),
            "arena": config.arena.model_copy(
                update={
                    "games": games,
                    "opening_moves": opening_moves,
                    "max_moves": max_moves,
                    "promotion_threshold": threshold,
                }
            ),
        }
    )


def _score(winner: Color | None) -> Score:
    return Score(
        black_stones=25 if winner is Color.BLACK else 0,
        white_stones=25 if winner is Color.WHITE else 0,
        black_territory=0,
        white_territory=0,
        neutral_points=25 if winner is None else 0,
        komi=0.0,
    )


def _game(
    pair_index: int,
    candidate_color: Color,
    outcome: CandidateOutcome,
    *,
    opening: tuple[int, ...] = (0,),
) -> ArenaGameResult:
    if outcome == "draw":
        winner = None
    elif outcome == "win":
        winner = candidate_color
    else:
        winner = candidate_color.opponent
    return ArenaGameResult(
        pair_index=pair_index,
        game_index=pair_index * 2 + (candidate_color is Color.WHITE),
        candidate_color=candidate_color,
        opening_actions=opening,
        move_count=len(opening) + 2,
        final_score=_score(winner),
        winner=winner,
        candidate_outcome=outcome,
    )


def test_seeded_openings_are_model_independent_legal_and_reproducible() -> None:
    config = _config(games=4, opening_moves=4)
    runner = ArenaRunner(ExplodingEvaluator(), ExplodingEvaluator(), config)

    first_state, first_actions = runner._generate_opening(0)
    repeated_state, repeated_actions = runner._generate_opening(0)
    other_state, other_actions = runner._generate_opening(1)

    assert first_actions == repeated_actions
    assert first_state == repeated_state
    assert len(first_actions) == 4
    assert first_actions != other_actions
    assert first_state != other_state

    state = GameState.new(
        Rules(board_size=config.game.board_size, komi=config.game.komi),
        zobrist_seed=config.zobrist.seed,
    )
    for action in first_actions:
        assert action != state.pass_action
        assert state.is_legal(action)
        state = state.apply(action)
    assert state == first_state


def test_runner_reuses_two_trees_advances_both_and_never_adds_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[SpyMCTS] = []

    class SpyMCTS:
        def __init__(self, evaluator: object, **settings: object) -> None:
            self.evaluator = evaluator
            self.settings = settings
            self.state: GameState | None = None
            self.resets: list[GameState] = []
            self.run_colors: list[Color] = []
            self.noise: list[bool] = []
            self.advances: list[int] = []
            instances.append(self)

        def reset(self, state: GameState) -> None:
            self.state = state
            self.resets.append(state)

        def run(self, state: GameState, *, add_root_noise: bool = False) -> SearchResult:
            assert self.state == state
            self.run_colors.append(state.to_play)
            self.noise.append(add_root_noise)
            counts = np.zeros(state.action_size, dtype=np.int64)
            counts[state.pass_action] = 1
            return SearchResult(
                state.pass_action,
                counts,
                counts.astype(np.float32),
                0.0,
                1,
            )

        def advance(self, action: int) -> GameState:
            assert self.state is not None
            self.advances.append(action)
            self.state = self.state.apply(action)
            return self.state

    monkeypatch.setattr("azgo.arena.MCTS", SpyMCTS)
    candidate = PassEvaluator()
    incumbent = PassEvaluator()
    config = _config()
    result = ArenaRunner(candidate, incumbent, config).run()

    assert len(instances) == 2
    assert instances[0].evaluator is candidate
    assert instances[1].evaluator is incumbent
    assert all(instance.settings["simulations"] == 1 for instance in instances)
    assert all(
        instance.settings["c_puct"] == config.search.c_puct for instance in instances
    )
    assert all(isinstance(instance.settings["seed"], int) for instance in instances)
    assert [len(instance.resets) for instance in instances] == [2, 2]
    assert [len(instance.advances) for instance in instances] == [4, 4]
    assert instances[0].noise == [False, False]
    assert instances[1].noise == [False, False]
    assert instances[0].run_colors == [Color.BLACK, Color.WHITE]
    assert instances[1].run_colors == [Color.WHITE, Color.BLACK]
    assert result.games[0].opening_actions == result.games[1].opening_actions
    assert [game.candidate_color for game in result.games] == [Color.BLACK, Color.WHITE]


def test_real_pass_arena_is_repeatable_and_scores_color_swapped_games() -> None:
    config = _config(games=4, opening_moves=2)
    runner = ArenaRunner(PassEvaluator(), PassEvaluator(), config)

    first = runner.run()
    repeated = runner.run()

    assert first == repeated
    assert len(first.games) == 4
    assert first.candidate_wins == 2
    assert first.incumbent_wins == 2
    assert first.draws == 0
    assert first.candidate_points == 2.0
    assert first.candidate_score == 0.5
    assert not first.promotion_eligible
    for game in first.games:
        assert game.move_count == config.arena.opening_moves + 2
        assert game.winner is Color.WHITE


def test_draws_are_half_points_and_threshold_is_inclusive() -> None:
    result = ArenaResult(
        (
            _game(0, Color.BLACK, "win"),
            _game(0, Color.WHITE, "draw"),
        ),
        0.75,
    )

    assert result.candidate_wins == 1
    assert result.incumbent_wins == 0
    assert result.draws == 1
    assert result.candidate_points == 1.5
    assert result.candidate_score == 0.75
    assert result.promotion_eligible


def test_move_limit_counts_opening_moves_and_aborts_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlacementMCTS:
        def __init__(self, evaluator: object, **settings: object) -> None:
            del evaluator, settings
            self.state: GameState | None = None

        def reset(self, state: GameState) -> None:
            self.state = state

        def run(self, state: GameState, *, add_root_noise: bool = False) -> SearchResult:
            assert not add_root_noise
            action = next(action for action in state.legal_actions() if action != state.pass_action)
            counts = np.zeros(state.action_size, dtype=np.int64)
            counts[action] = 1
            return SearchResult(action, counts, counts.astype(np.float32), 0.0, 1)

        def advance(self, action: int) -> GameState:
            assert self.state is not None
            self.state = self.state.apply(action)
            return self.state

    monkeypatch.setattr("azgo.arena.MCTS", PlacementMCTS)
    runner = ArenaRunner(
        PassEvaluator(),
        PassEvaluator(),
        _config(opening_moves=1, max_moves=2),
    )

    with pytest.raises(ArenaLimitError, match=r"game 0 reached max_moves=2"):
        runner.run()


def test_evaluator_failure_is_wrapped_as_typed_game_error() -> None:
    runner = ArenaRunner(
        ExplodingEvaluator(),
        PassEvaluator(),
        _config(opening_moves=0),
    )

    with pytest.raises(ArenaGameError, match="arena game 0 failed") as raised:
        runner.run()

    assert isinstance(raised.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"pair_index": True}, "pair_index must be a nonnegative integer"),
        ({"game_index": 1}, "game_index must identify"),
        ({"candidate_color": Color.EMPTY}, "candidate_color"),
        ({"opening_actions": (25,)}, "non-pass"),
        ({"move_count": 2}, "two passes"),
        ({"winner": Color.WHITE}, "winner must match"),
        ({"candidate_outcome": "loss"}, "candidate_outcome must match"),
    ],
)
def test_game_result_rejects_inconsistent_public_data(
    updates: dict[str, object],
    match: str,
) -> None:
    values: dict[str, object] = {
        "pair_index": 0,
        "game_index": 0,
        "candidate_color": Color.BLACK,
        "opening_actions": (0,),
        "move_count": 3,
        "final_score": _score(Color.BLACK),
        "winner": Color.BLACK,
        "candidate_outcome": "win",
    }
    values.update(updates)

    with pytest.raises(ArenaError, match=match):
        ArenaGameResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("games", "threshold", "match"),
    [
        ((), 0.55, "positive even"),
        ((_game(0, Color.BLACK, "win"),), 0.55, "positive even"),
        (
            (
                _game(0, Color.BLACK, "win", opening=(0,)),
                _game(0, Color.WHITE, "loss", opening=(1,)),
            ),
            0.55,
            "same opening",
        ),
        (
            (_game(0, Color.BLACK, "win"), _game(0, Color.WHITE, "loss")),
            0.5,
            "promotion_threshold",
        ),
        (
            (_game(0, Color.BLACK, "win"), _game(0, Color.WHITE, "loss")),
            1,
            "promotion_threshold",
        ),
    ],
)
def test_aggregate_rejects_invalid_pairing_and_thresholds(
    games: tuple[ArenaGameResult, ...],
    threshold: object,
    match: str,
) -> None:
    with pytest.raises(ArenaError, match=match):
        ArenaResult(games, threshold)  # type: ignore[arg-type]


def test_results_are_frozen() -> None:
    game = _game(0, Color.BLACK, "win")
    result = ArenaResult((game, _game(0, Color.WHITE, "loss")), 0.55)

    with pytest.raises(FrozenInstanceError):
        game.move_count = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidate_score = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("candidate", "incumbent", "config", "match"),
    [
        (object(), PassEvaluator(), _config(), "candidate_evaluator"),
        (PassEvaluator(), object(), _config(), "incumbent_evaluator"),
        (PassEvaluator(), PassEvaluator(), object(), "config"),
    ],
)
def test_runner_rejects_invalid_constructor_inputs(
    candidate: object,
    incumbent: object,
    config: object,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        ArenaRunner(candidate, incumbent, config)  # type: ignore[arg-type]
