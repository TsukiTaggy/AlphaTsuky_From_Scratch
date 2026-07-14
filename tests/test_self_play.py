"""Tests for deterministic self-play generation and immutable examples."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event
from threading import enumerate as enumerate_threads
from time import sleep
from typing import TYPE_CHECKING, Self

import numpy as np
import pytest

from azgo.config import AppConfig, load_config
from azgo.evaluator import EvaluationBatch, Evaluator
from azgo.game import Color, GameState, Score
from azgo.inference import (
    DeterministicInferenceCoordinator,
    InferenceClient,
    InferenceClosedError,
    InferenceMetrics,
)
from azgo.search import SearchResult
from azgo.self_play import (
    ParallelSelfPlayError,
    ParallelSelfPlayResult,
    ParallelSelfPlayRunner,
    SelfPlayError,
    SelfPlayGame,
    SelfPlayLimitError,
    SelfPlayRunner,
    TrainingSample,
    _assigned_game_indices,
    _random_streams,
    _sample_with_temperature,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from concurrent.futures import Future
    from types import TracebackType


class PassEvaluator:
    """An evaluator that makes a pass the only practically selectable move."""

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        logits = np.full((len(states), states[0].action_size), -100.0, dtype=np.float32)
        for row, state in enumerate(states):
            logits[row, state.pass_action] = 100.0
        return EvaluationBatch(logits, np.zeros(len(states), dtype=np.float32))


def _config(
    root: Path,
    *,
    simulations: int = 1,
    max_moves: int = 8,
    temperature: float = 1.0,
    temperature_moves: int = 8,
    root_noise: bool = False,
    komi: float = 5.5,
    games: int = 1,
    workers: int = 1,
    max_batch_size: int = 16,
) -> AppConfig:
    config = load_config(root / "configs" / "engine" / "go5.yaml")
    return config.model_copy(
        update={
            "game": config.game.model_copy(update={"komi": komi}),
            "search": config.search.model_copy(update={"simulations": simulations}),
            "self_play": config.self_play.model_copy(
                update={
                    "max_moves": max_moves,
                    "temperature": temperature,
                    "temperature_moves": temperature_moves,
                    "root_noise": root_noise,
                    "games": games,
                    "workers": workers,
                }
            ),
            "inference": config.inference.model_copy(
                update={"max_batch_size": max_batch_size}
            ),
        }
    )


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((3, 5, 5), dtype=np.float32)
    policy = np.zeros(26, dtype=np.float32)
    policy[25] = 1.0
    return features, policy


def _sample(
    *,
    value: float = -1.0,
    to_play: Color = Color.BLACK,
    move_number: int = 0,
    selected_action: int = 25,
    game_index: int = 3,
) -> TrainingSample:
    features, policy = _arrays()
    return TrainingSample(
        features,
        policy,
        value,
        to_play,
        move_number,
        selected_action,
        game_index,
    )


def test_pass_game_stores_pre_move_features_and_policy_then_labels_terminal_result(
) -> None:
    root = Path(__file__).resolve().parents[1]
    game = SelfPlayRunner(PassEvaluator(), _config(root)).play_game(11)

    assert game.actions == (25, 25)
    assert game.game_index == 11
    assert game.winner is Color.WHITE
    assert game.final_score.white == pytest.approx(5.5)
    assert [sample.to_play for sample in game.samples] == [Color.BLACK, Color.WHITE]
    assert [sample.move_number for sample in game.samples] == [0, 1]
    assert [sample.value for sample in game.samples] == [-1.0, 1.0]
    assert [sample.selected_action for sample in game.samples] == [25, 25]
    assert game.samples[0].features[:-1].sum() == 0.0
    assert game.samples[0].features[-1].all()
    assert game.samples[1].features[:-1].sum() == 0.0
    assert not game.samples[1].features[-1].any()
    np.testing.assert_array_equal(game.samples[0].policy, np.eye(1, 26, 25, dtype=np.float32)[0])


def test_empty_board_draw_labels_both_players_zero() -> None:
    root = Path(__file__).resolve().parents[1]

    game = SelfPlayRunner(PassEvaluator(), _config(root, komi=0.0)).play_game(0)

    assert game.winner is None
    assert [sample.value for sample in game.samples] == [0.0, 0.0]


def test_seed_streams_are_reproducible_and_game_indices_are_independent() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root)

    first_seed, first_rng = _random_streams(config, 19)
    repeated_seed, repeated_rng = _random_streams(config, 19)
    other_seed, other_rng = _random_streams(config, 20)
    first_values = first_rng.integers(0, 2**32, size=8)
    repeated_values = repeated_rng.integers(0, 2**32, size=8)
    other_values = other_rng.integers(0, 2**32, size=8)

    assert first_seed == repeated_seed
    assert first_seed != other_seed
    np.testing.assert_array_equal(first_values, repeated_values)
    assert not np.array_equal(first_values, other_values)


def test_replaying_a_game_index_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = SelfPlayRunner(PassEvaluator(), _config(root))

    first = runner.play_game(4)
    second = runner.play_game(4)

    assert first.actions == second.actions
    assert first.final_score == second.final_score
    for left, right in zip(first.samples, second.samples, strict=True):
        np.testing.assert_array_equal(left.features, right.features)
        np.testing.assert_array_equal(left.policy, right.policy)
        assert left.value == right.value


def test_temperature_sampling_uses_visit_count_power_and_never_samples_zero() -> None:
    counts = np.asarray([1, 0, 4], dtype=np.int64)
    actual_rng = np.random.default_rng(123)
    expected_rng = np.random.default_rng(123)
    expected = int(expected_rng.choice(np.asarray([0, 2]), p=np.asarray([1 / 3, 2 / 3])))

    assert _sample_with_temperature(counts, 2.0, actual_rng) == expected
    assert {
        _sample_with_temperature(counts, 2.0, np.random.default_rng(seed))
        for seed in range(20)
    } <= {0, 2}


def test_runner_switches_from_temperature_sampling_to_search_argmax() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = SelfPlayRunner(
        PassEvaluator(),
        _config(root, temperature_moves=1, temperature=1.0),
    )
    counts = np.asarray([1, 9, 0], dtype=np.int64)
    policy = counts.astype(np.float32) / 10.0
    result = SearchResult(1, counts, policy, 0.0, 10)

    assert runner._select_action(result, 0, np.random.default_rng(2)) in {0, 1}
    assert runner._select_action(result, 1, np.random.default_rng(2)) == 1


def test_root_noise_and_every_action_are_forwarded_to_one_reused_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    instances: list[SpyMCTS] = []

    class SpyMCTS:
        def __init__(self, evaluator: object, **settings: object) -> None:
            del evaluator
            self.settings = settings
            self.state: GameState | None = None
            self.noise: list[bool] = []
            self.advances: list[int] = []
            instances.append(self)

        def run(self, state: GameState, *, add_root_noise: bool = False) -> SearchResult:
            assert self.state is None or self.state == state
            self.state = state
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

    monkeypatch.setattr("azgo.self_play.MCTS", SpyMCTS)
    game = SelfPlayRunner(
        PassEvaluator(),
        _config(root, root_noise=True, temperature_moves=0),
    ).play_game(9)

    assert game.actions == (25, 25)
    assert len(instances) == 1
    assert instances[0].noise == [True, True]
    assert instances[0].advances == [25, 25]
    assert isinstance(instances[0].settings["seed"], int)


def test_move_limit_discards_unfinished_game(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]

    class PlacementMCTS:
        def __init__(self, evaluator: object, **settings: object) -> None:
            del evaluator, settings
            self.state: GameState | None = None

        def run(self, state: GameState, *, add_root_noise: bool = False) -> SearchResult:
            del add_root_noise
            self.state = state
            action = state.legal_actions()[0]
            counts = np.zeros(state.action_size, dtype=np.int64)
            counts[action] = 1
            return SearchResult(action, counts, counts.astype(np.float32), 0.0, 1)

        def advance(self, action: int) -> GameState:
            assert self.state is not None
            self.state = self.state.apply(action)
            return self.state

    monkeypatch.setattr("azgo.self_play.MCTS", PlacementMCTS)
    runner = SelfPlayRunner(
        PassEvaluator(),
        _config(root, max_moves=2, temperature_moves=0),
    )

    with pytest.raises(SelfPlayLimitError, match="max_moves=2"):
        runner.play_game(8)


def test_training_sample_copies_casts_and_protects_arrays() -> None:
    base_features = np.zeros((3, 5, 5), dtype=np.float64)
    features = base_features[:, :, ::-1]
    policy = np.zeros(26, dtype=np.float64)
    policy[25] = 1.0

    sample = TrainingSample(
        features,  # type: ignore[arg-type]
        policy,  # type: ignore[arg-type]
        -1,
        Color.BLACK,
        0,
        25,
        2,
    )
    base_features.fill(3.0)
    policy.fill(0.0)

    assert sample.features.dtype == np.float32
    assert sample.policy.dtype == np.float32
    assert sample.features.flags.c_contiguous
    assert sample.policy.flags.c_contiguous
    assert not sample.features.flags.writeable
    assert not sample.policy.flags.writeable
    assert sample.features.sum() == 0.0
    assert sample.policy[25] == 1.0
    assert sample.value == -1.0
    with pytest.raises(ValueError, match="read-only"):
        sample.features[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        sample.policy[25] = 0.0


@pytest.mark.parametrize(
    ("features", "policy", "match"),
    [
        (np.zeros((5, 5), dtype=np.float32), np.eye(1, 26, dtype=np.float32)[0], "shape"),
        (np.zeros((3, 5, 4), dtype=np.float32), np.eye(1, 21, dtype=np.float32)[0], "square"),
        (np.zeros((3, 5, 5), dtype=np.float32), np.zeros(25, dtype=np.float32), "shape"),
        (
            np.full((3, 5, 5), np.nan, dtype=np.float32),
            np.eye(1, 26, dtype=np.float32)[0],
            "finite",
        ),
        (
            np.zeros((3, 5, 5), dtype=np.float32),
            np.full(26, 1 / 25, dtype=np.float32),
            "sum to 1",
        ),
        (
            np.zeros((3, 5, 5), dtype=np.float32),
            np.asarray([-1.0, *([2.0 / 25] * 25)], dtype=np.float32),
            "nonnegative",
        ),
    ],
)
def test_training_sample_rejects_invalid_arrays(
    features: np.ndarray,
    policy: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(SelfPlayError, match=match):
        TrainingSample(features, policy, 0.0, Color.BLACK, 0, 0, 0)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("value", float("inf"), r"\[-1, 1\]"),
        ("value", 1.01, r"\[-1, 1\]"),
        ("to_play", Color.EMPTY, "to_play"),
        ("to_play", 1, "to_play"),
        ("move_number", True, "move_number"),
        ("move_number", -1, "move_number"),
        ("selected_action", 26, "selected_action"),
        ("game_index", -1, "game_index"),
        ("game_index", 1 << 64, "unsigned 64-bit"),
    ],
)
def test_training_sample_rejects_invalid_values_and_metadata(
    field: str,
    value: object,
    match: str,
) -> None:
    features, policy = _arrays()
    arguments: dict[str, object] = {
        "features": features,
        "policy": policy,
        "value": 0.0,
        "to_play": Color.BLACK,
        "move_number": 0,
        "selected_action": 25,
        "game_index": 0,
    }
    arguments[field] = value
    with pytest.raises(SelfPlayError, match=match):
        TrainingSample(**arguments)  # type: ignore[arg-type]


def test_self_play_game_normalizes_tuples_and_validates_alignment() -> None:
    score = Score(0, 0, 0, 0, 25, 5.5)
    first = _sample()
    second = _sample(value=1.0, to_play=Color.WHITE, move_number=1)

    game = SelfPlayGame([first, second], [25, 25], score, Color.WHITE, 3)  # type: ignore[arg-type]

    assert game.samples == (first, second)
    assert game.actions == (25, 25)
    with pytest.raises(SelfPlayError, match="matching lengths"):
        SelfPlayGame((first,), (25, 25), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="selected_action"):
        SelfPlayGame((first, second), (0, 25), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="game index"):
        SelfPlayGame((_sample(game_index=4),), (25,), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="move numbers"):
        SelfPlayGame((_sample(move_number=1),), (25,), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="alternate"):
        SelfPlayGame((_sample(to_play=Color.WHITE),), (25,), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="values"):
        SelfPlayGame((_sample(value=1.0),), (25,), score, Color.WHITE, 3)
    with pytest.raises(SelfPlayError, match="final_score"):
        SelfPlayGame((first,), (25,), score, Color.BLACK, 3)


def _assert_same_games(
    actual: tuple[SelfPlayGame, ...],
    expected: tuple[SelfPlayGame, ...],
) -> None:
    assert [game.game_index for game in actual] == [game.game_index for game in expected]
    for actual_game, expected_game in zip(actual, expected, strict=True):
        assert actual_game.actions == expected_game.actions
        assert actual_game.final_score == expected_game.final_score
        for actual_sample, expected_sample in zip(
            actual_game.samples,
            expected_game.samples,
            strict=True,
        ):
            np.testing.assert_array_equal(actual_sample.features, expected_sample.features)
            np.testing.assert_array_equal(actual_sample.policy, expected_sample.policy)
            assert actual_sample.value == expected_sample.value


def test_parallel_runner_direct_mode_matches_existing_sequential_order_and_metrics(
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, games=3, workers=1)
    expected_runner = SelfPlayRunner(PassEvaluator(), config)
    expected = tuple(expected_runner.play_game(index) for index in range(7, 10))

    result = ParallelSelfPlayRunner(PassEvaluator(), config).play_games(7)

    _assert_same_games(result.games, expected)
    assert result.effective_workers == 1
    assert result.inference_mode == "direct"
    assert result.inference_metrics.requests > 0
    assert result.inference_metrics.positions == result.inference_metrics.requests
    assert result.inference_metrics.batches == result.inference_metrics.requests
    assert result.inference_metrics.max_batch_size == 1
    assert result.inference_metrics.mean_batch_size == 1.0


def test_fixed_stride_worker_assignments_cover_range_exactly_once() -> None:
    assignments = tuple(
        tuple(
            _assigned_game_indices(
                11,
                10,
                worker_id=worker_id,
                workers=4,
            )
        )
        for worker_id in range(4)
    )

    assert assignments == (
        (11, 15, 19),
        (12, 16, 20),
        (13, 17),
        (14, 18),
    )
    assert sorted(index for assignment in assignments for index in assignment) == list(
        range(11, 21)
    )


def test_parallel_mode_sorts_games_matches_sequential_and_batches_workers() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, games=4, workers=4, max_batch_size=16)
    expected = tuple(
        SelfPlayRunner(PassEvaluator(), config).play_game(index) for index in range(3, 7)
    )

    result = ParallelSelfPlayRunner(PassEvaluator(), config).play_games(3)

    _assert_same_games(result.games, expected)
    assert tuple(game.game_index for game in result.games) == (3, 4, 5, 6)
    assert result.effective_workers == 4
    assert result.inference_mode == "deterministic_batch"
    assert result.inference_metrics.requests > result.inference_metrics.batches
    assert result.inference_metrics.positions == result.inference_metrics.requests
    assert result.inference_metrics.max_batch_size == 4
    assert result.inference_metrics.mean_batch_size == 4.0


def test_parallel_mode_is_repeatable_despite_delayed_game_worker_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(
        root,
        games=4,
        workers=4,
        root_noise=True,
        temperature_moves=8,
    )
    original = SelfPlayRunner.play_game

    def delayed_play_game(self: SelfPlayRunner, game_index: int) -> SelfPlayGame:
        sleep((game_index % 4) * 0.002)
        return original(self, game_index)

    monkeypatch.setattr(SelfPlayRunner, "play_game", delayed_play_game)

    first = ParallelSelfPlayRunner(PassEvaluator(), config).play_games(20)
    second = ParallelSelfPlayRunner(PassEvaluator(), config).play_games(20)

    _assert_same_games(first.games, second.games)
    assert first.inference_metrics == second.inference_metrics


@pytest.mark.parametrize("workers", [True, 0, -1, 4, "2"])
def test_parallel_runner_rejects_invalid_worker_overrides(workers: object) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, games=3, workers=1)

    with pytest.raises(ParallelSelfPlayError, match="workers"):
        ParallelSelfPlayRunner(PassEvaluator(), config, workers=workers)  # type: ignore[arg-type]


def test_parallel_runner_uses_configured_workers_and_validates_constructor_types() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, games=3, workers=2)

    runner = ParallelSelfPlayRunner(PassEvaluator(), config)

    assert runner.effective_workers == 2
    with pytest.raises(TypeError, match="evaluator"):
        ParallelSelfPlayRunner(object(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AppConfig"):
        ParallelSelfPlayRunner(PassEvaluator(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "first_game_index",
    [-1, True, 1 << 64],
)
def test_parallel_runner_rejects_invalid_first_game_index(
    first_game_index: object,
) -> None:
    root = Path(__file__).resolve().parents[1]
    runner = ParallelSelfPlayRunner(
        PassEvaluator(),
        _config(root, games=2, workers=1),
    )

    with pytest.raises(ParallelSelfPlayError, match="first_game_index"):
        runner.play_games(first_game_index)  # type: ignore[arg-type]


def test_parallel_runner_rejects_final_game_index_overflow_before_evaluation() -> None:
    root = Path(__file__).resolve().parents[1]

    class UnexpectedEvaluator:
        def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
            del states
            pytest.fail("overflow must be rejected before evaluation")

    runner = ParallelSelfPlayRunner(
        UnexpectedEvaluator(),
        _config(root, games=2, workers=1),
    )

    with pytest.raises(ParallelSelfPlayError, match="exceeds"):
        runner.play_games((1 << 64) - 1)


def test_evaluator_failure_aborts_parallel_batch_and_releases_all_threads() -> None:
    root = Path(__file__).resolve().parents[1]
    failure = RuntimeError("model unavailable")

    class FailingEvaluator:
        def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
            del states
            raise failure

    runner = ParallelSelfPlayRunner(
        FailingEvaluator(),
        _config(root, games=4, workers=4),
    )

    with pytest.raises(ParallelSelfPlayError, match="parallel") as captured:
        runner.play_games(0)

    assert captured.value.__cause__ is failure
    assert not any(
        thread.name.startswith(("azgo-self-play", "azgo-deterministic-inference"))
        for thread in enumerate_threads()
    )


def test_partial_executor_submission_aborts_before_shutdown_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    request_started = Event()
    submission_failure = RuntimeError("worker submission failed")
    watchdog_failure = RuntimeError("executor exit reached before coordinator abort")
    coordinators: list[TrackingCoordinator] = []
    abort_missing_at_exit: list[bool] = []
    original_client_evaluate = InferenceClient.evaluate_batch

    class TrackingCoordinator(DeterministicInferenceCoordinator):
        def __init__(
            self,
            evaluator: Evaluator,
            *,
            max_batch_size: int,
            client_ids: Iterable[int],
        ) -> None:
            super().__init__(
                evaluator,
                max_batch_size=max_batch_size,
                client_ids=client_ids,
            )
            self.abort_called = Event()
            coordinators.append(self)

        def abort(self, cause: BaseException) -> None:
            self.abort_called.set()
            super().abort(cause)

    class FaultInjectingExecutor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            self._executor = RealThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=thread_name_prefix,
            )
            self._submissions = 0

        def __enter__(self) -> Self:
            self._executor.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> bool | None:
            coordinator = coordinators[0]
            missing = not coordinator.abort_called.is_set()
            abort_missing_at_exit.append(missing)
            if missing:
                # Watchdog cleanup makes the regression fail without hanging pytest.
                coordinator.abort(watchdog_failure)
            return self._executor.__exit__(exc_type, exc_value, traceback)

        def submit(
            self,
            function: Callable[[int], tuple[SelfPlayGame, ...]],
            worker_id: int,
        ) -> Future[tuple[SelfPlayGame, ...]]:
            self._submissions += 1
            if self._submissions == 2:
                raise submission_failure
            future = self._executor.submit(function, worker_id)
            assert request_started.wait(timeout=2.0)
            return future

    def signaling_evaluate(
        client: InferenceClient,
        states: Sequence[GameState],
    ) -> EvaluationBatch:
        request_started.set()
        return original_client_evaluate(client, states)

    monkeypatch.setattr(
        "azgo.self_play.DeterministicInferenceCoordinator",
        TrackingCoordinator,
    )
    monkeypatch.setattr("azgo.self_play.ThreadPoolExecutor", FaultInjectingExecutor)
    monkeypatch.setattr(InferenceClient, "evaluate_batch", signaling_evaluate)
    runner = ParallelSelfPlayRunner(
        PassEvaluator(),
        _config(root, games=4, workers=4),
    )

    with pytest.raises(ParallelSelfPlayError, match="worker submission failed") as captured:
        runner.play_games(0)

    assert captured.value.__cause__ is submission_failure
    assert abort_missing_at_exit == [False]
    assert len(coordinators) == 1
    for client_id in range(4):
        with pytest.raises(InferenceClosedError):
            coordinators[0].client(client_id)
    assert not any(
        thread.name.startswith(("azgo-self-play", "azgo-deterministic-inference"))
        for thread in enumerate_threads()
    )


def test_direct_evaluator_failure_preserves_actionable_root_cause() -> None:
    root = Path(__file__).resolve().parents[1]
    failure = RuntimeError("direct model unavailable")

    class FailingEvaluator:
        def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
            del states
            raise failure

    runner = ParallelSelfPlayRunner(
        FailingEvaluator(),
        _config(root, games=1, workers=1),
    )

    with pytest.raises(ParallelSelfPlayError, match="direct model unavailable") as captured:
        runner.play_games(0)

    assert captured.value.__cause__ is failure


def test_game_failure_aborts_all_workers_and_never_returns_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _config(root, games=4, workers=4)
    failure = SelfPlayLimitError("forced game failure")
    original = SelfPlayRunner.play_game
    completed: list[int] = []

    def failing_play_game(self: SelfPlayRunner, game_index: int) -> SelfPlayGame:
        if game_index == 2:
            raise failure
        game = original(self, game_index)
        completed.append(game_index)
        return game

    monkeypatch.setattr(SelfPlayRunner, "play_game", failing_play_game)

    with pytest.raises(ParallelSelfPlayError, match="parallel") as captured:
        ParallelSelfPlayRunner(PassEvaluator(), config).play_games(0)

    assert captured.value.__cause__ is failure
    assert len(completed) < config.self_play.games
    assert not any(
        thread.name.startswith(("azgo-self-play", "azgo-deterministic-inference"))
        for thread in enumerate_threads()
    )


def test_parallel_result_is_frozen_and_rejects_unsorted_or_invalid_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    games = tuple(
        SelfPlayRunner(PassEvaluator(), _config(root)).play_game(index)
        for index in (1, 2)
    )
    metrics = InferenceMetrics(2, 2, 1, 2, 2.0)
    result = ParallelSelfPlayResult(games, 2, "deterministic_batch", metrics)

    assert result.games == games
    with pytest.raises(FrozenInstanceError):
        result.effective_workers = 1  # type: ignore[misc]
    with pytest.raises(ParallelSelfPlayError, match="contiguous"):
        ParallelSelfPlayResult(tuple(reversed(games)), 2, "deterministic_batch", metrics)
    with pytest.raises(ParallelSelfPlayError, match="effective_workers"):
        ParallelSelfPlayResult(games, 0, "deterministic_batch", metrics)
    with pytest.raises(ParallelSelfPlayError, match="number of games"):
        ParallelSelfPlayResult(games, 3, "deterministic_batch", metrics)
    with pytest.raises(ParallelSelfPlayError, match="inference_mode"):
        ParallelSelfPlayResult(games, 1, "deterministic_batch", metrics)
    with pytest.raises(ParallelSelfPlayError, match="inference_mode"):
        ParallelSelfPlayResult(games, 2, "direct", metrics)
    with pytest.raises(ParallelSelfPlayError, match="inference_mode"):
        ParallelSelfPlayResult(games, 2, "timed", metrics)  # type: ignore[arg-type]

    noncontiguous = (games[0], SelfPlayRunner(PassEvaluator(), _config(root)).play_game(3))
    with pytest.raises(ParallelSelfPlayError, match="contiguous"):
        ParallelSelfPlayResult(noncontiguous, 2, "deterministic_batch", metrics)
