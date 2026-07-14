"""Deterministic AlphaZero self-play game generation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import TYPE_CHECKING, Literal

import numpy as np

from azgo.config import AppConfig
from azgo.encoding import encode_state
from azgo.evaluator import Evaluator
from azgo.game import Color, GameState, Rules, Ruleset, Score
from azgo.inference import (
    CountingEvaluator,
    DeterministicInferenceCoordinator,
    InferenceMetrics,
)
from azgo.search import MCTS, SearchResult

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


_MAX_UINT64 = (1 << 64) - 1


class SelfPlayError(ValueError):
    """Raised when self-play data or an operation violates its contract."""


class SelfPlayLimitError(SelfPlayError):
    """Raised when a game reaches its safety limit before normal termination."""


class ParallelSelfPlayError(SelfPlayError):
    """Raised when a complete multi-game self-play batch cannot be produced."""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One immutable canonical policy-value training example."""

    features: NDArray[np.float32]
    policy: NDArray[np.float32]
    value: float
    to_play: Color
    move_number: int
    selected_action: int
    game_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.features, np.ndarray):
            raise SelfPlayError("features must be a NumPy array")
        if not isinstance(self.policy, np.ndarray):
            raise SelfPlayError("policy must be a NumPy array")

        with np.errstate(over="ignore", invalid="ignore"):
            features = np.array(self.features, dtype=np.float32, order="C", copy=True)
            policy = np.array(self.policy, dtype=np.float32, order="C", copy=True)
        if features.ndim != 3:
            raise SelfPlayError("features must have shape [channels, board_size, board_size]")
        channels, height, width = features.shape
        if channels <= 0 or height <= 0 or height != width:
            raise SelfPlayError(
                "features must have positive channels and square spatial dimensions"
            )
        if policy.ndim != 1 or policy.shape != (height * width + 1,):
            raise SelfPlayError(
                f"policy must have shape ({height * width + 1},) for the feature board size"
            )
        if not np.isfinite(features).all():
            raise SelfPlayError("features must contain only finite values")
        if not np.isfinite(policy).all():
            raise SelfPlayError("policy must contain only finite values")
        if np.any(policy < 0.0):
            raise SelfPlayError("policy probabilities must be nonnegative")
        policy_sum = float(np.sum(policy, dtype=np.float64))
        if not np.isclose(policy_sum, 1.0, rtol=1e-5, atol=1e-6):
            raise SelfPlayError("policy probabilities must sum to 1")

        if isinstance(self.value, bool) or not isinstance(
            self.value,
            (int, float, np.integer, np.floating),
        ):
            raise SelfPlayError("value must be a finite number in [-1, 1]")
        value = float(self.value)
        if not isfinite(value) or not -1.0 <= value <= 1.0:
            raise SelfPlayError("value must be a finite number in [-1, 1]")
        if not isinstance(self.to_play, Color) or self.to_play is Color.EMPTY:
            raise SelfPlayError("to_play must be Color.BLACK or Color.WHITE")
        move_number = _nonnegative_integer(self.move_number, "move_number")
        selected_action = _nonnegative_integer(self.selected_action, "selected_action")
        if selected_action >= policy.size:
            raise SelfPlayError(f"selected_action must be in [0, {policy.size - 1}]")
        game_index = _unsigned_64_bit_integer(self.game_index, "game_index")

        features.setflags(write=False)
        policy.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "move_number", move_number)
        object.__setattr__(self, "selected_action", selected_action)
        object.__setattr__(self, "game_index", game_index)


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    """A complete normally terminated self-play game and its labeled samples."""

    samples: tuple[TrainingSample, ...]
    actions: tuple[int, ...]
    final_score: Score
    winner: Color | None
    game_index: int

    def __post_init__(self) -> None:
        try:
            samples = tuple(self.samples)
            actions = tuple(self.actions)
        except TypeError as exc:
            raise SelfPlayError("samples and actions must be iterable") from exc
        game_index = _unsigned_64_bit_integer(self.game_index, "game_index")
        if not samples:
            raise SelfPlayError("a self-play game must contain at least one sample")
        if len(samples) != len(actions):
            raise SelfPlayError("samples and actions must have matching lengths")
        if not isinstance(self.final_score, Score):
            raise SelfPlayError("final_score must be a Score")
        if self.winner is not None and (
            not isinstance(self.winner, Color) or self.winner is Color.EMPTY
        ):
            raise SelfPlayError("winner must be Color.BLACK, Color.WHITE, or None")
        if self.final_score.winner is not self.winner:
            raise SelfPlayError("winner must match final_score.winner")

        for index, (sample, action) in enumerate(zip(samples, actions, strict=True)):
            if not isinstance(sample, TrainingSample):
                raise SelfPlayError("samples must contain only TrainingSample objects")
            normalized_action = _nonnegative_integer(action, f"actions[{index}]")
            if normalized_action != sample.selected_action:
                raise SelfPlayError("each action must match its sample selected_action")
            if sample.game_index != game_index:
                raise SelfPlayError("every sample must match the game index")
            if sample.move_number != index:
                raise SelfPlayError("sample move numbers must be contiguous and start at zero")
            expected_player = Color.BLACK if index % 2 == 0 else Color.WHITE
            if sample.to_play is not expected_player:
                raise SelfPlayError("sample players must alternate starting with Black")
            expected_value = float(self.final_score.outcome(sample.to_play))
            if sample.value != expected_value:
                raise SelfPlayError("sample values must match the final score outcome")

        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "game_index", game_index)


@dataclass(frozen=True, slots=True)
class ParallelSelfPlayResult:
    """One complete, ordered multi-game self-play run and its inference metrics."""

    games: tuple[SelfPlayGame, ...]
    effective_workers: int
    inference_mode: Literal["direct", "deterministic_batch"]
    inference_metrics: InferenceMetrics

    def __post_init__(self) -> None:
        try:
            games = tuple(self.games)
        except TypeError as exc:
            raise ParallelSelfPlayError("games must be iterable") from exc
        if not games:
            raise ParallelSelfPlayError("games must contain at least one completed game")
        if any(not isinstance(game, SelfPlayGame) for game in games):
            raise ParallelSelfPlayError("games must contain only SelfPlayGame objects")
        game_indices = tuple(game.game_index for game in games)
        expected_indices = tuple(range(game_indices[0], game_indices[0] + len(games)))
        if game_indices != expected_indices:
            raise ParallelSelfPlayError(
                "games must have contiguous ascending game indices"
            )
        effective_workers = _positive_integer(
            self.effective_workers,
            "effective_workers",
            ParallelSelfPlayError,
        )
        if effective_workers > len(games):
            raise ParallelSelfPlayError(
                "effective_workers must be no greater than the number of games"
            )
        if self.inference_mode not in {"direct", "deterministic_batch"}:
            raise ParallelSelfPlayError(
                "inference_mode must be 'direct' or 'deterministic_batch'"
            )
        expected_mode = "direct" if effective_workers == 1 else "deterministic_batch"
        if self.inference_mode != expected_mode:
            raise ParallelSelfPlayError(
                f"inference_mode must be '{expected_mode}' for {effective_workers} worker(s)"
            )
        if not isinstance(self.inference_metrics, InferenceMetrics):
            raise ParallelSelfPlayError("inference_metrics must be InferenceMetrics")

        object.__setattr__(self, "games", games)
        object.__setattr__(self, "effective_workers", effective_workers)


@dataclass(frozen=True, slots=True)
class _PendingSample:
    features: NDArray[np.float32]
    policy: NDArray[np.float32]
    to_play: Color
    move_number: int
    selected_action: int


class SelfPlayRunner:
    """Generate reproducible self-play games with reusable PUCT trees."""

    def __init__(self, evaluator: Evaluator, config: AppConfig) -> None:
        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must implement Evaluator")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self._evaluator = evaluator
        self._config = config

    def play_game(self, game_index: int) -> SelfPlayGame:
        """Play and label one game identified by a deterministic nonnegative index."""

        normalized_index = _unsigned_64_bit_integer(game_index, "game_index")
        mcts_seed, move_rng = _random_streams(self._config, normalized_index)
        rules = Rules(
            board_size=self._config.game.board_size,
            komi=self._config.game.komi,
            ruleset=Ruleset(self._config.game.rules.ruleset),
        )
        state = GameState.new(rules, zobrist_seed=self._config.zobrist.seed)
        search = MCTS(
            self._evaluator,
            simulations=self._config.search.simulations,
            c_puct=self._config.search.c_puct,
            seed=mcts_seed,
            dirichlet_alpha=self._config.search.dirichlet_alpha,
            dirichlet_fraction=self._config.search.dirichlet_fraction,
        )
        pending: list[_PendingSample] = []
        actions: list[int] = []

        while not state.is_terminal:
            if state.move_number >= self._config.self_play.max_moves:
                raise SelfPlayLimitError(
                    "self-play game "
                    f"{normalized_index} reached max_moves="
                    f"{self._config.self_play.max_moves} without terminating"
                )
            result = search.run(
                state,
                add_root_noise=self._config.self_play.root_noise,
            )
            action = self._select_action(result, state.move_number, move_rng)
            pending.append(
                _PendingSample(
                    features=encode_state(state, self._config.model.history_length),
                    policy=result.visit_policy,
                    to_play=state.to_play,
                    move_number=state.move_number,
                    selected_action=action,
                )
            )
            actions.append(action)
            state = search.advance(action)

        score = state.score()
        samples = tuple(
            TrainingSample(
                features=example.features,
                policy=example.policy,
                value=float(score.outcome(example.to_play)),
                to_play=example.to_play,
                move_number=example.move_number,
                selected_action=example.selected_action,
                game_index=normalized_index,
            )
            for example in pending
        )
        return SelfPlayGame(
            samples=samples,
            actions=tuple(actions),
            final_score=score,
            winner=score.winner,
            game_index=normalized_index,
        )

    def _select_action(
        self,
        result: SearchResult,
        move_number: int,
        rng: Generator,
    ) -> int:
        if move_number >= self._config.self_play.temperature_moves:
            return result.selected_action
        return _sample_with_temperature(
            result.visit_counts,
            self._config.self_play.temperature,
            rng,
        )


class ParallelSelfPlayRunner:
    """Generate a complete deterministic batch directly or with coordinated workers."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: AppConfig,
        workers: int | None = None,
    ) -> None:
        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must implement Evaluator")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")

        configured_workers = config.self_play.workers if workers is None else workers
        effective_workers = _positive_integer(
            configured_workers,
            "workers",
            ParallelSelfPlayError,
        )
        if effective_workers > config.self_play.games:
            raise ParallelSelfPlayError(
                "workers must be no greater than self_play.games "
                f"({config.self_play.games})"
            )

        self._evaluator = evaluator
        self._config = config
        self._workers = effective_workers

    @property
    def effective_workers(self) -> int:
        """Number of game workers used by this runner."""

        return self._workers

    def play_games(self, first_game_index: int = 0) -> ParallelSelfPlayResult:
        """Play the configured contiguous game range without returning partial work."""

        first_index = _unsigned_64_bit_integer_for_parallel(
            first_game_index,
            "first_game_index",
        )
        final_index = first_index + self._config.self_play.games - 1
        if final_index > _MAX_UINT64:
            raise ParallelSelfPlayError(
                "the configured game range exceeds the unsigned 64-bit game index limit"
            )

        if self._workers == 1:
            return self._play_direct(first_index)
        return self._play_coordinated(first_index)

    def _play_direct(self, first_game_index: int) -> ParallelSelfPlayResult:
        counted = CountingEvaluator(self._evaluator)
        runner = SelfPlayRunner(counted, self._config)
        try:
            games = tuple(
                runner.play_game(game_index)
                for game_index in _assigned_game_indices(
                    first_game_index,
                    self._config.self_play.games,
                    worker_id=0,
                    workers=1,
                )
            )
        except BaseException as exc:
            underlying = _underlying_cause(exc)
            raise ParallelSelfPlayError(
                f"self-play batch failed: {underlying}"
            ) from underlying

        ordered = _validate_complete_game_range(
            games,
            first_game_index,
            self._config.self_play.games,
        )
        return ParallelSelfPlayResult(
            games=ordered,
            effective_workers=1,
            inference_mode="direct",
            inference_metrics=counted.metrics,
        )

    def _play_coordinated(self, first_game_index: int) -> ParallelSelfPlayResult:
        worker_ids = tuple(range(self._workers))
        coordinator = DeterministicInferenceCoordinator(
            self._evaluator,
            max_batch_size=self._config.inference.max_batch_size,
            client_ids=worker_ids,
        )
        clients = tuple(coordinator.client(worker_id) for worker_id in worker_ids)
        assignments = tuple(
            tuple(
                _assigned_game_indices(
                    first_game_index,
                    self._config.self_play.games,
                    worker_id=worker_id,
                    workers=self._workers,
                )
            )
            for worker_id in worker_ids
        )
        failure_lock = Lock()
        primary_failure: list[BaseException] = []

        def record_failure(cause: BaseException) -> BaseException:
            underlying = _underlying_cause(cause)
            with failure_lock:
                if not primary_failure:
                    primary_failure.append(underlying)
                primary = primary_failure[0]
            with suppress(Exception):
                coordinator.abort(primary)
            return primary

        def run_worker(worker_id: int) -> tuple[SelfPlayGame, ...]:
            client = clients[worker_id]
            caught: BaseException | None = None
            try:
                runner = SelfPlayRunner(client, self._config)
                return tuple(
                    runner.play_game(game_index) for game_index in assignments[worker_id]
                )
            except BaseException as exc:
                caught = exc
                record_failure(exc)
                raise
            finally:
                try:
                    client.close()
                except BaseException as exc:
                    if caught is None:
                        record_failure(exc)
                        raise

        completed: list[tuple[SelfPlayGame, ...]] = []
        try:
            with coordinator, ThreadPoolExecutor(
                max_workers=self._workers,
                thread_name_prefix="azgo-self-play",
            ) as executor:
                futures: list[Future[tuple[SelfPlayGame, ...]]] = []
                try:
                    futures.extend(
                        executor.submit(run_worker, worker_id)
                        for worker_id in worker_ids
                    )
                except BaseException as exc:
                    # Abort while the coordinator is still inside its running context.
                    # Already-submitted workers may be waiting for clients whose tasks
                    # were never submitted, so executor shutdown must not begin first.
                    record_failure(exc)
                for future in futures:
                    try:
                        completed.append(future.result())
                    except BaseException as exc:
                        record_failure(exc)
        except BaseException as exc:
            record_failure(exc)

        if primary_failure:
            primary = primary_failure[0]
            raise ParallelSelfPlayError(
                f"parallel self-play batch failed: {primary}"
            ) from primary

        games = tuple(game for worker_games in completed for game in worker_games)
        ordered = _validate_complete_game_range(
            games,
            first_game_index,
            self._config.self_play.games,
        )
        return ParallelSelfPlayResult(
            games=ordered,
            effective_workers=self._workers,
            inference_mode="deterministic_batch",
            inference_metrics=coordinator.metrics,
        )


def _random_streams(config: AppConfig, game_index: int) -> tuple[int, Generator]:
    root = np.random.SeedSequence(
        [config.self_play.seed, config.search.seed, game_index]
    )
    mcts_sequence, move_sequence = root.spawn(2)
    mcts_seed = int(mcts_sequence.generate_state(1, dtype=np.uint64)[0])
    return mcts_seed, np.random.default_rng(move_sequence)


def _sample_with_temperature(
    visit_counts: NDArray[np.int64],
    temperature: float,
    rng: Generator,
) -> int:
    if not isinstance(visit_counts, np.ndarray) or visit_counts.ndim != 1:
        raise SelfPlayError("visit_counts must be a one-dimensional NumPy array")
    if not np.issubdtype(visit_counts.dtype, np.integer):
        raise SelfPlayError("visit_counts must contain integers")
    if np.any(visit_counts < 0):
        raise SelfPlayError("visit_counts must be nonnegative")
    if isinstance(temperature, bool) or not isinstance(
        temperature,
        (int, float, np.integer, np.floating),
    ):
        raise SelfPlayError("temperature must be a finite positive number")
    normalized_temperature = float(temperature)
    if not isfinite(normalized_temperature) or normalized_temperature <= 0.0:
        raise SelfPlayError("temperature must be a finite positive number")

    positive_actions = np.flatnonzero(visit_counts > 0)
    if positive_actions.size == 0:
        raise SelfPlayError("temperature sampling requires at least one visited action")
    log_counts = np.log(visit_counts[positive_actions].astype(np.float64, copy=False))
    with np.errstate(over="ignore", under="ignore"):
        scaled = (log_counts - float(np.max(log_counts))) / normalized_temperature
        weights = np.exp(scaled)
    probabilities = weights / float(np.sum(weights))
    return int(rng.choice(positive_actions, p=probabilities))


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SelfPlayError(f"{name} must be a nonnegative integer")
    return value


def _unsigned_64_bit_integer(value: int, name: str) -> int:
    normalized = _nonnegative_integer(value, name)
    if normalized > _MAX_UINT64:
        raise SelfPlayError(f"{name} must be an unsigned 64-bit integer")
    return normalized


def _unsigned_64_bit_integer_for_parallel(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParallelSelfPlayError(f"{name} must be a nonnegative integer")
    if value > _MAX_UINT64:
        raise ParallelSelfPlayError(f"{name} must be an unsigned 64-bit integer")
    return value


def _positive_integer(
    value: int,
    name: str,
    error_type: type[SelfPlayError],
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error_type(f"{name} must be a positive integer")
    return value


def _assigned_game_indices(
    first_game_index: int,
    games: int,
    *,
    worker_id: int,
    workers: int,
) -> range:
    """Return one worker's fixed-stride portion of a contiguous game range."""

    return range(first_game_index + worker_id, first_game_index + games, workers)


def _validate_complete_game_range(
    games: tuple[SelfPlayGame, ...],
    first_game_index: int,
    game_count: int,
) -> tuple[SelfPlayGame, ...]:
    if any(not isinstance(game, SelfPlayGame) for game in games):
        raise ParallelSelfPlayError(
            "self-play workers must return only completed SelfPlayGame objects"
        )
    ordered = tuple(sorted(games, key=lambda game: game.game_index))
    actual_indices = tuple(game.game_index for game in ordered)
    expected_indices = tuple(range(first_game_index, first_game_index + game_count))
    if actual_indices != expected_indices:
        raise ParallelSelfPlayError(
            "self-play workers did not return the exact requested contiguous game range"
        )
    return ordered


def _underlying_cause(cause: BaseException) -> BaseException:
    """Return the deepest explicit exception cause without following cycles."""

    current = cause
    seen: set[int] = set()
    while current.__cause__ is not None and id(current) not in seen:
        seen.add(id(current))
        current = current.__cause__
    return current


__all__ = [
    "ParallelSelfPlayError",
    "ParallelSelfPlayResult",
    "ParallelSelfPlayRunner",
    "SelfPlayError",
    "SelfPlayGame",
    "SelfPlayLimitError",
    "SelfPlayRunner",
    "TrainingSample",
]
