"""Deterministic AlphaZero self-play game generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from azgo.config import AppConfig
from azgo.encoding import encode_state
from azgo.evaluator import Evaluator
from azgo.game import Color, GameState, Rules, Ruleset, Score
from azgo.search import MCTS, SearchResult

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray


_MAX_UINT64 = (1 << 64) - 1


class SelfPlayError(ValueError):
    """Raised when self-play data or an operation violates its contract."""


class SelfPlayLimitError(SelfPlayError):
    """Raised when a game reaches its safety limit before normal termination."""


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


__all__ = [
    "SelfPlayError",
    "SelfPlayGame",
    "SelfPlayLimitError",
    "SelfPlayRunner",
    "TrainingSample",
]
