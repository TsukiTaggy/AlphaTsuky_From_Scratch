"""Deterministic paired arena evaluation for policy-value agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

import numpy as np

from azgo.config import AppConfig
from azgo.evaluator import Evaluator
from azgo.game import SUPPORTED_BOARD_SIZES, Color, GameState, Rules, Ruleset, Score
from azgo.search import MCTS

type CandidateOutcome = Literal["win", "draw", "loss"]


class ArenaError(ValueError):
    """Raised when an arena input, result, or operation violates its contract."""


class ArenaOpeningError(ArenaError):
    """Raised when a deterministic paired opening cannot be generated."""


class ArenaGameError(ArenaError):
    """Raised when an arena game cannot be completed safely."""


class ArenaLimitError(ArenaGameError):
    """Raised when an arena game reaches its move limit before termination."""


@dataclass(frozen=True, slots=True)
class ArenaGameResult:
    """One complete game from a color-balanced arena pair."""

    pair_index: int
    game_index: int
    candidate_color: Color
    opening_actions: tuple[int, ...]
    actions: tuple[int, ...]
    move_count: int
    final_score: Score
    winner: Color | None
    candidate_outcome: CandidateOutcome

    def __post_init__(self) -> None:
        pair_index = _nonnegative_integer(self.pair_index, "pair_index")
        game_index = _nonnegative_integer(self.game_index, "game_index")
        if not isinstance(self.candidate_color, Color) or self.candidate_color is Color.EMPTY:
            raise ArenaError("candidate_color must be Color.BLACK or Color.WHITE")

        expected_game_index = pair_index * 2 + (
            0 if self.candidate_color is Color.BLACK else 1
        )
        if game_index != expected_game_index:
            raise ArenaError(
                "game_index must identify the candidate's color-swapped game within its pair"
            )

        try:
            opening_actions = tuple(self.opening_actions)
        except TypeError as exc:
            raise ArenaError("opening_actions must be iterable") from exc
        try:
            actions = tuple(self.actions)
        except TypeError as exc:
            raise ArenaError("actions must be iterable") from exc

        point_count = _validate_score(self.final_score)
        for index, action in enumerate(opening_actions):
            normalized_action = _nonnegative_integer(action, f"opening_actions[{index}]")
            if normalized_action >= point_count:
                raise ArenaError("opening_actions must contain only non-pass board actions")

        for index, action in enumerate(actions):
            normalized_action = _nonnegative_integer(action, f"actions[{index}]")
            if normalized_action > point_count:
                raise ArenaError(f"actions must be in [0, {point_count}]")

        move_count = _nonnegative_integer(self.move_count, "move_count")
        if move_count != len(actions):
            raise ArenaError("move_count must equal the complete action count")
        if actions[: len(opening_actions)] != opening_actions:
            raise ArenaError("opening_actions must be the prefix of actions")
        if len(actions) < len(opening_actions) + 2:
            raise ArenaError(
                "move_count must include the opening and the two passes needed to terminate"
            )
        if actions[-2:] != (point_count, point_count):
            raise ArenaError("actions must terminate with two consecutive passes")

        if self.winner is not None and (
            not isinstance(self.winner, Color) or self.winner is Color.EMPTY
        ):
            raise ArenaError("winner must be Color.BLACK, Color.WHITE, or None")
        if self.winner is not self.final_score.winner:
            raise ArenaError("winner must match final_score.winner")

        expected_outcome = _candidate_outcome(self.winner, self.candidate_color)
        candidate_outcome = _normalize_candidate_outcome(self.candidate_outcome)
        if candidate_outcome != expected_outcome:
            raise ArenaError("candidate_outcome must match the final score and candidate color")

        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "game_index", game_index)
        object.__setattr__(self, "opening_actions", opening_actions)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "move_count", move_count)
        object.__setattr__(self, "candidate_outcome", candidate_outcome)


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """Immutable aggregate for a complete paired arena run."""

    games: tuple[ArenaGameResult, ...]
    promotion_threshold: float
    candidate_wins: int = field(init=False)
    incumbent_wins: int = field(init=False)
    draws: int = field(init=False)
    candidate_points: float = field(init=False)
    candidate_score: float = field(init=False)
    promotion_eligible: bool = field(init=False)

    def __post_init__(self) -> None:
        try:
            games = tuple(self.games)
        except TypeError as exc:
            raise ArenaError("games must be iterable") from exc
        if not games or len(games) % 2 != 0:
            raise ArenaError("games must contain a positive even number of results")

        for index, game in enumerate(games):
            if not isinstance(game, ArenaGameResult):
                raise ArenaError("games must contain only ArenaGameResult objects")
            if game.game_index != index or game.pair_index != index // 2:
                raise ArenaError("games must be contiguous and ordered by pair and game index")
            expected_color = Color.BLACK if index % 2 == 0 else Color.WHITE
            if game.candidate_color is not expected_color:
                raise ArenaError("each pair must swap the candidate between Black and White")
            if index % 2 == 1 and game.opening_actions != games[index - 1].opening_actions:
                raise ArenaError("both games in a pair must use the same opening actions")

        threshold = _promotion_threshold(self.promotion_threshold)
        candidate_wins = sum(game.candidate_outcome == "win" for game in games)
        draws = sum(game.candidate_outcome == "draw" for game in games)
        incumbent_wins = len(games) - candidate_wins - draws
        candidate_points = float(candidate_wins) + 0.5 * float(draws)
        candidate_score = candidate_points / float(len(games))

        object.__setattr__(self, "games", games)
        object.__setattr__(self, "promotion_threshold", threshold)
        object.__setattr__(self, "candidate_wins", candidate_wins)
        object.__setattr__(self, "incumbent_wins", incumbent_wins)
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "candidate_points", candidate_points)
        object.__setattr__(self, "candidate_score", candidate_score)
        object.__setattr__(self, "promotion_eligible", candidate_score >= threshold)


class ArenaRunner:
    """Evaluate candidate and incumbent agents in deterministic paired games."""

    def __init__(
        self,
        candidate_evaluator: Evaluator,
        incumbent_evaluator: Evaluator,
        config: AppConfig,
    ) -> None:
        if not isinstance(candidate_evaluator, Evaluator):
            raise TypeError("candidate_evaluator must implement Evaluator")
        if not isinstance(incumbent_evaluator, Evaluator):
            raise TypeError("incumbent_evaluator must implement Evaluator")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        self._candidate_evaluator = candidate_evaluator
        self._incumbent_evaluator = incumbent_evaluator
        self._config = config

    def run(self) -> ArenaResult:
        """Play every configured pair or raise without returning partial results."""

        candidate_search, incumbent_search = self._build_searches()
        games: list[ArenaGameResult] = []
        for pair_index in range(self._config.arena.games // 2):
            opening_state, opening_actions = self._generate_opening(pair_index)
            games.append(
                self._play_game(
                    opening_state,
                    opening_actions,
                    pair_index=pair_index,
                    game_index=pair_index * 2,
                    candidate_color=Color.BLACK,
                    candidate_search=candidate_search,
                    incumbent_search=incumbent_search,
                )
            )
            games.append(
                self._play_game(
                    opening_state,
                    opening_actions,
                    pair_index=pair_index,
                    game_index=pair_index * 2 + 1,
                    candidate_color=Color.WHITE,
                    candidate_search=candidate_search,
                    incumbent_search=incumbent_search,
                )
            )

        return ArenaResult(tuple(games), self._config.arena.promotion_threshold)

    def _build_searches(self) -> tuple[MCTS, MCTS]:
        seed_sequence = np.random.SeedSequence(
            [self._config.arena.seed, self._config.search.seed]
        )
        candidate_seed_sequence, incumbent_seed_sequence = seed_sequence.spawn(2)
        candidate_seed = int(
            candidate_seed_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        incumbent_seed = int(
            incumbent_seed_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        try:
            return (
                MCTS(
                    self._candidate_evaluator,
                    simulations=self._config.search.simulations,
                    c_puct=self._config.search.c_puct,
                    seed=candidate_seed,
                    dirichlet_alpha=self._config.search.dirichlet_alpha,
                    dirichlet_fraction=self._config.search.dirichlet_fraction,
                ),
                MCTS(
                    self._incumbent_evaluator,
                    simulations=self._config.search.simulations,
                    c_puct=self._config.search.c_puct,
                    seed=incumbent_seed,
                    dirichlet_alpha=self._config.search.dirichlet_alpha,
                    dirichlet_fraction=self._config.search.dirichlet_fraction,
                ),
            )
        except Exception as exc:
            raise ArenaError("arena search initialization failed") from exc

    def _generate_opening(self, pair_index: int) -> tuple[GameState, tuple[int, ...]]:
        try:
            normalized_pair_index = _nonnegative_integer(pair_index, "pair_index")
            rules = Rules(
                board_size=self._config.game.board_size,
                komi=self._config.game.komi,
                ruleset=Ruleset(self._config.game.rules.ruleset),
            )
            state = GameState.new(rules, zobrist_seed=self._config.zobrist.seed)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [self._config.arena.seed, normalized_pair_index]
                )
            )
            actions: list[int] = []
            for _ in range(self._config.arena.opening_moves):
                legal_non_pass = tuple(
                    action for action in state.legal_actions() if action != state.pass_action
                )
                if not legal_non_pass:
                    raise ArenaOpeningError(
                        "arena pair "
                        f"{normalized_pair_index} has no legal non-pass opening action"
                    )
                action = int(rng.choice(np.asarray(legal_non_pass, dtype=np.int64)))
                state = state.apply(action)
                actions.append(action)
        except ArenaOpeningError:
            raise
        except Exception as exc:
            raise ArenaOpeningError(
                f"arena pair {pair_index} opening generation failed"
            ) from exc

        return state, tuple(actions)

    def _play_game(
        self,
        opening_state: GameState,
        opening_actions: tuple[int, ...],
        *,
        pair_index: int,
        game_index: int,
        candidate_color: Color,
        candidate_search: MCTS,
        incumbent_search: MCTS,
    ) -> ArenaGameResult:
        try:
            candidate_search.reset(opening_state)
            incumbent_search.reset(opening_state)
            state = opening_state
            actions = list(opening_actions)
            while not state.is_terminal:
                if state.move_number >= self._config.arena.max_moves:
                    raise ArenaLimitError(
                        f"arena game {game_index} reached max_moves="
                        f"{self._config.arena.max_moves} without terminating"
                    )
                active_search = (
                    candidate_search if state.to_play is candidate_color else incumbent_search
                )
                result = active_search.run(state, add_root_noise=False)
                action = result.selected_action
                candidate_state = candidate_search.advance(action)
                incumbent_state = incumbent_search.advance(action)
                if candidate_state != incumbent_state:
                    raise ArenaGameError("candidate and incumbent search trees diverged")
                state = candidate_state
                actions.append(action)
        except ArenaLimitError:
            raise
        except ArenaGameError:
            raise
        except Exception as exc:
            raise ArenaGameError(f"arena game {game_index} failed") from exc

        final_score = state.score()
        winner = final_score.winner
        return ArenaGameResult(
            pair_index=pair_index,
            game_index=game_index,
            candidate_color=candidate_color,
            opening_actions=opening_actions,
            actions=tuple(actions),
            move_count=state.move_number,
            final_score=final_score,
            winner=winner,
            candidate_outcome=_candidate_outcome(winner, candidate_color),
        )


def _candidate_outcome(winner: Color | None, candidate_color: Color) -> CandidateOutcome:
    if winner is None:
        return "draw"
    return "win" if winner is candidate_color else "loss"


def _normalize_candidate_outcome(value: object) -> CandidateOutcome:
    if type(value) is not str:
        raise ArenaError("candidate_outcome must be 'win', 'draw', or 'loss'")
    if value == "win":
        return "win"
    if value == "draw":
        return "draw"
    if value == "loss":
        return "loss"
    raise ArenaError("candidate_outcome must be 'win', 'draw', or 'loss'")


def _validate_score(score: Score) -> int:
    if not isinstance(score, Score):
        raise ArenaError("final_score must be a Score")
    count_names = (
        "black_stones",
        "white_stones",
        "black_territory",
        "white_territory",
        "neutral_points",
    )
    point_count = 0
    for name in count_names:
        count = getattr(score, name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ArenaError(f"final_score.{name} must be a nonnegative integer")
        point_count += count
    if point_count not in {size * size for size in SUPPORTED_BOARD_SIZES}:
        raise ArenaError("final_score counts must partition a supported board")
    if type(score.komi) is not float:
        raise ArenaError("final_score.komi must be finite")
    if not isfinite(score.komi):
        raise ArenaError("final_score.komi must be finite")
    return point_count


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArenaError(f"{name} must be a nonnegative integer")
    return value


def _promotion_threshold(value: float) -> float:
    if type(value) is not float or not isfinite(value) or not 0.5 < value <= 1.0:
        raise ArenaError("promotion_threshold must be a finite float in (0.5, 1.0]")
    return value


__all__ = [
    "ArenaError",
    "ArenaGameError",
    "ArenaGameResult",
    "ArenaLimitError",
    "ArenaOpeningError",
    "ArenaResult",
    "ArenaRunner",
    "CandidateOutcome",
]
