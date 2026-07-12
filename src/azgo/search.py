"""Deterministic, reusable PUCT Monte Carlo tree search."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import TYPE_CHECKING

import numpy as np

from azgo.evaluator import EvaluationBatch, EvaluationError, Evaluator
from azgo.game import GameState

if TYPE_CHECKING:
    from numpy.typing import NDArray


_MAX_UINT64 = (1 << 64) - 1


class SearchError(ValueError):
    """Raised when a search operation cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Immutable root statistics returned by :meth:`MCTS.run`."""

    selected_action: int
    visit_counts: NDArray[np.int64]
    visit_policy: NDArray[np.float32]
    root_value: float
    simulations: int

    def __post_init__(self) -> None:
        counts = np.array(self.visit_counts, dtype=np.int64, order="C", copy=True)
        policy = np.array(self.visit_policy, dtype=np.float32, order="C", copy=True)
        if counts.ndim != 1 or policy.ndim != 1 or counts.shape != policy.shape:
            raise ValueError("visit arrays must be one-dimensional and have matching shapes")
        counts.setflags(write=False)
        policy.setflags(write=False)
        object.__setattr__(self, "visit_counts", counts)
        object.__setattr__(self, "visit_policy", policy)


@dataclass(slots=True)
class _Node:
    state: GameState
    children: dict[int, _Node] = field(default_factory=dict)
    base_priors: NDArray[np.float64] | None = None
    priors: NDArray[np.float64] | None = None
    visit_counts: NDArray[np.int64] = field(init=False)
    value_sums: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.visit_counts = np.zeros(self.state.action_size, dtype=np.int64)
        self.value_sums = np.zeros(self.state.action_size, dtype=np.float64)

    @property
    def is_expanded(self) -> bool:
        return self.base_priors is not None

    def expand(self, priors: NDArray[np.float64]) -> None:
        if self.is_expanded:
            raise RuntimeError("a search node cannot be expanded twice")
        base_priors = np.array(priors, dtype=np.float64, order="C", copy=True)
        base_priors.setflags(write=False)
        self.base_priors = base_priors
        self.priors = base_priors.copy()

    def restore_base_priors(self) -> None:
        if self.base_priors is None:
            raise RuntimeError("cannot restore priors on an unexpanded node")
        self.priors = self.base_priors.copy()


class MCTS:
    """Single-threaded PUCT search with deterministic tie breaking.

    The tree stores complete :class:`~azgo.game.GameState` objects and therefore
    deliberately does not merge transpositions.  This preserves positional
    superko history even when two nodes have the same visible board.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        simulations: int,
        c_puct: float,
        seed: int,
        dirichlet_alpha: float,
        dirichlet_fraction: float,
    ) -> None:
        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must implement Evaluator")
        self._simulations = _positive_integer(simulations, "simulations")
        self._c_puct = _positive_finite_number(c_puct, "c_puct")
        self._seed = _unsigned_64_bit_integer(seed, "seed")
        self._dirichlet_alpha = _positive_finite_number(
            dirichlet_alpha,
            "dirichlet_alpha",
        )
        self._dirichlet_fraction = _unit_interval_number(
            dirichlet_fraction,
            "dirichlet_fraction",
        )
        self._evaluator = evaluator
        self._rng = np.random.default_rng(self._seed)
        self._root: _Node | None = None

    @property
    def root_state(self) -> GameState | None:
        """Return the current immutable root state, or ``None`` before setup."""

        return None if self._root is None else self._root.state

    def reset(self, state: GameState) -> None:
        """Discard the current tree and install ``state`` as an unexpanded root."""

        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        self._root = _Node(state)

    def advance(self, action: int) -> GameState:
        """Advance the root through a legal action, reusing its child subtree."""

        root = self._root
        if root is None:
            raise SearchError("cannot advance before a root state has been set")
        if root.state.is_terminal:
            raise SearchError("cannot advance a terminal root state")
        if not root.state.is_legal(action):
            raise SearchError(f"action {action!r} is not legal at the current root")

        child = root.children.get(action)
        if child is None:
            child = _Node(root.state.apply(action))
            root.children[action] = child
        self._root = child
        return child.state

    def run(self, state: GameState, *, add_root_noise: bool = False) -> SearchResult:
        """Add the configured number of simulations and return root statistics."""

        if not isinstance(state, GameState):
            raise TypeError("state must be a GameState")
        if not isinstance(add_root_noise, bool):
            raise TypeError("add_root_noise must be a bool")
        if state.is_terminal:
            raise SearchError("cannot search a terminal root state")

        if self._root is None:
            self._root = _Node(state)
        elif not _states_semantically_equal(self._root.state, state):
            raise SearchError("state does not match the current search root; call reset() first")

        root = self._root
        if not root.is_expanded:
            self._evaluate_and_expand(root)
        root.restore_base_priors()
        if add_root_noise:
            self._add_root_noise(root)

        for _ in range(self._simulations):
            self._simulate(root)

        return _search_result(root)

    def _evaluate_and_expand(self, node: _Node) -> float:
        evaluation = self._evaluator.evaluate_batch((node.state,))
        logits, value = _validate_evaluation(evaluation, node.state.action_size)
        legal_actions = node.state.legal_actions()
        if not legal_actions:
            raise SearchError("an evaluator was called for a state without legal actions")

        legal_indices = np.fromiter(legal_actions, dtype=np.intp)
        legal_logits = logits[legal_indices].astype(np.float64, copy=False)
        shifted = legal_logits - float(np.max(legal_logits))
        weights = np.exp(shifted)
        weight_sum = float(np.sum(weights))
        if not isfinite(weight_sum) or weight_sum <= 0.0:
            raise EvaluationError("legal policy logits could not be normalized")

        priors = np.zeros(node.state.action_size, dtype=np.float64)
        priors[legal_indices] = weights / weight_sum
        node.expand(priors)
        return value

    def _add_root_noise(self, root: _Node) -> None:
        priors = root.priors
        if priors is None:
            raise RuntimeError("root noise requires an expanded node")
        legal_actions = root.state.legal_actions()
        concentration = np.full(len(legal_actions), self._dirichlet_alpha, dtype=np.float64)
        noise = self._rng.dirichlet(concentration)
        fraction = self._dirichlet_fraction
        for index, action in enumerate(legal_actions):
            priors[action] = (1.0 - fraction) * priors[action] + fraction * noise[index]

    def _simulate(self, root: _Node) -> None:
        node = root
        path: list[tuple[_Node, int]] = []

        while node.is_expanded and not node.state.is_terminal:
            action = self._select_action(node)
            child = node.children.get(action)
            if child is None:
                child = _Node(node.state.apply(action))
                node.children[action] = child
            path.append((node, action))
            node = child

        if node.state.is_terminal:
            leaf_value = float(node.state.outcome(node.state.to_play))
        else:
            leaf_value = self._evaluate_and_expand(node)

        value = leaf_value
        for parent, action in reversed(path):
            value = -value
            parent.visit_counts[action] += 1
            parent.value_sums[action] += value

    def _select_action(self, node: _Node) -> int:
        priors = node.priors
        if priors is None:
            raise RuntimeError("PUCT selection requires an expanded node")

        total_visits = int(np.sum(node.visit_counts))
        exploration_scale = self._c_puct * sqrt(total_visits + 1.0)
        best_action: int | None = None
        best_score = -float("inf")
        for action in node.state.legal_actions():
            visits = int(node.visit_counts[action])
            q_value = 0.0 if visits == 0 else float(node.value_sums[action]) / visits
            exploration = exploration_scale * float(priors[action]) / (1.0 + visits)
            score = q_value + exploration
            if score > best_score:
                best_action = action
                best_score = score

        if best_action is None:
            raise SearchError("PUCT selection found no legal action")
        return best_action


def _validate_evaluation(
    evaluation: EvaluationBatch,
    action_size: int,
) -> tuple[NDArray[np.float32], float]:
    if not isinstance(evaluation, EvaluationBatch):
        raise EvaluationError("evaluator must return an EvaluationBatch")
    logits = evaluation.policy_logits
    values = evaluation.values
    if not isinstance(logits, np.ndarray) or not isinstance(values, np.ndarray):
        raise EvaluationError("evaluation outputs must be NumPy arrays")
    if logits.dtype != np.float32 or values.dtype != np.float32:
        raise EvaluationError("evaluation outputs must have dtype float32")
    if logits.shape != (1, action_size):
        raise EvaluationError(f"policy logits must have shape (1, {action_size})")
    if values.shape != (1,):
        raise EvaluationError("values must have shape (1,)")
    if not np.isfinite(logits).all() or not np.isfinite(values).all():
        raise EvaluationError("evaluation outputs must contain only finite values")
    value = float(values[0])
    if not -1.0 <= value <= 1.0:
        raise EvaluationError("evaluation values must lie in [-1, 1]")
    return logits[0], value


def _search_result(root: _Node) -> SearchResult:
    visit_counts = np.array(root.visit_counts, dtype=np.int64, order="C", copy=True)
    total_visits = int(np.sum(visit_counts))
    if total_visits <= 0:
        raise RuntimeError("a completed search must contain at least one root visit")

    maximum = int(np.max(visit_counts))
    selected_action = int(np.flatnonzero(visit_counts == maximum)[0])
    visit_policy = np.ascontiguousarray(
        visit_counts.astype(np.float32) / np.float32(total_visits),
        dtype=np.float32,
    )
    root_value = float(np.sum(root.value_sums)) / total_visits
    return SearchResult(
        selected_action=selected_action,
        visit_counts=visit_counts,
        visit_policy=visit_policy,
        root_value=root_value,
        simulations=total_visits,
    )


def _states_semantically_equal(first: GameState, second: GameState) -> bool:
    return (
        first.rules == second.rules
        and first.board == second.board
        and first.to_play is second.to_play
        and first.consecutive_passes == second.consecutive_passes
        and first.move_number == second.move_number
        and first.history == second.history
        and first.last_action == second.last_action
    )


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unsigned_64_bit_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_UINT64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return value


def _positive_finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return normalized


def _unit_interval_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return normalized


__all__ = ["MCTS", "SearchError", "SearchResult"]
