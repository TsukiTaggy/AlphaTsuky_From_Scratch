"""Tests for deterministic PUCT search and tree reuse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from azgo.evaluator import EvaluationBatch, EvaluationError
from azgo.game import GameState, Rules
from azgo.search import MCTS, SearchError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray


class FunctionalEvaluator:
    def __init__(
        self,
        logits: Callable[[GameState], NDArray[np.float32]] | None = None,
        value: Callable[[GameState], float] | None = None,
    ) -> None:
        self._logits = logits
        self._value = value
        self.calls: list[GameState] = []

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        assert len(states) == 1
        state = states[0]
        self.calls.append(state)
        logits = (
            np.zeros(state.action_size, dtype=np.float32)
            if self._logits is None
            else self._logits(state)
        )
        value = 0.0 if self._value is None else self._value(state)
        return EvaluationBatch(
            policy_logits=np.ascontiguousarray(logits[None, :], dtype=np.float32),
            values=np.asarray([value], dtype=np.float32),
        )


class MalformedEvaluator:
    def __init__(self, result: EvaluationBatch) -> None:
        self.result = result

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        assert len(states) == 1
        return self.result


def _search(
    evaluator: FunctionalEvaluator | MalformedEvaluator,
    *,
    simulations: int = 1,
    seed: int = 7,
    fraction: float = 0.25,
) -> MCTS:
    return MCTS(
        evaluator,
        simulations=simulations,
        c_puct=1.5,
        seed=seed,
        dirichlet_alpha=0.3,
        dirichlet_fraction=fraction,
    )


def test_legal_masking_and_stable_softmax_ignore_an_illegal_maximum() -> None:
    state = GameState.new(5).apply(0)

    def logits(position: GameState) -> NDArray[np.float32]:
        result = np.full(position.action_size, -1_000.0, dtype=np.float32)
        result[0] = 1_000.0  # Occupied and therefore illegal.
        result[1] = 999.0
        return result

    result = _search(FunctionalEvaluator(logits)).run(state)

    assert result.selected_action == 1
    assert result.visit_counts[0] == 0
    assert result.visit_counts[1] == 1


def test_root_is_expanded_before_exactly_the_configured_number_of_visits() -> None:
    evaluator = FunctionalEvaluator()
    state = GameState.new(5)

    result = _search(evaluator, simulations=7).run(state)

    assert result.simulations == 7
    assert int(result.visit_counts.sum()) == 7
    assert float(result.visit_policy.sum()) == pytest.approx(1.0)
    assert len(evaluator.calls) == 8  # Root expansion plus one new leaf per simulation.


def test_uniform_puct_ties_are_broken_by_the_smallest_action() -> None:
    result = _search(FunctionalEvaluator(), simulations=26).run(GameState.new(5))

    np.testing.assert_array_equal(result.visit_counts, np.ones(26, dtype=np.int64))
    assert result.selected_action == 0


def test_nonterminal_leaf_value_is_negated_to_the_parent_perspective() -> None:
    def logits(state: GameState) -> NDArray[np.float32]:
        result = np.full(state.action_size, -100.0, dtype=np.float32)
        result[0] = 100.0
        return result

    evaluator = FunctionalEvaluator(logits, lambda state: 0.75 if state.move_number else 0.0)

    result = _search(evaluator).run(GameState.new(5))

    assert result.selected_action == 0
    assert result.root_value == pytest.approx(-0.75)


def test_terminal_leaf_uses_exact_outcome_without_calling_evaluator() -> None:
    state = GameState.new(Rules(board_size=5, komi=7.5)).apply(25)

    def pass_first(position: GameState) -> NDArray[np.float32]:
        result = np.full(position.action_size, -100.0, dtype=np.float32)
        result[position.pass_action] = 100.0
        return result

    evaluator = FunctionalEvaluator(pass_first, lambda _state: -0.25)
    result = _search(evaluator).run(state)

    assert result.selected_action == state.pass_action
    assert result.root_value == pytest.approx(1.0)  # White wins the empty board by komi.
    assert evaluator.calls == [state]


def test_seeded_root_noise_is_reproducible_across_identical_instances() -> None:
    state = GameState.new(5).apply(0)
    first = _search(FunctionalEvaluator(), simulations=40, seed=90210, fraction=1.0)
    second = _search(FunctionalEvaluator(), simulations=40, seed=90210, fraction=1.0)

    first_result = first.run(state, add_root_noise=True)
    second_result = second.run(state, add_root_noise=True)

    np.testing.assert_array_equal(first_result.visit_counts, second_result.visit_counts)
    assert first_result.visit_counts[0] == 0
    assert second_result.visit_counts[0] == 0


def test_disabled_noise_is_deterministic_and_restores_base_priors() -> None:
    state = GameState.new(5)
    noisy = _search(FunctionalEvaluator(), simulations=20, seed=3, fraction=1.0)
    noisy.run(state, add_root_noise=True)
    after_noise = noisy.run(state, add_root_noise=False)

    reference = _search(FunctionalEvaluator(), simulations=40, seed=999, fraction=1.0)
    expected = reference.run(state, add_root_noise=False)

    # Earlier noisy visits remain, but a subsequent noiseless run is valid and normalized.
    assert after_noise.simulations == 40
    assert int(after_noise.visit_counts.sum()) == 40
    assert float(after_noise.visit_policy.sum()) == pytest.approx(1.0)
    assert expected.simulations == 40


@pytest.mark.parametrize(
    ("evaluation", "match"),
    [
        (
            EvaluationBatch(
                np.zeros((2, 26), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            ),
            "policy logits must have shape",
        ),
        (
            EvaluationBatch(
                np.zeros((1, 26), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
            ),
            "values must have shape",
        ),
        (
            EvaluationBatch(
                np.full((1, 26), np.nan, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            ),
            "finite",
        ),
        (
            EvaluationBatch(
                np.zeros((1, 26), dtype=np.float32),
                np.asarray([1.01], dtype=np.float32),
            ),
            r"\[-1, 1\]",
        ),
        (
            EvaluationBatch(
                np.zeros((1, 26), dtype=np.float64),
                np.zeros(1, dtype=np.float32),
            ),
            "dtype float32",
        ),
    ],
)
def test_malformed_evaluator_outputs_are_rejected(
    evaluation: EvaluationBatch,
    match: str,
) -> None:
    with pytest.raises(EvaluationError, match=match):
        _search(MalformedEvaluator(evaluation)).run(GameState.new(5))


def test_results_are_contiguous_write_protected_copies() -> None:
    state = GameState.new(5)
    snapshot = (state.board, state.history, state.hash_history, state.to_play)

    result = _search(FunctionalEvaluator(), simulations=3).run(state)

    assert result.visit_counts.flags.c_contiguous
    assert result.visit_policy.flags.c_contiguous
    assert not result.visit_counts.flags.writeable
    assert not result.visit_policy.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.visit_counts[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        result.visit_policy[0] = 1.0
    assert (state.board, state.history, state.hash_history, state.to_play) == snapshot


def test_advance_reuses_child_statistics_and_counts_new_simulations() -> None:
    search = _search(FunctionalEvaluator(), simulations=27)
    state = GameState.new(5)
    first = search.run(state)
    assert first.selected_action == 0

    child = search.advance(first.selected_action)
    reused = search.run(child)

    assert search.root_state == child
    assert reused.simulations == 28  # One prior child-edge visit plus 27 new visits.
    assert int(reused.visit_counts.sum()) == reused.simulations


def test_advance_can_create_an_unexpanded_child() -> None:
    search = _search(FunctionalEvaluator())
    state = GameState.new(5)
    search.reset(state)

    child = search.advance(7)

    assert child == state.apply(7)
    assert search.root_state == child


def test_run_requires_matching_root_until_reset() -> None:
    search = _search(FunctionalEvaluator())
    state = GameState.new(5)
    search.run(state)

    with pytest.raises(SearchError, match=r"reset\(\)"):
        search.run(state.apply(1))

    replacement = state.apply(2)
    search.reset(replacement)
    assert search.run(replacement).simulations == 1


def test_semantically_equal_reconstructed_root_is_accepted() -> None:
    search = _search(FunctionalEvaluator())
    search.run(GameState.new(5, zobrist_seed=1))

    result = search.run(GameState.new(5, zobrist_seed=999))

    assert result.simulations == 2


def test_terminal_and_invalid_tree_operations_raise_search_errors() -> None:
    search = _search(FunctionalEvaluator())
    terminal = GameState.new(5).apply(25).apply(25)

    with pytest.raises(SearchError, match="before a root"):
        search.advance(0)
    with pytest.raises(SearchError, match="terminal root"):
        search.run(terminal)

    search.reset(GameState.new(5).apply(0))
    with pytest.raises(SearchError, match="not legal"):
        search.advance(0)

    search.reset(terminal)
    with pytest.raises(SearchError, match="terminal root"):
        search.advance(25)


@pytest.mark.parametrize("simulations", [True, 0, -1, 1.5])
def test_constructor_rejects_invalid_simulation_counts(simulations: object) -> None:
    with pytest.raises(ValueError, match="simulations must be a positive integer"):
        MCTS(
            FunctionalEvaluator(),
            simulations=simulations,  # type: ignore[arg-type]
            c_puct=1.5,
            seed=0,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
        )


@pytest.mark.parametrize("seed", [True, -1, 1 << 64, 1.5])
def test_constructor_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit integer"):
        MCTS(
            FunctionalEvaluator(),
            simulations=1,
            c_puct=1.5,
            seed=seed,  # type: ignore[arg-type]
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
        )


@pytest.mark.parametrize("name", ["c_puct", "dirichlet_alpha"])
@pytest.mark.parametrize("invalid", [True, 0.0, -1.0, float("inf"), float("nan"), "1"])
def test_constructor_rejects_invalid_positive_floats(name: str, invalid: object) -> None:
    arguments: dict[str, object] = {
        "simulations": 1,
        "c_puct": 1.5,
        "seed": 0,
        "dirichlet_alpha": 0.3,
        "dirichlet_fraction": 0.25,
        name: invalid,
    }
    with pytest.raises(ValueError, match=rf"{name} must be a finite positive number"):
        MCTS(FunctionalEvaluator(), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fraction",
    [True, -0.01, 1.01, float("inf"), float("nan"), "0.25"],
)
def test_constructor_rejects_invalid_noise_fractions(fraction: object) -> None:
    with pytest.raises(ValueError, match=r"dirichlet_fraction.*\[0, 1\]"):
        MCTS(
            FunctionalEvaluator(),
            simulations=1,
            c_puct=1.5,
            seed=0,
            dirichlet_alpha=0.3,
            dirichlet_fraction=fraction,  # type: ignore[arg-type]
        )
