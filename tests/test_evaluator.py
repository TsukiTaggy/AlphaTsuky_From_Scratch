"""Tests for batched MCTS evaluators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from azgo.encoding import encode_state
from azgo.evaluator import (
    EvaluationError,
    Evaluator,
    TorchEvaluator,
    UniformEvaluator,
)
from azgo.game import GameState
from azgo.network import PolicyValueNetwork

if TYPE_CHECKING:
    from collections.abc import Sequence


def _small_network(board_size: int = 5) -> PolicyValueNetwork:
    return PolicyValueNetwork(
        board_size=board_size,
        history_length=2,
        channels=2,
        residual_blocks=1,
        value_hidden_size=2,
    )


def test_concrete_evaluators_conform_to_runtime_protocol() -> None:
    assert isinstance(UniformEvaluator(), Evaluator)
    assert isinstance(TorchEvaluator(_small_network()), Evaluator)


def test_uniform_evaluator_returns_contiguous_float32_zeros() -> None:
    states = [GameState.new(5), GameState.new(5).apply(0)]

    result = UniformEvaluator().evaluate_batch(states)

    assert result.policy_logits.shape == (2, 26)
    assert result.values.shape == (2,)
    assert result.policy_logits.dtype == np.float32
    assert result.values.dtype == np.float32
    assert result.policy_logits.flags.c_contiguous
    assert result.values.flags.c_contiguous
    assert not result.policy_logits.any()
    assert not result.values.any()


@pytest.mark.parametrize("evaluator", [UniformEvaluator(), TorchEvaluator(_small_network())])
def test_evaluators_reject_empty_batches(evaluator: Evaluator) -> None:
    with pytest.raises(EvaluationError, match="at least one state"):
        evaluator.evaluate_batch([])


@pytest.mark.parametrize("evaluator", [UniformEvaluator(), TorchEvaluator(_small_network())])
def test_evaluators_reject_mixed_board_and_action_sizes(evaluator: Evaluator) -> None:
    with pytest.raises(EvaluationError, match="same board and action sizes"):
        evaluator.evaluate_batch([GameState.new(5), GameState.new(9)])


def test_torch_evaluator_rejects_states_incompatible_with_network() -> None:
    evaluator = TorchEvaluator(_small_network(5))

    with pytest.raises(EvaluationError, match="do not match"):
        evaluator.evaluate_batch([GameState.new(9)])


def test_torch_evaluator_switches_network_to_evaluation_mode() -> None:
    network = _small_network().train()

    evaluator = TorchEvaluator(network)

    assert evaluator.network is network
    assert not network.training


def test_torch_evaluator_matches_direct_network_evaluation() -> None:
    torch.manual_seed(101)
    network = _small_network()
    states = [GameState.new(5), GameState.new(5).apply(7)]
    evaluator = TorchEvaluator(network)
    inputs = torch.from_numpy(
        np.stack([encode_state(state, network.history_length) for state in states])
    )

    with torch.inference_mode():
        expected_policy, expected_values = network(inputs)
    result = evaluator.evaluate_batch(states)

    np.testing.assert_array_equal(result.policy_logits, expected_policy.numpy())
    np.testing.assert_array_equal(result.values, expected_values.numpy())
    assert result.policy_logits.flags.c_contiguous
    assert result.values.flags.c_contiguous


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_torch_evaluator_supports_every_board_size(board_size: int) -> None:
    network = _small_network(board_size)

    result = TorchEvaluator(network).evaluate_batch([GameState.new(board_size)])

    assert result.policy_logits.shape == (1, board_size * board_size + 1)
    assert result.values.shape == (1,)
    assert result.policy_logits.dtype == np.float32
    assert result.values.dtype == np.float32
    assert np.isfinite(result.policy_logits).all()
    assert np.isfinite(result.values).all()
    assert (result.values >= -1.0).all()
    assert (result.values <= 1.0).all()


class _OutputNetwork(PolicyValueNetwork):
    def __init__(self, output_kind: str) -> None:
        super().__init__(
            board_size=5,
            history_length=1,
            channels=2,
            residual_blocks=1,
            value_hidden_size=2,
        )
        self.output_kind = output_kind

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]
        policy = torch.zeros(batch_size, self.action_size, device=inputs.device)
        values = torch.zeros(batch_size, device=inputs.device)
        if self.output_kind == "policy_shape":
            policy = policy[:, :-1]
        elif self.output_kind == "value_shape":
            values = values[:, None]
        elif self.output_kind == "policy_nan":
            policy[0, 0] = torch.nan
        elif self.output_kind == "value_inf":
            values[0] = torch.inf
        elif self.output_kind == "value_range":
            values[0] = 1.01
        return policy, values


@pytest.mark.parametrize(
    ("output_kind", "message"),
    [
        ("policy_shape", "policy logits must have shape"),
        ("value_shape", "values must have shape"),
        ("policy_nan", "policy logits must all be finite"),
        ("value_inf", "values must all be finite"),
        ("value_range", r"values must lie in \[-1, 1\]"),
    ],
)
def test_torch_evaluator_rejects_malformed_model_outputs(
    output_kind: str,
    message: str,
) -> None:
    evaluator = TorchEvaluator(_OutputNetwork(output_kind))

    with pytest.raises(EvaluationError, match=message):
        evaluator.evaluate_batch([GameState.new(5)])


class _FailingNetwork(PolicyValueNetwork):
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("deliberate model failure")


def test_torch_evaluator_wraps_model_failures() -> None:
    network = _FailingNetwork(
        board_size=5,
        history_length=1,
        channels=2,
        residual_blocks=1,
        value_hidden_size=2,
    )

    with pytest.raises(EvaluationError, match="network evaluation failed") as error:
        TorchEvaluator(network).evaluate_batch([GameState.new(5)])

    assert isinstance(error.value.__cause__, RuntimeError)


def test_evaluation_batch_fields_are_frozen() -> None:
    result = UniformEvaluator().evaluate_batch([GameState.new(5)])

    with pytest.raises(AttributeError):
        result.values = np.ones(1, dtype=np.float32)  # type: ignore[misc]


def test_evaluators_reject_non_state_members() -> None:
    invalid_states: Sequence[GameState] = [object()]  # type: ignore[list-item]

    with pytest.raises(EvaluationError, match="GameState"):
        UniformEvaluator().evaluate_batch(invalid_states)
