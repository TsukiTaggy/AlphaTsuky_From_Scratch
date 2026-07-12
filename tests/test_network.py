"""Tests for the residual policy-value network."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from azgo.network import PolicyValueNetwork


def _small_network(board_size: int = 5) -> PolicyValueNetwork:
    return PolicyValueNetwork(
        board_size=board_size,
        history_length=2,
        channels=8,
        residual_blocks=1,
        value_hidden_size=8,
    )


def test_default_network_dimensions() -> None:
    network = PolicyValueNetwork(board_size=5)

    assert network.input_channels == 17
    assert network.action_size == 26
    assert network.history_length == 8
    assert network.channels == 64
    assert network.residual_blocks == 4
    assert network.value_hidden_size == 64


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_forward_shapes_are_finite_on_cpu(board_size: int) -> None:
    network = _small_network(board_size)
    inputs = torch.randn(2, network.input_channels, board_size, board_size)

    policy_logits, values = network(inputs)

    assert policy_logits.shape == (2, board_size * board_size + 1)
    assert values.shape == (2,)
    assert policy_logits.device.type == "cpu"
    assert values.device.type == "cpu"
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


@pytest.mark.parametrize(
    "argument", ["history_length", "channels", "residual_blocks", "value_hidden_size"]
)
@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_constructor_rejects_invalid_positive_counts(argument: str, invalid: object) -> None:
    arguments: dict[str, object] = {"board_size": 5, argument: invalid}

    with pytest.raises(ValueError, match=rf"{argument} must be a positive integer"):
        PolicyValueNetwork(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("board_size", [True, 5.0, 0, 4, 6, 8, 20])
def test_constructor_rejects_unsupported_board_sizes(board_size: object) -> None:
    with pytest.raises(ValueError, match="board_size must be one of"):
        PolicyValueNetwork(board_size=board_size)  # type: ignore[arg-type]


def test_backward_propagates_gradients_through_both_heads() -> None:
    network = _small_network()
    inputs = torch.randn(2, network.input_channels, 5, 5)
    policy_targets = torch.tensor([3, 17])
    value_targets = torch.tensor([0.75, -0.5])

    policy_logits, values = network(inputs)
    loss = nn.functional.cross_entropy(policy_logits, policy_targets)
    loss = loss + nn.functional.mse_loss(values, value_targets)
    loss.backward()  # type: ignore[no-untyped-call]

    policy_output = network.policy_head[-1]
    value_output = network.value_head[-2]
    assert isinstance(policy_output, nn.Linear)
    assert isinstance(value_output, nn.Linear)
    assert policy_output.weight.grad is not None
    assert value_output.weight.grad is not None
    assert torch.count_nonzero(policy_output.weight.grad) > 0
    assert torch.count_nonzero(value_output.weight.grad) > 0


def test_state_dict_reload_is_deterministic_in_evaluation_mode() -> None:
    torch.manual_seed(90210)
    original = _small_network().eval()
    restored = _small_network().eval()
    restored.load_state_dict(original.state_dict())
    inputs = torch.randn(3, original.input_channels, 5, 5)

    with torch.no_grad():
        expected = original(inputs)
        actual = restored(inputs)

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_forward_rejects_non_tensor_input() -> None:
    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        _small_network().forward("not a tensor")  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(5, 5, 5), (1, 5, 5, 5, 1)])
def test_forward_rejects_wrong_rank(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="rank 4"):
        _small_network()(torch.randn(shape))


def test_forward_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="must have 5 channels"):
        _small_network()(torch.randn(1, 3, 5, 5))


@pytest.mark.parametrize("shape", [(1, 5, 4, 5), (1, 5, 5, 6), (1, 5, 9, 9)])
def test_forward_rejects_wrong_spatial_dimensions(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="spatial dimensions 5x5"):
        _small_network()(torch.randn(shape))
