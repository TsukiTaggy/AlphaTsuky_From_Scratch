"""Tests for deterministic CPU policy-value optimization."""

from __future__ import annotations

import copy
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from torch import nn

from azgo.config import AppConfig, load_config
from azgo.game import Color
from azgo.learner import Learner, TrainingError
from azgo.network import PolicyValueNetwork
from azgo.replay import ReplayBatch, ReplayBuffer
from azgo.self_play import TrainingSample

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(**learner_updates: object) -> AppConfig:
    config = load_config(PROJECT_ROOT / "configs" / "engine" / "go5.yaml")
    learner = config.learner.model_copy(
        update={"batch_size": 2, "steps": 2, **learner_updates}
    )
    return config.model_copy(update={"learner": learner})


def _network(*, board_size: int = 5, history_length: int = 8) -> PolicyValueNetwork:
    return PolicyValueNetwork(
        board_size=board_size,
        history_length=history_length,
        channels=4,
        residual_blocks=1,
        value_hidden_size=4,
    )


def _batch(
    batch_size: int = 2,
    *,
    board_size: int = 5,
    history_length: int = 8,
) -> ReplayBatch:
    rng = np.random.default_rng(91)
    features = rng.integers(
        0,
        2,
        size=(batch_size, 2 * history_length + 1, board_size, board_size),
    ).astype(np.float32)
    policies = np.zeros((batch_size, board_size * board_size + 1), dtype=np.float32)
    for index in range(batch_size):
        policies[index, index] = 1.0
    values = np.linspace(-1.0, 1.0, batch_size, dtype=np.float32)
    return _batch_from_arrays(features, policies, values)


def _batch_from_arrays(
    features: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
) -> ReplayBatch:
    batch_size = features.shape[0] if features.ndim else 0
    return ReplayBatch(
        features=features,
        policies=policies,
        values=values,
        to_play=np.ones(batch_size, dtype=np.uint8),
        move_numbers=np.arange(batch_size, dtype=np.int64),
        selected_actions=np.arange(batch_size, dtype=np.int64),
        game_indices=np.arange(batch_size, dtype=np.uint64),
    )


def _sample(index: int, *, board_size: int = 5, history_length: int = 8) -> TrainingSample:
    features = np.zeros(
        (2 * history_length + 1, board_size, board_size),
        dtype=np.float32,
    )
    action = index % (board_size * board_size + 1)
    if action < board_size * board_size:
        features[0].flat[action] = 1.0
    policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
    policy[action] = 1.0
    return TrainingSample(
        features=features,
        policy=policy,
        value=float((-1, 0, 1)[index % 3]),
        to_play=Color.BLACK if index % 2 == 0 else Color.WHITE,
        move_number=index,
        selected_action=action,
        game_index=index,
    )


def _clone_model_state(network: PolicyValueNetwork) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in network.state_dict().items()}


def _assert_model_state(
    network: PolicyValueNetwork,
    expected: Mapping[str, torch.Tensor],
) -> None:
    actual = network.state_dict()
    assert set(actual) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0.0, atol=0.0)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key, value in left.items():
            _assert_nested_equal(value, right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


class RecordingReplayBuffer(ReplayBuffer):
    def __init__(self) -> None:
        super().__init__(5, history_length=8, capacity=8)
        self.extend(_sample(index) for index in range(6))
        self.requests: list[tuple[int, int, bool]] = []
        self.outputs: list[tuple[np.ndarray, np.ndarray]] = []

    def sample(self, batch_size: int, seed: int, augment: bool = False) -> ReplayBatch:
        self.requests.append((batch_size, seed, augment))
        batch = super().sample(batch_size, seed, augment)
        self.outputs.append((batch.game_indices.copy(), batch.selected_actions.copy()))
        return batch


def test_constructor_configures_sgd_and_exposes_state() -> None:
    config = _config(
        learning_rate=0.02,
        momentum=0.75,
        weight_decay=0.003,
    )
    network = _network()
    learner = Learner(network, config)

    assert learner.network is network
    assert isinstance(learner.optimizer, torch.optim.SGD)
    assert learner.optimizer.param_groups[0]["lr"] == 0.02
    assert learner.optimizer.param_groups[0]["momentum"] == 0.75
    assert learner.optimizer.param_groups[0]["weight_decay"] == 0.003
    assert learner.step == 0


def test_exact_soft_policy_and_weighted_value_losses() -> None:
    network = _network()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    learner = Learner(network, _config(value_loss_weight=2.0))

    metrics = learner.train_step(_batch())

    assert metrics.step == 1
    assert metrics.batch_size == 2
    assert metrics.policy_loss == pytest.approx(log(26), abs=1e-6)
    assert metrics.value_loss == pytest.approx(1.0, abs=1e-6)
    assert metrics.total_loss == pytest.approx(log(26) + 2.0, abs=1e-6)
    assert metrics.gradient_norm > 0.0
    assert learner.step == 1


def test_train_step_updates_both_output_heads() -> None:
    torch.manual_seed(7)
    network = _network()
    learner = Learner(network, _config())
    policy_output = network.policy_head[-1]
    value_output = network.value_head[-2]
    assert isinstance(policy_output, nn.Linear)
    assert isinstance(value_output, nn.Linear)
    policy_before = policy_output.bias.detach().clone()
    value_before = value_output.bias.detach().clone()

    learner.train_step(_batch())

    assert not torch.equal(policy_output.bias, policy_before)
    assert not torch.equal(value_output.bias, value_before)


def test_gradient_clipping_reports_pre_clip_norm_and_clips_gradients() -> None:
    network = _network()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    clip_norm = 1e-3
    learner = Learner(network, _config(gradient_clip_norm=clip_norm))

    metrics = learner.train_step(_batch())
    gradients = tuple(
        parameter.grad for parameter in network.parameters() if parameter.grad is not None
    )
    post_clip_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients])
    ).item()

    assert metrics.gradient_norm > clip_norm
    assert post_clip_norm <= clip_norm * 1.001


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("features", np.zeros((2, 17, 4, 5), dtype=np.float32), "features.*shape"),
        ("features", np.zeros((2, 17, 5, 5), dtype=np.float64), "features.*float32"),
        ("policies", np.zeros((2, 25), dtype=np.float32), "policies.*shape"),
        ("policies", np.ones((2, 26), dtype=np.float64) / 26.0, "policies.*float32"),
        ("policies", np.full((2, 26), 1.0 / 25.0, dtype=np.float32), "sum to one"),
        ("values", np.zeros((2, 1), dtype=np.float32), "values.*shape"),
        ("values", np.zeros(2, dtype=np.float64), "values.*float32"),
        ("values", np.asarray([-1.1, 0.0], dtype=np.float32), r"\[-1, 1\]"),
    ],
)
def test_train_step_rejects_malformed_batches_without_incrementing(
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    original = _batch()
    arrays = {
        "features": original.features,
        "policies": original.policies,
        "values": original.values,
    }
    arrays[field] = replacement
    learner = Learner(_network(), _config())

    with pytest.raises(TrainingError, match=message):
        learner.train_step(_batch_from_arrays(**arrays))

    assert learner.step == 0


@pytest.mark.parametrize("field", ["features", "policies", "values"])
def test_train_step_rejects_nonfinite_arrays(field: str) -> None:
    original = _batch()
    arrays = {
        "features": original.features.copy(),
        "policies": original.policies.copy(),
        "values": original.values.copy(),
    }
    arrays[field].flat[0] = np.nan
    learner = Learner(_network(), _config())

    with pytest.raises(TrainingError, match="finite"):
        learner.train_step(_batch_from_arrays(**arrays))

    assert learner.step == 0


def test_nonfinite_loss_and_gradient_do_not_increment_step() -> None:
    network = _network()
    learner = Learner(network, _config())
    policy_output = network.policy_head[-1]
    assert isinstance(policy_output, nn.Linear)
    with torch.no_grad():
        policy_output.bias.fill_(float("inf"))
    network.eval()
    loss_model_state = _clone_model_state(network)
    loss_optimizer_state = copy.deepcopy(learner.optimizer.state_dict())
    loss_rng_state = torch.get_rng_state().clone()
    loss_training_modes = tuple(module.training for module in network.modules())

    with pytest.raises(TrainingError, match="loss must be finite"):
        learner.train_step(_batch())
    assert learner.step == 0
    _assert_model_state(network, loss_model_state)
    _assert_nested_equal(learner.optimizer.state_dict(), loss_optimizer_state)
    torch.testing.assert_close(torch.get_rng_state(), loss_rng_state, rtol=0.0, atol=0.0)
    assert tuple(module.training for module in network.modules()) == loss_training_modes
    assert all(parameter.grad is None for parameter in network.parameters())

    with torch.no_grad():
        policy_output.bias.zero_()
    network.train()
    network.value_head.eval()
    gradient_model_state = _clone_model_state(network)
    gradient_optimizer_state = copy.deepcopy(learner.optimizer.state_dict())
    gradient_rng_state = torch.get_rng_state().clone()
    gradient_training_modes = tuple(module.training for module in network.modules())
    parameter = next(network.parameters())
    hook = parameter.register_hook(  # type: ignore[no-untyped-call]
        lambda gradient: torch.full_like(gradient, float("inf"))
    )
    with pytest.raises(TrainingError, match="gradient norm must be finite"):
        learner.train_step(_batch())
    hook.remove()
    assert learner.step == 0
    _assert_model_state(network, gradient_model_state)
    _assert_nested_equal(learner.optimizer.state_dict(), gradient_optimizer_state)
    torch.testing.assert_close(
        torch.get_rng_state(), gradient_rng_state, rtol=0.0, atol=0.0
    )
    assert tuple(module.training for module in network.modules()) == gradient_training_modes
    assert all(item.grad is None for item in network.parameters())


def test_optimizer_failure_rolls_back_and_retry_matches_uninterrupted_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(314159)
    config = _config(momentum=0.9, weight_decay=0.001)
    network = _network()
    learner = Learner(network, config)
    learner.train_step(_batch())
    assert learner.optimizer.state_dict()["state"]

    network.train()
    network.value_head.eval()
    model_before = _clone_model_state(network)
    optimizer_before = copy.deepcopy(learner.optimizer.state_dict())
    modes_before = tuple(module.training for module in network.modules())
    step_before = learner.step

    expected = Learner(_network(), config)
    expected.network.load_state_dict(model_before, strict=True)
    expected.optimizer.load_state_dict(copy.deepcopy(optimizer_before))
    expected.restore_step(step_before)
    for module, mode in zip(expected.network.modules(), modes_before, strict=True):
        module.training = mode

    torch.manual_seed(271828)
    rng_before = torch.get_rng_state().clone()
    original_step = cast("Callable[..., object]", learner.optimizer.step)

    def fail_after_update(*args: object, **kwargs: object) -> None:
        original_step(*args, **kwargs)
        torch.rand(7)
        raise RuntimeError("injected optimizer failure")

    with monkeypatch.context() as patch:
        patch.setattr(learner.optimizer, "step", fail_after_update)
        with pytest.raises(TrainingError, match="network optimization failed"):
            learner.train_step(_batch())

    assert learner.step == step_before
    _assert_model_state(network, model_before)
    _assert_nested_equal(learner.optimizer.state_dict(), optimizer_before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0.0, atol=0.0)
    assert tuple(module.training for module in network.modules()) == modes_before
    assert all(parameter.grad is None for parameter in network.parameters())

    torch.set_rng_state(rng_before.clone())
    expected_metrics = expected.train_step(_batch())
    expected_model = _clone_model_state(expected.network)
    expected_optimizer = copy.deepcopy(expected.optimizer.state_dict())

    torch.set_rng_state(rng_before.clone())
    actual_metrics = learner.train_step(_batch())

    assert actual_metrics == expected_metrics
    _assert_model_state(network, expected_model)
    _assert_nested_equal(learner.optimizer.state_dict(), expected_optimizer)
    assert learner.step == expected.step == step_before + 1


def test_train_steps_uses_step_keyed_seeded_augmentation_and_resumes() -> None:
    config = _config(seed=987654, batch_size=2, steps=3, augment=True)
    uninterrupted_replay = RecordingReplayBuffer()
    uninterrupted = Learner(_network(), config)

    summary = uninterrupted.train_steps(uninterrupted_replay)

    expected_seeds = [
        int(np.random.SeedSequence([config.learner.seed, step]).generate_state(1, np.uint64)[0])
        for step in range(3)
    ]
    assert uninterrupted_replay.requests == [
        (2, seed, True) for seed in expected_seeds
    ]
    assert summary.start_step == 0
    assert summary.end_step == 3
    assert tuple(metric.step for metric in summary.metrics) == (1, 2, 3)

    resumed_replay = RecordingReplayBuffer()
    resumed = Learner(_network(), config)
    resumed.restore_step(2)
    resumed_summary = resumed.train_steps(resumed_replay, steps=1)

    assert resumed_replay.requests[0] == uninterrupted_replay.requests[2]
    np.testing.assert_array_equal(
        resumed_replay.outputs[0][0], uninterrupted_replay.outputs[2][0]
    )
    np.testing.assert_array_equal(
        resumed_replay.outputs[0][1], uninterrupted_replay.outputs[2][1]
    )
    assert resumed_summary.start_step == 2
    assert resumed_summary.end_step == 3
    assert resumed_summary.metrics[0].step == 3


def test_training_summary_reports_mean_losses() -> None:
    summary = Learner(_network(), _config(steps=2)).train_steps(
        RecordingReplayBuffer()
    )

    assert summary.mean_policy_loss == pytest.approx(
        sum(metric.policy_loss for metric in summary.metrics) / 2
    )
    assert summary.mean_value_loss == pytest.approx(
        sum(metric.value_loss for metric in summary.metrics) / 2
    )
    assert summary.mean_total_loss == pytest.approx(
        sum(metric.total_loss for metric in summary.metrics) / 2
    )


def test_train_steps_rejects_invalid_requests_and_incompatible_replay() -> None:
    learner = Learner(_network(), _config(batch_size=2))
    insufficient = ReplayBuffer(5, history_length=8, capacity=2)
    insufficient.append(_sample(0))
    with pytest.raises(TrainingError, match="smaller than batch_size"):
        learner.train_steps(insufficient)

    wrong_board = ReplayBuffer(9, history_length=8, capacity=2)
    with pytest.raises(TrainingError, match="board_size"):
        learner.train_steps(wrong_board)

    wrong_history = ReplayBuffer(5, history_length=7, capacity=2)
    with pytest.raises(TrainingError, match="history_length"):
        learner.train_steps(wrong_history)

    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(TrainingError, match="steps must be a positive integer"):
            learner.train_steps(RecordingReplayBuffer(), steps=invalid)  # type: ignore[arg-type]


def test_constructor_restore_and_cpu_restrictions_are_strict() -> None:
    config = _config()
    with pytest.raises(TypeError, match="PolicyValueNetwork"):
        Learner(object(), config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AppConfig"):
        Learner(_network(), object())  # type: ignore[arg-type]
    with pytest.raises(TrainingError, match="board_size"):
        Learner(_network(board_size=9), config)
    with pytest.raises(TrainingError, match="history_length"):
        Learner(_network(history_length=7), config)
    with pytest.raises(TrainingError, match="CPU"):
        Learner(_network().to("meta"), config)

    learner = Learner(_network(), config)
    learner.restore_step(17)
    assert learner.step == 17
    for invalid in (True, -1, 1.5):
        with pytest.raises(TrainingError, match="non-negative integer"):
            learner.restore_step(invalid)  # type: ignore[arg-type]


def test_wrong_public_argument_types_are_rejected() -> None:
    learner = Learner(_network(), _config())
    with pytest.raises(TypeError, match="ReplayBatch"):
        learner.train_step(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ReplayBuffer"):
        learner.train_steps(object())  # type: ignore[arg-type]
