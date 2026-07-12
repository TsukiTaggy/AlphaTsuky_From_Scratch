"""CPU-first policy-value optimization over deterministic replay samples."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from azgo.config import AppConfig
from azgo.network import PolicyValueNetwork
from azgo.replay import ReplayBatch, ReplayBuffer, ReplayError

if TYPE_CHECKING:
    from collections.abc import Iterable


_POLICY_ATOL = 1e-6
_POLICY_RTOL = 1e-5


class TrainingError(ValueError):
    """Raised when learner input or optimization state violates its contract."""


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    """Detached scalar measurements from one successful optimizer update."""

    step: int
    batch_size: int
    policy_loss: float
    value_loss: float
    total_loss: float
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Measurements and global-step bounds for a sequence of updates."""

    start_step: int
    end_step: int
    metrics: tuple[TrainingMetrics, ...]

    def __post_init__(self) -> None:
        try:
            normalized = tuple(self.metrics)
        except TypeError as exc:
            raise TrainingError("metrics must be an iterable of TrainingMetrics") from exc
        if any(not isinstance(metric, TrainingMetrics) for metric in normalized):
            raise TrainingError("metrics must contain only TrainingMetrics")
        object.__setattr__(self, "metrics", normalized)

    @property
    def mean_policy_loss(self) -> float:
        """Arithmetic mean of the policy losses in this run."""

        return _mean(metric.policy_loss for metric in self.metrics)

    @property
    def mean_value_loss(self) -> float:
        """Arithmetic mean of the value losses in this run."""

        return _mean(metric.value_loss for metric in self.metrics)

    @property
    def mean_total_loss(self) -> float:
        """Arithmetic mean of the weighted total losses in this run."""

        return _mean(metric.total_loss for metric in self.metrics)


class Learner:
    """Train a configured policy-value network from a replay buffer on CPU."""

    def __init__(self, network: PolicyValueNetwork, config: AppConfig) -> None:
        if not isinstance(network, PolicyValueNetwork):
            raise TypeError("network must be a PolicyValueNetwork")
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if network.board_size != config.game.board_size:
            raise TrainingError(
                "network board_size does not match configuration: "
                f"expected {config.game.board_size}, got {network.board_size}"
            )
        if network.history_length != config.model.history_length:
            raise TrainingError(
                "network history_length does not match configuration: "
                f"expected {config.model.history_length}, got {network.history_length}"
            )

        parameters = tuple(network.parameters())
        if not parameters:
            raise TrainingError("network must have trainable parameters")
        _require_cpu_parameters(parameters)

        self._network = network
        self._config = config
        self._optimizer = torch.optim.SGD(
            parameters,
            lr=config.learner.learning_rate,
            momentum=config.learner.momentum,
            weight_decay=config.learner.weight_decay,
        )
        self._step = 0

    @property
    def network(self) -> PolicyValueNetwork:
        """The network updated by this learner."""

        return self._network

    @property
    def optimizer(self) -> torch.optim.SGD:
        """The SGD optimizer whose state is part of a training checkpoint."""

        return self._optimizer

    @property
    def step(self) -> int:
        """Number of optimizer updates completed or restored."""

        return self._step

    def restore_step(self, step: int) -> None:
        """Restore the global update counter after loading checkpoint state."""

        self._step = _nonnegative_integer(step, "step")

    def train_step(self, batch: ReplayBatch) -> TrainingMetrics:
        """Validate and optimize one replay batch, incrementing on success only."""

        if not isinstance(batch, ReplayBatch):
            raise TypeError("batch must be a ReplayBatch")
        batch_size = self._validate_batch(batch)
        parameters = tuple(self._network.parameters())
        _require_cpu_parameters(parameters)

        features = torch.tensor(batch.features, dtype=torch.float32, device="cpu")
        policy_targets = torch.tensor(batch.policies, dtype=torch.float32, device="cpu")
        value_targets = torch.tensor(batch.values, dtype=torch.float32, device="cpu")

        try:
            model_state = copy.deepcopy(self._network.state_dict())
            optimizer_state = copy.deepcopy(self._optimizer.state_dict())
            rng_state = torch.get_rng_state().clone()
            training_modes = tuple(
                (module, module.training) for module in self._network.modules()
            )
        except Exception as exc:
            raise TrainingError("could not snapshot training state") from exc

        try:
            self._network.train()
            self._optimizer.zero_grad(set_to_none=True)
            outputs = self._network(features)
            policy_logits, predicted_values = _validate_network_outputs(
                outputs,
                batch_size,
                self._network.action_size,
            )
            log_probabilities = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(
                policy_targets * log_probabilities,
                dim=1,
            ).mean()
            value_loss = torch.nn.functional.mse_loss(predicted_values, value_targets)
            total_loss = policy_loss + self._config.learner.value_loss_weight * value_loss
            if not bool(torch.isfinite(total_loss)):
                raise TrainingError("training loss must be finite")

            total_loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=self._config.learner.gradient_clip_norm,
            )
            if not bool(torch.isfinite(gradient_norm_tensor)):
                raise TrainingError("gradient norm must be finite")
            gradient_norm = float(gradient_norm_tensor.detach().item())
            self._optimizer.step()
        except Exception as exc:
            try:
                self._restore_training_state(
                    model_state,
                    optimizer_state,
                    rng_state,
                    training_modes,
                )
            except Exception as rollback_error:
                raise TrainingError(
                    "network optimization failed and training state could not be restored"
                ) from rollback_error
            if isinstance(exc, TrainingError):
                raise
            raise TrainingError("network optimization failed") from exc

        self._step += 1
        return TrainingMetrics(
            step=self._step,
            batch_size=batch_size,
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            total_loss=float(total_loss.detach().item()),
            gradient_norm=gradient_norm,
        )

    def train_steps(
        self,
        replay: ReplayBuffer,
        steps: int | None = None,
    ) -> TrainingSummary:
        """Run deterministic replay sampling and optimization for several steps."""

        if not isinstance(replay, ReplayBuffer):
            raise TypeError("replay must be a ReplayBuffer")
        count = self._config.learner.steps if steps is None else _positive_integer(steps, "steps")
        if replay.board_size != self._network.board_size:
            raise TrainingError(
                "replay board_size does not match learner network: "
                f"expected {self._network.board_size}, got {replay.board_size}"
            )
        if replay.history_length != self._network.history_length:
            raise TrainingError(
                "replay history_length does not match learner network: "
                f"expected {self._network.history_length}, got {replay.history_length}"
            )
        batch_size = self._config.learner.batch_size
        if len(replay) < batch_size:
            raise TrainingError(
                f"replay size {len(replay)} is smaller than batch_size {batch_size}"
            )

        start_step = self._step
        metrics: list[TrainingMetrics] = []
        for _ in range(count):
            sample_seed = _sample_seed(self._config.learner.seed, self._step)
            try:
                batch = replay.sample(
                    batch_size,
                    sample_seed,
                    augment=self._config.learner.augment,
                )
            except ReplayError as exc:
                raise TrainingError("could not sample a learner replay batch") from exc
            metrics.append(self.train_step(batch))
        return TrainingSummary(start_step, self._step, tuple(metrics))

    def _validate_batch(self, batch: ReplayBatch) -> int:
        features = batch.features
        policies = batch.policies
        values = batch.values
        if features.ndim != 4:
            raise TrainingError(
                "batch features must have shape "
                "[batch, input_channels, board_size, board_size]"
            )
        batch_size = int(features.shape[0])
        if batch_size <= 0:
            raise TrainingError("batch_size must be positive")
        expected_features = (
            batch_size,
            self._network.input_channels,
            self._network.board_size,
            self._network.board_size,
        )
        if features.shape != expected_features:
            raise TrainingError(f"batch features must have shape {expected_features}")
        expected_policies = (batch_size, self._network.action_size)
        if policies.shape != expected_policies:
            raise TrainingError(f"batch policies must have shape {expected_policies}")
        if values.shape != (batch_size,):
            raise TrainingError(f"batch values must have shape ({batch_size},)")

        for name, array in (
            ("features", features),
            ("policies", policies),
            ("values", values),
        ):
            if array.dtype != np.float32:
                raise TrainingError(f"batch {name} must have dtype float32")
            if not bool(np.isfinite(array).all()):
                raise TrainingError(f"batch {name} must contain only finite values")
        if bool(np.any(policies < 0.0)):
            raise TrainingError("batch policies must be non-negative")
        if not bool(
            np.allclose(
                np.sum(policies, axis=1, dtype=np.float64),
                1.0,
                rtol=_POLICY_RTOL,
                atol=_POLICY_ATOL,
            )
        ):
            raise TrainingError("every batch policy must sum to one")
        if bool(np.any(np.abs(values) > 1.0)):
            raise TrainingError("batch values must lie in [-1, 1]")
        return batch_size

    def _restore_training_state(
        self,
        model_state: dict[str, torch.Tensor],
        optimizer_state: dict[str, object],
        rng_state: torch.Tensor,
        training_modes: tuple[tuple[torch.nn.Module, bool], ...],
    ) -> None:
        self._network.load_state_dict(model_state, strict=True)
        self._optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        torch.set_rng_state(rng_state)
        for module, mode in training_modes:
            module.training = mode
        for parameter in self._network.parameters():
            parameter.grad = None


def _validate_network_outputs(
    outputs: object,
    batch_size: int,
    action_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise TrainingError("network must return a (policy_logits, values) tuple")
    policy_logits, values = outputs
    if not isinstance(policy_logits, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise TrainingError("network outputs must be torch.Tensor instances")
    if policy_logits.shape != (batch_size, action_size):
        raise TrainingError(
            "network policy logits must have shape "
            f"({batch_size}, {action_size}), got {tuple(policy_logits.shape)}"
        )
    if values.shape != (batch_size,):
        raise TrainingError(
            f"network values must have shape ({batch_size},), got {tuple(values.shape)}"
        )
    if policy_logits.device.type != "cpu" or values.device.type != "cpu":
        raise TrainingError("network outputs must remain on CPU")
    return policy_logits, values


def _require_cpu_parameters(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    if any(parameter.device.type != "cpu" for parameter in parameters):
        raise TrainingError("learner network parameters must be on CPU")


def _sample_seed(seed: int, step: int) -> int:
    sequence = np.random.SeedSequence([seed, step])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingError(f"{name} must be a non-negative integer")
    return value


def _mean(values: Iterable[float]) -> float:
    normalized = tuple(values)
    if not normalized:
        raise TrainingError("cannot compute a mean for an empty training summary")
    return float(sum(normalized) / len(normalized))


__all__ = [
    "Learner",
    "TrainingError",
    "TrainingMetrics",
    "TrainingSummary",
]
