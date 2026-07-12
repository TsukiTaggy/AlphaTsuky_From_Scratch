"""Batched position evaluators for Monte Carlo tree search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

from .encoding import encode_state
from .game import GameState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from .network import PolicyValueNetwork


class EvaluationError(ValueError):
    """Raised when an evaluator input or output violates its public contract."""


@dataclass(frozen=True, slots=True)
class EvaluationBatch:
    """Contiguous CPU results for a homogeneous batch of game states."""

    policy_logits: NDArray[np.float32]
    values: NDArray[np.float32]


@runtime_checkable
class Evaluator(Protocol):
    """Interface used by search to evaluate one or more positions."""

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        """Return policy logits and current-player values for ``states``."""
        ...


def _batch_dimensions(states: Sequence[GameState]) -> tuple[int, int]:
    if len(states) == 0:
        raise EvaluationError("evaluation requires at least one state")

    first = states[0]
    if not isinstance(first, GameState):
        raise EvaluationError("states must contain only GameState instances")
    board_size = first.board_size
    action_size = first.action_size
    for state in states[1:]:
        if not isinstance(state, GameState):
            raise EvaluationError("states must contain only GameState instances")
        if state.board_size != board_size or state.action_size != action_size:
            raise EvaluationError("all states in a batch must have the same board and action sizes")
    return board_size, action_size


class UniformEvaluator:
    """Deterministic baseline evaluator returning zero logits and values."""

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        """Return uniform-softmax logits and neutral values for ``states``."""

        _, action_size = _batch_dimensions(states)
        batch_size = len(states)
        return EvaluationBatch(
            policy_logits=np.zeros((batch_size, action_size), dtype=np.float32),
            values=np.zeros(batch_size, dtype=np.float32),
        )


class TorchEvaluator:
    """Evaluate encoded positions with a :class:`PolicyValueNetwork`.

    Construction switches ``network`` to evaluation mode. The evaluator does
    not restore training mode after inference; callers beginning a later
    training phase must explicitly call ``network.train()``.
    """

    def __init__(self, network: PolicyValueNetwork) -> None:
        self.network = network
        self.network.eval()

    def evaluate_batch(self, states: Sequence[GameState]) -> EvaluationBatch:
        """Evaluate a non-empty homogeneous batch on the network's device."""

        board_size, action_size = _batch_dimensions(states)
        if board_size != self.network.board_size or action_size != self.network.action_size:
            raise EvaluationError(
                "state dimensions do not match the evaluator network: "
                f"expected board size {self.network.board_size} and action size "
                f"{self.network.action_size}, got {board_size} and {action_size}"
            )

        parameter = next(self.network.parameters(), None)
        if parameter is None:
            raise EvaluationError("evaluator network has no parameters")

        try:
            encoded = np.stack(
                [encode_state(state, self.network.history_length) for state in states],
                axis=0,
            )
            inputs = torch.from_numpy(encoded).to(device=parameter.device)
            with torch.inference_mode():
                outputs = self.network(inputs)
        except Exception as exc:
            raise EvaluationError("network evaluation failed") from exc

        return self._normalize_outputs(outputs, len(states), action_size)

    @staticmethod
    def _normalize_outputs(
        outputs: object,
        batch_size: int,
        action_size: int,
    ) -> EvaluationBatch:
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise EvaluationError("network must return a (policy_logits, values) tuple")
        policy_logits, values = outputs
        if not isinstance(policy_logits, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise EvaluationError("network outputs must be torch.Tensor instances")
        if policy_logits.shape != (batch_size, action_size):
            raise EvaluationError(
                "network policy logits must have shape "
                f"({batch_size}, {action_size}), got {tuple(policy_logits.shape)}"
            )
        if values.shape != (batch_size,):
            raise EvaluationError(
                f"network values must have shape ({batch_size},), got {tuple(values.shape)}"
            )

        try:
            policy_cpu = policy_logits.detach().to(device="cpu", dtype=torch.float32)
            values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(policy_cpu).all()):
                raise EvaluationError("network policy logits must all be finite")
            if not bool(torch.isfinite(values_cpu).all()):
                raise EvaluationError("network values must all be finite")
            if not bool(((values_cpu >= -1.0) & (values_cpu <= 1.0)).all()):
                raise EvaluationError("network values must lie in [-1, 1]")
            policy_array = np.ascontiguousarray(policy_cpu.numpy(), dtype=np.float32)
            value_array = np.ascontiguousarray(values_cpu.numpy(), dtype=np.float32)
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(
                "network outputs could not be converted to CPU float32 arrays"
            ) from exc

        return EvaluationBatch(policy_logits=policy_array, values=value_array)
