"""Residual policy-value network used by AlphaZero search and training."""

from __future__ import annotations

import torch
from torch import nn

from .game.types import SUPPORTED_BOARD_SIZES


def _validate_board_size(board_size: int) -> int:
    if (
        isinstance(board_size, bool)
        or not isinstance(board_size, int)
        or board_size not in SUPPORTED_BOARD_SIZES
    ):
        supported = ", ".join(str(size) for size in sorted(SUPPORTED_BOARD_SIZES))
        raise ValueError(f"board_size must be one of {{{supported}}}")
    return board_size


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return torch.relu(outputs + residual)


class PolicyValueNetwork(nn.Module):
    """Configurable residual network producing policy logits and a position value."""

    def __init__(
        self,
        *,
        board_size: int,
        history_length: int = 8,
        channels: int = 64,
        residual_blocks: int = 4,
        value_hidden_size: int = 64,
    ) -> None:
        super().__init__()

        self.board_size = _validate_board_size(board_size)
        self.history_length = _validate_positive_int("history_length", history_length)
        self.channels = _validate_positive_int("channels", channels)
        self.residual_blocks = _validate_positive_int("residual_blocks", residual_blocks)
        self.value_hidden_size = _validate_positive_int(
            "value_hidden_size", value_hidden_size
        )
        self.input_channels = 2 * self.history_length + 1
        self.action_size = self.board_size * self.board_size + 1

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                self.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(
            *(_ResidualBlock(self.channels) for _ in range(self.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.board_size * self.board_size, self.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size * self.board_size, self.value_hidden_size),
            nn.ReLU(),
            nn.Linear(self.value_hidden_size, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw policy logits and current-player values for a batch."""

        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4:
            raise ValueError(
                "inputs must have rank 4 with shape "
                "[batch, input_channels, board_size, board_size]"
            )
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"inputs must have {self.input_channels} channels, got {inputs.shape[1]}"
            )
        if inputs.shape[2:] != (self.board_size, self.board_size):
            raise ValueError(
                "inputs must have spatial dimensions "
                f"{self.board_size}x{self.board_size}, got {inputs.shape[2]}x{inputs.shape[3]}"
            )

        features = self.tower(self.stem(inputs))
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(1)
        return policy_logits, values
