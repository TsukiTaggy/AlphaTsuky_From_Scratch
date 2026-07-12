"""The eight dihedral symmetries of a square Go board."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


class Symmetry(StrEnum):
    """A square-board symmetry in row-major coordinates.

    Rotations are counterclockwise. ``FLIP_HORIZONTAL`` reflects across the
    horizontal axis (swapping top and bottom), while ``FLIP_VERTICAL`` reflects
    across the vertical axis (swapping left and right). Diagonal flips reflect
    across the top-left-to-bottom-right main diagonal or the opposite diagonal.
    """

    IDENTITY = "identity"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    FLIP_HORIZONTAL = "flip_horizontal"
    FLIP_VERTICAL = "flip_vertical"
    FLIP_MAIN_DIAGONAL = "flip_main_diagonal"
    FLIP_ANTI_DIAGONAL = "flip_anti_diagonal"

    @property
    def inverse(self) -> Symmetry:
        """Return the symmetry that exactly reverses this transformation."""

        if self is Symmetry.ROTATE_90:
            return Symmetry.ROTATE_270
        if self is Symmetry.ROTATE_270:
            return Symmetry.ROTATE_90
        return self

    def transform_action(self, action: int, board_size: int) -> int:
        """Transform a row-major board action, leaving pass unchanged."""

        size = _validate_board_size(board_size)
        normalized_action = _validate_action(action, size)
        if normalized_action == size * size:
            return normalized_action

        row, col = divmod(normalized_action, size)
        if self is Symmetry.IDENTITY:
            transformed = (row, col)
        elif self is Symmetry.ROTATE_90:
            transformed = (size - 1 - col, row)
        elif self is Symmetry.ROTATE_180:
            transformed = (size - 1 - row, size - 1 - col)
        elif self is Symmetry.ROTATE_270:
            transformed = (col, size - 1 - row)
        elif self is Symmetry.FLIP_HORIZONTAL:
            transformed = (size - 1 - row, col)
        elif self is Symmetry.FLIP_VERTICAL:
            transformed = (row, size - 1 - col)
        elif self is Symmetry.FLIP_MAIN_DIAGONAL:
            transformed = (col, row)
        else:
            transformed = (size - 1 - col, size - 1 - row)
        return transformed[0] * size + transformed[1]

    def transform_features(self, features: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
        """Transform the final two axes of square feature tensors.

        Any number of leading dimensions is accepted, including none. The
        returned array preserves the input dtype and shape and is C-contiguous,
        so it can be passed directly to tensor libraries such as PyTorch.
        """

        if features.ndim < 2:
            raise ValueError("features must have at least two dimensions")
        if features.shape[-2] != features.shape[-1] or features.shape[-1] == 0:
            raise ValueError("features must have non-empty square spatial dimensions")

        if self is Symmetry.IDENTITY:
            transformed = features
        elif self is Symmetry.ROTATE_90:
            transformed = np.rot90(features, 1, axes=(-2, -1))
        elif self is Symmetry.ROTATE_180:
            transformed = np.rot90(features, 2, axes=(-2, -1))
        elif self is Symmetry.ROTATE_270:
            transformed = np.rot90(features, 3, axes=(-2, -1))
        elif self is Symmetry.FLIP_HORIZONTAL:
            transformed = np.flip(features, axis=-2)
        elif self is Symmetry.FLIP_VERTICAL:
            transformed = np.flip(features, axis=-1)
        elif self is Symmetry.FLIP_MAIN_DIAGONAL:
            transformed = np.swapaxes(features, -2, -1)
        else:
            transformed = np.flip(np.swapaxes(features, -2, -1), axis=(-2, -1))
        return np.ascontiguousarray(transformed)

    def transform_policy(
        self,
        policy: NDArray[_ScalarT],
        board_size: int,
    ) -> NDArray[_ScalarT]:
        """Transform policy vectors shaped ``[..., N*N+1]``.

        Board-action entries follow the same spatial transformation as feature
        tensors. The final pass entry is never moved.
        """

        size = _validate_board_size(board_size)
        action_size = size * size + 1
        if policy.ndim < 1 or policy.shape[-1] != action_size:
            raise ValueError(f"policy must have final dimension {action_size}")

        leading_shape = policy.shape[:-1]
        board_policy = policy[..., :-1].reshape((*leading_shape, size, size))
        transformed_board = self.transform_features(board_policy)
        flattened_board = transformed_board.reshape((*leading_shape, size * size))
        return np.concatenate((flattened_board, policy[..., -1:]), axis=-1)


def _validate_board_size(board_size: int) -> int:
    if isinstance(board_size, bool):
        raise ValueError("board_size must be a positive integer")
    try:
        normalized = operator.index(board_size)
    except TypeError as exc:
        raise ValueError("board_size must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError("board_size must be a positive integer")
    return normalized


def _validate_action(action: int, board_size: int) -> int:
    if isinstance(action, bool):
        raise ValueError("action must be an integer")
    try:
        normalized = operator.index(action)
    except TypeError as exc:
        raise ValueError("action must be an integer") from exc
    maximum = board_size * board_size
    if not 0 <= normalized <= maximum:
        raise ValueError(f"action must be in [0, {maximum}]")
    return normalized
