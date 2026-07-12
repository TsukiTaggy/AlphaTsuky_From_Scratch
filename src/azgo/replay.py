"""Bounded replay storage and portable Phase 5 dataset snapshots."""

from __future__ import annotations

import os
import sys
import tempfile
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from azgo.game import SUPPORTED_BOARD_SIZES, Color
from azgo.self_play import SelfPlayGame, TrainingSample
from azgo.symmetry import Symmetry

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from numpy.typing import NDArray


_FORMAT_VERSION: Final = 1
_UINT64_MAX: Final = int(np.iinfo(np.uint64).max)
_POLICY_ATOL: Final = 1e-6
_POLICY_RTOL: Final = 1e-5
_SNAPSHOT_KEYS: Final = frozenset(
    {
        "version",
        "board_size",
        "history_length",
        "capacity",
        "next_game_index",
        "features",
        "policies",
        "values",
        "to_play",
        "move_numbers",
        "selected_actions",
        "game_indices",
    }
)

_VERSION_DTYPE: Final = np.dtype(np.uint32)
_SCALAR_DTYPE: Final = np.dtype(np.uint64)
_COLOR_DTYPE: Final = np.dtype(np.uint8)
_MOVE_DTYPE: Final = np.dtype(np.int64)
_ACTION_DTYPE: Final = np.dtype(np.int64)
_GAME_INDEX_DTYPE: Final = np.dtype(np.uint64)


class ReplayError(ValueError):
    """Raised when replay data or a replay operation is invalid."""


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    """A detached, immutable batch sampled from a replay buffer."""

    features: NDArray[np.float32]
    policies: NDArray[np.float32]
    values: NDArray[np.float32]
    to_play: NDArray[np.uint8]
    move_numbers: NDArray[np.int64]
    selected_actions: NDArray[np.int64]
    game_indices: NDArray[np.uint64]

    def __post_init__(self) -> None:
        arrays = (
            "features",
            "policies",
            "values",
            "to_play",
            "move_numbers",
            "selected_actions",
            "game_indices",
        )
        for name in arrays:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise ReplayError(f"{name} must be a NumPy array")
            copied = np.array(value, copy=True, order="C")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


class ReplayBuffer:
    """A fixed-shape, position-capacity FIFO replay buffer."""

    def __init__(
        self,
        board_size: int,
        history_length: int,
        capacity: int,
        next_game_index: int = 0,
    ) -> None:
        if (
            isinstance(board_size, bool)
            or not isinstance(board_size, int)
            or board_size not in SUPPORTED_BOARD_SIZES
        ):
            supported = ", ".join(str(size) for size in sorted(SUPPORTED_BOARD_SIZES))
            raise ReplayError(f"board_size must be one of {{{supported}}}")
        self._board_size = board_size
        self._history_length = _positive_int(history_length, "history_length")
        self._capacity = _positive_int(capacity, "capacity")
        if self._capacity > sys.maxsize:
            raise ReplayError(f"capacity must be at most {sys.maxsize}")
        self._next_game_index = _uint64(next_game_index, "next_game_index")
        self._samples: deque[TrainingSample] = deque(maxlen=self._capacity)

    @property
    def board_size(self) -> int:
        """Board width expected by every stored feature and policy."""

        return self._board_size

    @property
    def history_length(self) -> int:
        """Number of historical positions represented in each feature tensor."""

        return self._history_length

    @property
    def capacity(self) -> int:
        """Maximum number of positions retained by this buffer."""

        return self._capacity

    @property
    def next_game_index(self) -> int:
        """First unused deterministic self-play game index."""

        return self._next_game_index

    @property
    def _feature_shape(self) -> tuple[int, int, int]:
        return (2 * self._history_length + 1, self._board_size, self._board_size)

    @property
    def _action_size(self) -> int:
        return self._board_size * self._board_size + 1

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, sample: TrainingSample) -> None:
        """Validate, detach, and append one position, evicting the oldest if full."""

        stored = self._validated_copy(sample)
        self._append_validated((stored,))

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        """Append positions transactionally after validating the complete input."""

        try:
            candidates = tuple(samples)
        except TypeError as exc:
            raise ReplayError("samples must be an iterable of TrainingSample objects") from exc
        stored = tuple(self._validated_copy(sample) for sample in candidates)
        self._append_validated(stored)

    def add_game(self, game: SelfPlayGame) -> None:
        """Append all positions from one internally consistent completed game."""

        if not isinstance(game, SelfPlayGame):
            raise ReplayError("game must be a SelfPlayGame")
        game_index = _sample_game_index(game.game_index)
        if not game.samples:
            raise ReplayError("a replay game must contain at least one sample")
        if len(game.samples) != len(game.actions):
            raise ReplayError("game samples and actions must have equal lengths")
        for position, (sample, action) in enumerate(
            zip(game.samples, game.actions, strict=True)
        ):
            normalized_action = _nonnegative_int(action, f"game action {position}")
            if sample.game_index != game_index:
                raise ReplayError(f"sample {position} has an inconsistent game_index")
            if sample.selected_action != normalized_action:
                raise ReplayError(f"sample {position} does not match the recorded game action")
        self.extend(game.samples)

    def sample(self, batch_size: int, seed: int, augment: bool = False) -> ReplayBatch:
        """Sample positions without replacement, optionally applying seeded D4 transforms."""

        normalized_batch_size = _positive_int(batch_size, "batch_size")
        normalized_seed = _uint64(seed, "seed")
        if not isinstance(augment, bool):
            raise ReplayError("augment must be a boolean")
        if normalized_batch_size > len(self._samples):
            raise ReplayError(
                f"batch_size {normalized_batch_size} exceeds replay size {len(self._samples)}"
            )

        rng = np.random.default_rng(normalized_seed)
        indices = rng.choice(len(self._samples), size=normalized_batch_size, replace=False)
        selected = tuple(self._samples[int(index)] for index in indices)

        features: list[NDArray[np.float32]] = []
        policies: list[NDArray[np.float32]] = []
        selected_actions: list[int] = []
        if augment:
            operations = tuple(Symmetry)
            symmetry_indices = rng.integers(0, len(operations), size=normalized_batch_size)
        else:
            operations = (Symmetry.IDENTITY,)
            symmetry_indices = np.zeros(normalized_batch_size, dtype=np.int64)

        for item, symmetry_index in zip(selected, symmetry_indices, strict=True):
            symmetry = operations[int(symmetry_index)]
            features.append(symmetry.transform_features(item.features))
            policies.append(symmetry.transform_policy(item.policy, self._board_size))
            selected_actions.append(
                symmetry.transform_action(item.selected_action, self._board_size)
            )

        return ReplayBatch(
            features=np.stack(features).astype(np.float32, copy=False),
            policies=np.stack(policies).astype(np.float32, copy=False),
            values=np.asarray([item.value for item in selected], dtype=np.float32),
            to_play=np.asarray([int(item.to_play) for item in selected], dtype=np.uint8),
            move_numbers=np.asarray(
                [item.move_number for item in selected],
                dtype=_MOVE_DTYPE,
            ),
            selected_actions=np.asarray(selected_actions, dtype=_ACTION_DTYPE),
            game_indices=np.asarray(
                [item.game_index for item in selected],
                dtype=_GAME_INDEX_DTYPE,
            ),
        )

    def save(self, path: str | PathLike[str]) -> None:
        """Atomically write a compressed, pickle-free version-1 NPZ snapshot."""

        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ReplayError(f"could not create replay directory: {target.parent}") from exc

        payload = self._snapshot_payload()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(temporary, **payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, target)  # noqa: PTH105
        except Exception as exc:
            raise ReplayError(f"could not save replay snapshot: {target}") from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> ReplayBuffer:
        """Load and strictly validate a version-1 replay snapshot."""

        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as snapshot:
                files = tuple(snapshot.files)
                if len(files) != len(_SNAPSHOT_KEYS) or set(files) != _SNAPSHOT_KEYS:
                    missing = sorted(_SNAPSHOT_KEYS.difference(files))
                    extra = sorted(set(files).difference(_SNAPSHOT_KEYS))
                    details: list[str] = []
                    if missing:
                        details.append(f"missing keys: {', '.join(missing)}")
                    if extra:
                        details.append(f"unknown keys: {', '.join(extra)}")
                    raise ReplayError("invalid replay snapshot fields (" + "; ".join(details) + ")")

                version = _read_scalar(snapshot["version"], "version", _VERSION_DTYPE)
                if version != _FORMAT_VERSION:
                    raise ReplayError(f"unsupported replay format version: {version}")
                board_size = _read_scalar(snapshot["board_size"], "board_size", _SCALAR_DTYPE)
                history_length = _read_scalar(
                    snapshot["history_length"],
                    "history_length",
                    _SCALAR_DTYPE,
                )
                capacity = _read_scalar(snapshot["capacity"], "capacity", _SCALAR_DTYPE)
                next_game_index = _read_scalar(
                    snapshot["next_game_index"],
                    "next_game_index",
                    _SCALAR_DTYPE,
                )

                buffer = cls(board_size, history_length, capacity, next_game_index)
                arrays = {
                    name: np.array(snapshot[name], copy=True, order="C")
                    for name in _SNAPSHOT_KEYS
                    if name not in {
                        "version",
                        "board_size",
                        "history_length",
                        "capacity",
                        "next_game_index",
                    }
                }
        except ReplayError:
            raise
        except Exception as exc:
            raise ReplayError(f"could not load replay snapshot: {source}") from exc

        samples = buffer._samples_from_snapshot(arrays)
        if len(samples) > buffer.capacity:
            raise ReplayError("snapshot contains more positions than its declared capacity")
        if samples:
            required_next_index = max(sample.game_index for sample in samples) + 1
            if buffer.next_game_index < required_next_index:
                raise ReplayError("next_game_index does not follow stored game indices")
        buffer._samples.extend(samples)
        return buffer

    def _validated_copy(self, sample: TrainingSample) -> TrainingSample:
        if not isinstance(sample, TrainingSample):
            raise ReplayError("sample must be a TrainingSample")
        if sample.features.shape != self._feature_shape:
            raise ReplayError(f"sample features must have shape {self._feature_shape}")
        if sample.features.dtype != np.float32:
            raise ReplayError("sample features must have dtype float32")
        if not np.isfinite(sample.features).all():
            raise ReplayError("sample features must contain only finite values")
        if sample.policy.shape != (self._action_size,):
            raise ReplayError(f"sample policy must have shape ({self._action_size},)")
        if sample.policy.dtype != np.float32:
            raise ReplayError("sample policy must have dtype float32")
        if not np.isfinite(sample.policy).all():
            raise ReplayError("sample policy must contain only finite values")
        if np.any(sample.policy < 0.0):
            raise ReplayError("sample policy must be non-negative")
        if not np.isclose(
            float(sample.policy.sum(dtype=np.float64)),
            1.0,
            rtol=_POLICY_RTOL,
            atol=_POLICY_ATOL,
        ):
            raise ReplayError("sample policy must sum to one")

        value = _sample_value(sample.value)
        if not isinstance(sample.to_play, Color) or sample.to_play is Color.EMPTY:
            raise ReplayError("sample to_play must be BLACK or WHITE")
        to_play = sample.to_play
        move_number = _nonnegative_int(sample.move_number, "sample move_number")
        if move_number > np.iinfo(_MOVE_DTYPE).max:
            raise ReplayError("sample move_number is too large")
        selected_action = _nonnegative_int(sample.selected_action, "sample selected_action")
        if selected_action >= self._action_size:
            raise ReplayError(f"sample selected_action must be in [0, {self._action_size - 1}]")
        game_index = _sample_game_index(sample.game_index)

        try:
            return TrainingSample(
                features=np.array(sample.features, dtype=np.float32, copy=True, order="C"),
                policy=np.array(sample.policy, dtype=np.float32, copy=True, order="C"),
                value=value,
                to_play=to_play,
                move_number=move_number,
                selected_action=selected_action,
                game_index=game_index,
            )
        except ValueError as exc:
            raise ReplayError("sample failed replay validation") from exc

    def _append_validated(self, samples: tuple[TrainingSample, ...]) -> None:
        for sample in samples:
            if sample.game_index == _UINT64_MAX:
                raise ReplayError("sample game_index leaves no representable next_game_index")
        self._samples.extend(samples)
        if samples:
            self._next_game_index = max(
                self._next_game_index,
                max(sample.game_index for sample in samples) + 1,
            )

    def _snapshot_payload(self) -> dict[str, Any]:
        samples = tuple(self._samples)
        if samples:
            features = np.stack([sample.features for sample in samples])
            policies = np.stack([sample.policy for sample in samples])
        else:
            features = np.empty((0, *self._feature_shape), dtype=np.float32)
            policies = np.empty((0, self._action_size), dtype=np.float32)
        return {
            "version": np.asarray(_FORMAT_VERSION, dtype=_VERSION_DTYPE),
            "board_size": np.asarray(self._board_size, dtype=_SCALAR_DTYPE),
            "history_length": np.asarray(self._history_length, dtype=_SCALAR_DTYPE),
            "capacity": np.asarray(self._capacity, dtype=_SCALAR_DTYPE),
            "next_game_index": np.asarray(self._next_game_index, dtype=_SCALAR_DTYPE),
            "features": np.ascontiguousarray(features, dtype=np.float32),
            "policies": np.ascontiguousarray(policies, dtype=np.float32),
            "values": np.asarray([sample.value for sample in samples], dtype=np.float32),
            "to_play": np.asarray([int(sample.to_play) for sample in samples], dtype=_COLOR_DTYPE),
            "move_numbers": np.asarray(
                [sample.move_number for sample in samples],
                dtype=_MOVE_DTYPE,
            ),
            "selected_actions": np.asarray(
                [sample.selected_action for sample in samples],
                dtype=_ACTION_DTYPE,
            ),
            "game_indices": np.asarray(
                [sample.game_index for sample in samples],
                dtype=_GAME_INDEX_DTYPE,
            ),
        }

    def _samples_from_snapshot(
        self,
        arrays: dict[str, NDArray[np.generic]],
    ) -> tuple[TrainingSample, ...]:
        features = cast(
            "NDArray[np.float32]",
            _require_array(
                arrays["features"],
                "features",
                np.dtype(np.float32),
                (None, *self._feature_shape),
            ),
        )
        length = features.shape[0]
        policies = cast(
            "NDArray[np.float32]",
            _require_array(
                arrays["policies"],
                "policies",
                np.dtype(np.float32),
                (length, self._action_size),
            ),
        )
        values = cast(
            "NDArray[np.float32]",
            _require_array(
                arrays["values"],
                "values",
                np.dtype(np.float32),
                (length,),
            ),
        )
        to_play = cast(
            "NDArray[np.uint8]",
            _require_array(arrays["to_play"], "to_play", _COLOR_DTYPE, (length,)),
        )
        move_numbers = cast(
            "NDArray[np.int64]",
            _require_array(
                arrays["move_numbers"],
                "move_numbers",
                _MOVE_DTYPE,
                (length,),
            ),
        )
        selected_actions = cast(
            "NDArray[np.int64]",
            _require_array(
                arrays["selected_actions"],
                "selected_actions",
                _ACTION_DTYPE,
                (length,),
            ),
        )
        game_indices = cast(
            "NDArray[np.uint64]",
            _require_array(
                arrays["game_indices"],
                "game_indices",
                _GAME_INDEX_DTYPE,
                (length,),
            ),
        )

        if not np.isfinite(features).all():
            raise ReplayError("snapshot features must contain only finite values")
        if not np.isfinite(policies).all():
            raise ReplayError("snapshot policies must contain only finite values")
        if np.any(policies < 0.0):
            raise ReplayError("snapshot policies must be non-negative")
        if length and not np.allclose(
            np.sum(policies, axis=1, dtype=np.float64),
            1.0,
            rtol=_POLICY_RTOL,
            atol=_POLICY_ATOL,
        ):
            raise ReplayError("every snapshot policy must sum to one")
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0):
            raise ReplayError("snapshot values must be finite and within [-1, 1]")
        if np.any((to_play != int(Color.BLACK)) & (to_play != int(Color.WHITE))):
            raise ReplayError("snapshot to_play values must be BLACK or WHITE")
        if np.any(move_numbers < 0):
            raise ReplayError("snapshot move_numbers must be non-negative")
        if np.any((selected_actions < 0) | (selected_actions >= self._action_size)):
            raise ReplayError("snapshot selected_actions are outside the action space")

        result: list[TrainingSample] = []
        for index in range(length):
            try:
                result.append(
                    TrainingSample(
                        features=features[index],
                        policy=policies[index],
                        value=float(values[index]),
                        to_play=Color(int(to_play[index])),
                        move_number=int(move_numbers[index]),
                        selected_action=int(selected_actions[index]),
                        game_index=int(game_indices[index]),
                    )
                )
            except ValueError as exc:
                raise ReplayError(f"invalid snapshot sample at index {index}") from exc
        return tuple(result)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReplayError(f"{name} must be a non-negative integer")
    return value


def _uint64(value: object, name: str) -> int:
    normalized = _nonnegative_int(value, name)
    if normalized > _UINT64_MAX:
        raise ReplayError(f"{name} must fit in an unsigned 64-bit integer")
    return normalized


def _sample_game_index(value: object) -> int:
    return _uint64(value, "sample game_index")


def _sample_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ReplayError("sample value must be a finite number within [-1, 1]")
    normalized = float(value)
    if not isfinite(normalized) or not -1.0 <= normalized <= 1.0:
        raise ReplayError("sample value must be a finite number within [-1, 1]")
    return normalized


def _read_scalar(array: NDArray[np.generic], name: str, dtype: np.dtype[np.generic]) -> int:
    if not isinstance(array, np.ndarray) or array.shape != () or array.dtype != dtype:
        raise ReplayError(f"snapshot {name} must be a scalar with dtype {dtype.name}")
    return int(array.item())


def _require_array(
    array: NDArray[np.generic],
    name: str,
    dtype: np.dtype[np.generic],
    shape: tuple[int | None, ...],
) -> NDArray[np.generic]:
    if not isinstance(array, np.ndarray) or array.dtype != dtype:
        raise ReplayError(f"snapshot {name} must have dtype {dtype.name}")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        rendered = tuple("*" if item is None else item for item in shape)
        raise ReplayError(f"snapshot {name} must have shape {rendered}")
    return array
