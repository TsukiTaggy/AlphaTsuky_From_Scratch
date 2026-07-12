"""Safe, atomic checkpoints for policy-value network training."""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from pydantic import ValidationError

from .config import AppConfig
from .network import PolicyValueNetwork

_FORMAT_VERSION = 1
_CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "model_state",
        "optimizer_state",
        "step",
        "torch_rng_state",
        "config",
        "compatibility",
    }
)
_OPTIMIZER_STATE_KEYS = frozenset({"state", "param_groups"})


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be saved or safely restored."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Non-tensor metadata recovered from a checkpoint."""

    step: int
    config: dict[str, object]


def save_checkpoint(
    path: str | Path,
    *,
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: AppConfig,
) -> None:
    """Atomically save model, optimizer, RNG, and configuration state."""

    try:
        destination = _coerce_path(path)
        _validate_network_config(network, config)
        training_optimizer = _validate_training_optimizer(optimizer, network, config)
        validated_step = _validate_step(step)

        primitive_config = _config_dump(config)
        model_state = network.state_dict()
        _validate_model_state(model_state, network)
        optimizer_state = training_optimizer.state_dict()
        _validate_optimizer_state(
            optimizer_state,
            config=config,
            expected_parameter_count=len(tuple(network.parameters())),
            optimizer=training_optimizer,
        )
        rng_state = torch.get_rng_state().clone()
        _validate_rng_state(rng_state)
        bundle: dict[str, object] = {
            "format_version": _FORMAT_VERSION,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "step": validated_step,
            "torch_rng_state": rng_state,
            "config": primitive_config,
            "compatibility": _compatibility(config),
        }
        _atomic_torch_save(destination, bundle)
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"could not save checkpoint: {exc}") from exc


def load_checkpoint(
    path: str | Path,
    *,
    network: PolicyValueNetwork,
    config: AppConfig,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool | None = None,
) -> CheckpointMetadata:
    """Safely load a compatible checkpoint without partial in-memory updates."""

    original_rng = torch.get_rng_state().clone()
    should_restore_rng = optimizer is not None if restore_rng is None else restore_rng
    if type(should_restore_rng) is not bool:
        raise CheckpointError("restore_rng must be a boolean or None")

    try:
        source = _coerce_path(path)
        _validate_network_config(network, config)
        training_optimizer = (
            _validate_training_optimizer(optimizer, network, config)
            if optimizer is not None
            else None
        )
        raw = torch.load(source, map_location="cpu", weights_only=True)
        payload = _validate_payload(
            raw,
            network=network,
            config=config,
            optimizer=training_optimizer,
        )

        model_state = cast("Mapping[str, torch.Tensor]", payload["model_state"])
        optimizer_state = cast("dict[str, object]", payload["optimizer_state"])
        checkpoint_rng = cast("torch.Tensor", payload["torch_rng_state"])
        step = cast("int", payload["step"])
        saved_config = cast("dict[str, object]", payload["config"])

        original_model_state = _clone_model_state(network.state_dict())
        original_optimizer_state = (
            copy.deepcopy(training_optimizer.state_dict())
            if training_optimizer is not None
            else None
        )
        try:
            network.load_state_dict(model_state, strict=True)
            if training_optimizer is not None:
                training_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
            if should_restore_rng:
                torch.set_rng_state(checkpoint_rng.clone())
            else:
                torch.set_rng_state(original_rng)
        except Exception as exc:
            network.load_state_dict(original_model_state, strict=True)
            if training_optimizer is not None and original_optimizer_state is not None:
                training_optimizer.load_state_dict(original_optimizer_state)
            torch.set_rng_state(original_rng)
            raise CheckpointError(f"could not apply checkpoint state: {exc}") from exc

        return CheckpointMetadata(step=step, config=copy.deepcopy(saved_config))
    except CheckpointError:
        torch.set_rng_state(original_rng)
        raise
    except Exception as exc:
        torch.set_rng_state(original_rng)
        raise CheckpointError(f"could not load checkpoint: {exc}") from exc


def _coerce_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)):
        raise CheckpointError("checkpoint path must be a string or pathlib.Path")
    return Path(path).expanduser()


def _validate_step(step: int) -> int:
    if type(step) is not int or step < 0:
        raise CheckpointError("checkpoint step must be a nonnegative integer")
    return step


def _config_dump(config: AppConfig) -> dict[str, object]:
    if not isinstance(config, AppConfig):
        raise CheckpointError("config must be an AppConfig")
    dumped = config.model_dump(mode="json")
    _validate_primitive(dumped, "config")
    return copy.deepcopy(cast("dict[str, object]", dumped))


def _compatibility(config: AppConfig) -> dict[str, object]:
    compatibility: dict[str, object] = {
        "game": {"board_size": config.game.board_size},
        "model": config.model.model_dump(mode="json"),
        "learner": config.learner.model_dump(mode="json"),
    }
    _validate_primitive(compatibility, "compatibility")
    return copy.deepcopy(compatibility)


def _validate_network_config(network: PolicyValueNetwork, config: AppConfig) -> None:
    if not isinstance(network, PolicyValueNetwork):
        raise CheckpointError("network must be a PolicyValueNetwork")
    _config_dump(config)

    expected = {
        "board_size": config.game.board_size,
        "history_length": config.model.history_length,
        "channels": config.model.channels,
        "residual_blocks": config.model.residual_blocks,
        "value_hidden_size": config.model.value_hidden_size,
    }
    for name, expected_value in expected.items():
        actual_value = getattr(network, name)
        if actual_value != expected_value:
            raise CheckpointError(
                f"network {name}={actual_value} does not match config value {expected_value}"
            )


def _validate_training_optimizer(
    optimizer: object,
    network: PolicyValueNetwork,
    config: AppConfig,
) -> torch.optim.SGD:
    if type(optimizer) is not torch.optim.SGD:
        raise CheckpointError("training optimizer must be exactly torch.optim.SGD")

    training_optimizer = optimizer
    expected_parameters = tuple(network.parameters())
    expected_identifiers = {id(parameter) for parameter in expected_parameters}
    optimizer_parameters: list[torch.Tensor] = []

    for group_index, group in enumerate(training_optimizer.param_groups):
        parameters = group.get("params")
        if type(parameters) is not list or not all(
            isinstance(parameter, torch.Tensor) for parameter in parameters
        ):
            raise CheckpointError(
                f"optimizer parameter group {group_index} params must be tensors"
            )
        optimizer_parameters.extend(cast("list[torch.Tensor]", parameters))
        _validate_sgd_hyperparameters(group, group_index, config)

    optimizer_identifiers = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_identifiers) != len(set(optimizer_identifiers)):
        raise CheckpointError("optimizer parameters contain duplicate tensor identities")
    if set(optimizer_identifiers) != expected_identifiers:
        raise CheckpointError("optimizer must own exactly the supplied network parameters")
    if len(optimizer_parameters) != len(expected_parameters):
        raise CheckpointError("optimizer parameter count does not match the network")
    if any(
        actual is not expected
        for actual, expected in zip(
            optimizer_parameters,
            expected_parameters,
            strict=True,
        )
    ):
        raise CheckpointError("optimizer parameter order does not match the network")

    for state_parameter in training_optimizer.state:
        if (
            not isinstance(state_parameter, torch.Tensor)
            or id(state_parameter) not in expected_identifiers
        ):
            raise CheckpointError("optimizer state contains a parameter outside the network")

    return training_optimizer


def _atomic_torch_save(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)  # noqa: PTH105 - required atomic primitive
        temporary = None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _validate_payload(
    raw: object,
    *,
    network: PolicyValueNetwork,
    config: AppConfig,
    optimizer: torch.optim.SGD | None,
) -> dict[str, object]:
    if type(raw) is not dict:
        raise CheckpointError("checkpoint root must be a dictionary")
    payload = cast("dict[object, object]", raw)
    if set(payload) != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - set(payload))
        extra = sorted(str(key) for key in set(payload) - _CHECKPOINT_KEYS)
        raise CheckpointError(
            f"checkpoint keys do not match format (missing={missing}, extra={extra})"
        )
    typed_payload = cast("dict[str, object]", payload)

    version = typed_payload["format_version"]
    if type(version) is not int or version != _FORMAT_VERSION:
        raise CheckpointError(f"unsupported checkpoint format version: {version!r}")
    _validate_step(cast("int", typed_payload["step"]))
    _validate_model_state(typed_payload["model_state"], network)
    _validate_rng_state(typed_payload["torch_rng_state"])

    saved_config = typed_payload["config"]
    if type(saved_config) is not dict:
        raise CheckpointError("checkpoint config must be a dictionary")
    _validate_primitive(saved_config, "config")
    typed_saved_config = cast("dict[str, object]", saved_config)
    config_for_validation = typed_saved_config
    if "arena" not in typed_saved_config:
        config_for_validation = copy.deepcopy(typed_saved_config)
        config_for_validation["arena"] = config.arena.model_dump(mode="json")
    try:
        validated_saved_config = AppConfig.model_validate(config_for_validation)
    except ValidationError as exc:
        raise CheckpointError(f"checkpoint config is invalid: {exc}") from exc

    saved_compatibility = typed_payload["compatibility"]
    if type(saved_compatibility) is not dict:
        raise CheckpointError("checkpoint compatibility must be a dictionary")
    _validate_primitive(saved_compatibility, "compatibility")
    if saved_compatibility != _compatibility(validated_saved_config):
        raise CheckpointError("checkpoint compatibility does not match checkpoint config")
    expected_compatibility = _compatibility(config)
    if saved_compatibility != expected_compatibility:
        raise CheckpointError("checkpoint is incompatible with the current configuration")

    _validate_optimizer_state(
        typed_payload["optimizer_state"],
        config=validated_saved_config,
        expected_parameter_count=len(tuple(network.parameters())),
        optimizer=optimizer,
    )

    return typed_payload


def _validate_model_state(raw: object, network: PolicyValueNetwork) -> None:
    if not isinstance(raw, Mapping):
        raise CheckpointError("model_state must be a mapping")
    if not all(type(key) is str for key in raw):
        raise CheckpointError("model_state keys must be strings")

    loaded = cast("Mapping[str, object]", raw)
    current = network.state_dict()
    if set(loaded) != set(current):
        raise CheckpointError("model_state parameter keys do not match the network")
    for name, expected in current.items():
        value = loaded[name]
        if not isinstance(value, torch.Tensor):
            raise CheckpointError(f"model_state[{name!r}] must be a tensor")
        if value.shape != expected.shape:
            raise CheckpointError(
                f"model_state[{name!r}] has shape {tuple(value.shape)}, "
                f"expected {tuple(expected.shape)}"
            )
        if value.dtype != expected.dtype:
            raise CheckpointError(
                f"model_state[{name!r}] has dtype {value.dtype}, expected {expected.dtype}"
            )
        if value.layout != expected.layout:
            raise CheckpointError(
                f"model_state[{name!r}] has layout {value.layout}, expected {expected.layout}"
            )
        _validate_finite_tensor(value, f"model_state[{name!r}]")


def _validate_optimizer_state(
    raw: object,
    *,
    config: AppConfig,
    expected_parameter_count: int,
    optimizer: torch.optim.SGD | None,
) -> None:
    if type(raw) is not dict:
        raise CheckpointError("optimizer_state must be a dictionary")
    optimizer_state = cast("dict[object, object]", raw)
    if set(optimizer_state) != _OPTIMIZER_STATE_KEYS:
        raise CheckpointError("optimizer_state must contain exactly state and param_groups")

    state = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    if type(state) is not dict:
        raise CheckpointError("optimizer_state.state must be a dictionary")
    if not all(type(key) is int and key >= 0 for key in cast("dict[object, object]", state)):
        raise CheckpointError("optimizer state parameter identifiers must be nonnegative integers")
    if type(groups) is not list:
        raise CheckpointError("optimizer_state.param_groups must be a list")
    parameter_identifiers: list[int] = []
    for index, group in enumerate(cast("list[object]", groups)):
        if type(group) is not dict:
            raise CheckpointError(f"optimizer parameter group {index} must be a dictionary")
        typed_group = cast("dict[object, object]", group)
        if not all(type(key) is str for key in typed_group):
            raise CheckpointError(f"optimizer parameter group {index} keys must be strings")
        string_group = cast("dict[str, object]", group)
        parameters = string_group.get("params")
        if type(parameters) is not list or not all(
            type(parameter) is int and parameter >= 0
            for parameter in cast("list[object]", parameters)
        ):
            raise CheckpointError(
                f"optimizer parameter group {index} params must be nonnegative integers"
            )
        parameter_identifiers.extend(cast("list[int]", parameters))
        _validate_sgd_hyperparameters(string_group, index, config)

    if len(parameter_identifiers) != len(set(parameter_identifiers)):
        raise CheckpointError("optimizer parameter identifiers must be unique")
    if set(parameter_identifiers) != set(range(expected_parameter_count)):
        raise CheckpointError("optimizer parameter identifiers do not match the network")

    typed_state = cast("dict[int, object]", state)
    if not set(typed_state).issubset(parameter_identifiers):
        raise CheckpointError("optimizer state references an unknown parameter identifier")
    _validate_sgd_parameter_state(typed_state, config)
    _validate_safe_state(optimizer_state, "optimizer_state")

    if optimizer is not None:
        _validate_loaded_optimizer_structure(
            cast("dict[str, object]", optimizer_state),
            optimizer,
        )


def _validate_sgd_hyperparameters(
    group: Mapping[str, object],
    group_index: int,
    config: AppConfig,
) -> None:
    expected = {
        "lr": config.learner.learning_rate,
        "momentum": config.learner.momentum,
        "weight_decay": config.learner.weight_decay,
    }
    for name, expected_value in expected.items():
        value = group.get(name)
        if type(value) is not float or value != expected_value:
            raise CheckpointError(
                f"optimizer parameter group {group_index} {name}={value!r} "
                f"does not match config value {expected_value}"
            )


def _validate_sgd_parameter_state(
    state: dict[int, object],
    config: AppConfig,
) -> None:
    if config.learner.momentum == 0.0 and state:
        raise CheckpointError("SGD without momentum must not contain parameter state")
    for identifier, value in state.items():
        if type(value) is not dict:
            raise CheckpointError(
                f"optimizer state for parameter {identifier} must be a dictionary"
            )
        parameter_state = cast("dict[object, object]", value)
        if set(parameter_state) != {"momentum_buffer"}:
            raise CheckpointError(
                f"optimizer state for parameter {identifier} is not valid SGD momentum state"
            )
        if not isinstance(parameter_state["momentum_buffer"], torch.Tensor):
            raise CheckpointError(
                f"optimizer momentum buffer for parameter {identifier} must be a tensor"
            )


def _validate_loaded_optimizer_structure(
    loaded: dict[str, object],
    optimizer: torch.optim.SGD,
) -> None:
    destination = optimizer.state_dict()
    loaded_groups = cast("list[dict[str, object]]", loaded["param_groups"])
    destination_groups = cast("list[dict[str, object]]", destination["param_groups"])
    if len(loaded_groups) != len(destination_groups):
        raise CheckpointError("checkpoint optimizer parameter group count does not match")

    parameter_by_identifier: dict[int, torch.Tensor] = {}
    for index, (loaded_group, destination_group, live_group) in enumerate(
        zip(loaded_groups, destination_groups, optimizer.param_groups, strict=True)
    ):
        if set(loaded_group) != set(destination_group):
            raise CheckpointError(
                f"checkpoint optimizer parameter group {index} structure does not match"
            )
        loaded_parameters = cast("list[int]", loaded_group["params"])
        destination_parameters = cast("list[int]", destination_group["params"])
        if loaded_parameters != destination_parameters:
            raise CheckpointError(
                f"checkpoint optimizer parameter group {index} identifiers do not match"
            )
        for key in set(loaded_group) - {"params"}:
            if not _state_values_equal(loaded_group[key], destination_group[key]):
                raise CheckpointError(
                    f"checkpoint optimizer parameter group {index} option {key!r} does not match"
                )

        live_parameters = cast("list[torch.Tensor]", live_group["params"])
        parameter_by_identifier.update(
            zip(destination_parameters, live_parameters, strict=True)
        )

    loaded_state = cast("dict[int, dict[str, object]]", loaded["state"])
    for identifier, parameter_state in loaded_state.items():
        momentum_buffer = cast("torch.Tensor", parameter_state["momentum_buffer"])
        parameter = parameter_by_identifier[identifier]
        if momentum_buffer.shape != parameter.shape:
            raise CheckpointError(
                f"optimizer momentum buffer for parameter {identifier} has the wrong shape"
            )
        if momentum_buffer.dtype != parameter.dtype:
            raise CheckpointError(
                f"optimizer momentum buffer for parameter {identifier} has the wrong dtype"
            )
        if momentum_buffer.layout != parameter.layout:
            raise CheckpointError(
                f"optimizer momentum buffer for parameter {identifier} has the wrong layout"
            )


def _state_values_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if type(left) in {list, tuple}:
        left_items = cast("list[object] | tuple[object, ...]", left)
        right_items = cast("list[object] | tuple[object, ...]", right)
        return len(left_items) == len(right_items) and all(
            _state_values_equal(left_item, right_item)
            for left_item, right_item in zip(left_items, right_items, strict=True)
        )
    if type(left) is dict:
        left_mapping = cast("dict[object, object]", left)
        right_mapping = cast("dict[object, object]", right)
        return set(left_mapping) == set(right_mapping) and all(
            _state_values_equal(value, right_mapping[key])
            for key, value in left_mapping.items()
        )
    return bool(left == right)


def _validate_rng_state(raw: object) -> None:
    if not isinstance(raw, torch.Tensor):
        raise CheckpointError("torch_rng_state must be a tensor")
    expected = torch.get_rng_state()
    if raw.dtype != torch.uint8 or raw.device.type != "cpu" or raw.shape != expected.shape:
        raise CheckpointError("torch_rng_state is not a valid CPU RNG state tensor")


def _validate_primitive(value: object, location: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise CheckpointError(f"{location} contains a non-finite floating-point value")
    if value is None or type(value) in {bool, int, float, str}:
        return
    if type(value) is list:
        for index, item in enumerate(cast("list[object]", value)):
            _validate_primitive(item, f"{location}[{index}]")
        return
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        if not all(type(key) is str for key in mapping):
            raise CheckpointError(f"{location} keys must be strings")
        for key, item in mapping.items():
            _validate_primitive(item, f"{location}.{key}")
        return
    raise CheckpointError(f"{location} contains unsupported value type {type(value).__name__}")


def _validate_safe_state(value: object, location: str) -> None:
    if isinstance(value, torch.Tensor):
        _validate_finite_tensor(value, location)
        return
    if value is None:
        return
    if type(value) is float and not math.isfinite(value):
        raise CheckpointError(f"{location} contains a non-finite floating-point value")
    if type(value) in {bool, int, float, str}:
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(cast("list[object] | tuple[object, ...]", value)):
            _validate_safe_state(item, f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast("dict[object, object]", value).items():
            if type(key) not in {int, str}:
                raise CheckpointError(f"{location} contains an unsupported key type")
            _validate_safe_state(item, f"{location}.{key}")
        return
    raise CheckpointError(f"{location} contains unsupported value type {type(value).__name__}")


def _clone_model_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def _validate_finite_tensor(value: torch.Tensor, location: str) -> None:
    if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
        raise CheckpointError(f"{location} contains non-finite tensor values")
