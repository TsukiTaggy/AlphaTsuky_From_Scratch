"""Tests for safe, atomic policy-value network checkpoints."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from azgo.checkpoint import (
    CheckpointError,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from azgo.config import AppConfig, load_config
from azgo.network import PolicyValueNetwork

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class _UnsafeValue:
    pass


@pytest.fixture
def config() -> AppConfig:
    raw = load_config(Path("configs/engine/go5.yaml")).model_dump(mode="json")
    raw["model"] = {
        "history_length": 2,
        "channels": 4,
        "residual_blocks": 1,
        "value_hidden_size": 4,
    }
    raw["learner"] = {
        "seed": 5301,
        "batch_size": 2,
        "steps": 3,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "value_loss_weight": 1.0,
        "gradient_clip_norm": 5.0,
        "checkpoint_interval": 1,
        "augment": True,
    }
    return AppConfig.model_validate(raw)


def _network(config: AppConfig) -> PolicyValueNetwork:
    return PolicyValueNetwork(
        board_size=config.game.board_size,
        history_length=config.model.history_length,
        channels=config.model.channels,
        residual_blocks=config.model.residual_blocks,
        value_hidden_size=config.model.value_hidden_size,
    )


def _optimizer(
    network: PolicyValueNetwork,
    config: AppConfig,
) -> torch.optim.SGD:
    return torch.optim.SGD(
        network.parameters(),
        lr=config.learner.learning_rate,
        momentum=config.learner.momentum,
        weight_decay=config.learner.weight_decay,
    )


def _train_once(
    network: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
) -> None:
    network.train()
    inputs = torch.randn(2, network.input_channels, network.board_size, network.board_size)
    policy, value = network(inputs)
    loss = policy.square().mean() + value.square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _clone_state(network: PolicyValueNetwork) -> dict[str, torch.Tensor]:
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


def _read_payload(path: Path) -> dict[str, object]:
    raw: object = torch.load(path, map_location="cpu", weights_only=True)
    assert type(raw) is dict
    return cast("dict[str, object]", raw)


def _rewrite_payload(
    path: Path,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    payload = _read_payload(path)
    mutate(payload)
    torch.save(payload, path)


def _saved_checkpoint(
    path: Path,
    config: AppConfig,
) -> tuple[PolicyValueNetwork, torch.optim.SGD]:
    network = _network(config)
    optimizer = _optimizer(network, config)
    _train_once(network, optimizer)
    save_checkpoint(path, network=network, optimizer=optimizer, step=7, config=config)
    return network, optimizer


def _modified_config(config: AppConfig, section: str, **changes: object) -> AppConfig:
    raw = config.model_dump(mode="json")
    selected = cast("dict[str, object]", raw[section])
    selected.update(changes)
    return AppConfig.model_validate(raw)


def test_round_trip_restores_model_optimizer_step_bn_and_rng(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    torch.manual_seed(1729)
    path = tmp_path / "nested" / "checkpoint.pt"
    source, source_optimizer = _saved_checkpoint(path, config)
    expected_model = _clone_state(source)
    expected_optimizer = copy.deepcopy(source_optimizer.state_dict())
    expected_random = torch.rand(8)

    torch.manual_seed(999)
    restored = _network(config)
    restored_optimizer = _optimizer(restored, config)
    metadata = load_checkpoint(
        path,
        network=restored,
        optimizer=restored_optimizer,
        config=config,
    )

    assert metadata == CheckpointMetadata(step=7, config=config.model_dump(mode="json"))
    _assert_model_state(restored, expected_model)
    _assert_nested_equal(restored_optimizer.state_dict(), expected_optimizer)
    torch.testing.assert_close(torch.rand(8), expected_random, rtol=0.0, atol=0.0)
    assert path.is_file()
    assert any("running_mean" in key for key in expected_model)
    assert expected_optimizer["state"]


def test_inference_load_does_not_change_rng(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    source, _ = _saved_checkpoint(path, config)
    target = _network(config)
    torch.manual_seed(31013)
    expected_rng = torch.get_rng_state().clone()

    metadata = load_checkpoint(path, network=target, config=config)

    assert metadata.step == 7
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0.0, atol=0.0)
    _assert_model_state(target, source.state_dict())


def test_explicit_rng_restore_without_optimizer_is_supported(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    torch.manual_seed(4141)
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    expected = torch.rand(3)
    torch.manual_seed(5151)

    load_checkpoint(path, network=_network(config), config=config, restore_rng=True)

    torch.testing.assert_close(torch.rand(3), expected, rtol=0.0, atol=0.0)


def test_metadata_config_is_detached_and_metadata_is_frozen(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    metadata = load_checkpoint(path, network=_network(config), config=config)
    metadata.config["changed"] = True

    fresh = load_checkpoint(path, network=_network(config), config=config)

    assert "changed" not in fresh.config
    with pytest.raises(FrozenInstanceError):
        metadata.step = 9  # type: ignore[misc]


def test_payload_has_exact_versioned_contract(tmp_path: Path, config: AppConfig) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    payload = _read_payload(path)

    assert set(payload) == {
        "format_version",
        "model_state",
        "optimizer_state",
        "step",
        "torch_rng_state",
        "config",
        "compatibility",
    }
    assert payload["format_version"] == 1
    assert payload["step"] == 7
    assert payload["config"] == config.model_dump(mode="json")
    assert payload["compatibility"] == {
        "game": {"board_size": config.game.board_size},
        "model": config.model.model_dump(mode="json"),
        "learner": config.learner.model_dump(mode="json"),
    }


def test_load_rejects_config_that_fails_strict_app_validation(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        saved_config = cast("dict[str, object]", payload["config"])
        learner = cast("dict[str, object]", saved_config["learner"])
        learner["learning_rate"] = "0.01"

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="config is invalid"):
        load_checkpoint(path, network=_network(config), config=config)


@pytest.mark.parametrize("tamper", ["config", "compatibility"])
def test_load_rejects_internally_inconsistent_compatibility_metadata(
    tmp_path: Path,
    config: AppConfig,
    tamper: str,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        root = cast("dict[str, object]", payload[tamper])
        learner = cast("dict[str, object]", root["learner"])
        learner["augment"] = False

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="does not match checkpoint config"):
        load_checkpoint(path, network=_network(config), config=config)


def test_load_rejects_self_consistent_saved_metadata_incompatible_with_current_config(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        for field in ("config", "compatibility"):
            root = cast("dict[str, object]", payload[field])
            learner = cast("dict[str, object]", root["learner"])
            learner["augment"] = False

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="incompatible"):
        load_checkpoint(path, network=_network(config), config=config)


def test_noncompatibility_config_changes_are_allowed(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    source, _ = _saved_checkpoint(path, config)
    changed = _modified_config(config, "search", simulations=17)
    target = _network(changed)

    metadata = load_checkpoint(path, network=target, config=changed)

    assert metadata.config == config.model_dump(mode="json")
    _assert_model_state(target, source.state_dict())


@pytest.mark.parametrize(
    ("section", "changes"),
    [
        ("learner", {"learning_rate": 0.02}),
        ("learner", {"augment": False}),
    ],
)
def test_load_rejects_changed_learner_compatibility(
    tmp_path: Path,
    config: AppConfig,
    section: str,
    changes: dict[str, object],
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    changed = _modified_config(config, section, **changes)
    target = _network(changed)
    before = _clone_state(target)

    with pytest.raises(CheckpointError, match="incompatible"):
        load_checkpoint(path, network=target, config=changed)

    _assert_model_state(target, before)


@pytest.mark.parametrize(
    ("section", "changes"),
    [
        ("model", {"channels": 6}),
        ("model", {"history_length": 3}),
        ("game", {"board_size": 9}),
    ],
)
def test_load_rejects_changed_model_or_board_without_mutation(
    tmp_path: Path,
    config: AppConfig,
    section: str,
    changes: dict[str, object],
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    changed = _modified_config(config, section, **changes)
    target = _network(changed)
    before = _clone_state(target)

    with pytest.raises(CheckpointError):
        load_checkpoint(path, network=target, config=changed)

    _assert_model_state(target, before)


@pytest.mark.parametrize(
    "attribute",
    ["board_size", "history_length", "channels", "residual_blocks", "value_hidden_size"],
)
def test_save_rejects_network_that_does_not_match_config(
    tmp_path: Path,
    config: AppConfig,
    attribute: str,
) -> None:
    changes = {attribute: getattr(_network(config), attribute) + 1}
    if attribute == "board_size":
        changes[attribute] = 9
    arguments = {
        "board_size": config.game.board_size,
        "history_length": config.model.history_length,
        "channels": config.model.channels,
        "residual_blocks": config.model.residual_blocks,
        "value_hidden_size": config.model.value_hidden_size,
        **changes,
    }
    network = PolicyValueNetwork(**arguments)

    with pytest.raises(CheckpointError, match=attribute):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=_optimizer(network, config),
            step=0,
            config=config,
        )


def test_save_rejects_non_sgd_optimizer(tmp_path: Path, config: AppConfig) -> None:
    network = _network(config)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learner.learning_rate,
        weight_decay=config.learner.weight_decay,
    )

    with pytest.raises(CheckpointError, match=r"exactly torch\.optim\.SGD"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=optimizer,
            step=0,
            config=config,
        )


def test_load_rejects_non_sgd_optimizer_without_mutation(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    target = _network(config)
    target_optimizer = torch.optim.Adam(
        target.parameters(),
        lr=config.learner.learning_rate,
    )
    before = _clone_state(target)

    with pytest.raises(CheckpointError, match=r"exactly torch\.optim\.SGD"):
        load_checkpoint(
            path,
            network=target,
            optimizer=target_optimizer,
            config=config,
        )

    _assert_model_state(target, before)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("lr", 0.02),
        ("momentum", 0.5),
        ("weight_decay", 0.0),
    ],
)
def test_save_rejects_sgd_hyperparameters_that_do_not_match_config(
    tmp_path: Path,
    config: AppConfig,
    name: str,
    value: float,
) -> None:
    network = _network(config)
    optimizer = _optimizer(network, config)
    optimizer.param_groups[0][name] = value

    with pytest.raises(CheckpointError, match=name):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=optimizer,
            step=0,
            config=config,
        )


def test_load_rejects_destination_sgd_hyperparameters_without_mutation(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    target = _network(config)
    target_optimizer = _optimizer(target, config)
    target_optimizer.param_groups[0]["lr"] = 0.02
    before = _clone_state(target)

    with pytest.raises(CheckpointError, match="lr"):
        load_checkpoint(
            path,
            network=target,
            optimizer=target_optimizer,
            config=config,
        )

    _assert_model_state(target, before)


@pytest.mark.parametrize("damage", ["missing", "extra", "duplicate", "reordered"])
def test_save_rejects_optimizer_parameter_ownership_damage(
    tmp_path: Path,
    config: AppConfig,
    damage: str,
) -> None:
    network = _network(config)
    parameters = list(network.parameters())
    if damage == "missing":
        parameters.pop()
    elif damage == "extra":
        parameters.append(torch.nn.Parameter(torch.zeros(())))
    elif damage == "reordered":
        parameters[0], parameters[1] = parameters[1], parameters[0]

    optimizer = torch.optim.SGD(
        parameters,
        lr=config.learner.learning_rate,
        momentum=config.learner.momentum,
        weight_decay=config.learner.weight_decay,
    )
    if damage == "duplicate":
        optimizer.param_groups[0]["params"].append(parameters[0])

    with pytest.raises(CheckpointError, match="optimizer"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=optimizer,
            step=0,
            config=config,
        )


def test_save_rejects_optimizer_state_for_parameter_outside_network(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    network = _network(config)
    optimizer = _optimizer(network, config)
    extra = torch.nn.Parameter(torch.zeros(()))
    optimizer.state[extra] = {"momentum_buffer": torch.zeros_like(extra)}

    with pytest.raises(CheckpointError, match="state contains a parameter outside"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=optimizer,
            step=0,
            config=config,
        )


@pytest.mark.parametrize("step", [True, -1, 1.5, "1"])
def test_save_rejects_invalid_step(tmp_path: Path, config: AppConfig, step: object) -> None:
    network = _network(config)
    with pytest.raises(CheckpointError, match="step"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            network=network,
            optimizer=_optimizer(network, config),
            step=step,  # type: ignore[arg-type]
            config=config,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("step"),
        lambda payload: payload.update({"extra": 1}),
        lambda payload: payload.update({"format_version": 2}),
        lambda payload: payload.update({"format_version": True}),
        lambda payload: payload.update({"step": True}),
        lambda payload: payload.update({"optimizer_state": []}),
        lambda payload: payload.update({"torch_rng_state": "bad"}),
    ],
    ids=[
        "missing-key",
        "extra-key",
        "wrong-version",
        "boolean-version",
        "boolean-step",
        "optimizer-type",
        "rng-type",
    ],
)
def test_load_rejects_invalid_root_contract(
    tmp_path: Path,
    config: AppConfig,
    mutate: Callable[[dict[str, object]], object],
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError):
        load_checkpoint(path, network=_network(config), config=config)


def test_load_rejects_non_dictionary_root(tmp_path: Path, config: AppConfig) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save([], path)

    with pytest.raises(CheckpointError, match="root"):
        load_checkpoint(path, network=_network(config), config=config)


def test_load_rejects_corrupt_or_missing_file(tmp_path: Path, config: AppConfig) -> None:
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")

    for path in [corrupt, tmp_path / "missing.pt"]:
        with pytest.raises(CheckpointError, match="could not load"):
            load_checkpoint(path, network=_network(config), config=config)


def test_weights_only_load_rejects_unsafe_custom_object(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    payload = _read_payload(path)
    payload["config"] = {"unsafe": _UnsafeValue()}
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="could not load"):
        load_checkpoint(path, network=_network(config), config=config)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "type", "nonfinite"])
def test_load_rejects_invalid_model_tensor_without_mutation(
    tmp_path: Path,
    config: AppConfig,
    invalid: str,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        model_state = cast("dict[str, object]", payload["model_state"])
        key = next(
            name
            for name, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
        )
        tensor = cast("torch.Tensor", model_state[key])
        if invalid == "shape":
            model_state[key] = tensor.reshape(-1)[:1]
        elif invalid == "dtype":
            model_state[key] = tensor.to(torch.float64)
        elif invalid == "type":
            model_state[key] = "not a tensor"
        else:
            replacement = tensor.clone()
            replacement.reshape(-1)[0] = torch.nan
            model_state[key] = replacement

    _rewrite_payload(path, mutate)
    target = _network(config)
    before = _clone_state(target)

    with pytest.raises(CheckpointError):
        load_checkpoint(path, network=target, config=config)

    _assert_model_state(target, before)


@pytest.mark.parametrize(
    "damage",
    [
        "group-count",
        "identifier-order",
        "unknown-state-identifier",
        "state-schema",
        "momentum-shape",
        "group-option",
    ],
)
def test_load_rejects_mismatched_optimizer_state_before_any_mutation(
    tmp_path: Path,
    config: AppConfig,
    damage: str,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = cast("dict[str, object]", payload["optimizer_state"])
        groups = cast("list[dict[str, object]]", optimizer_state["param_groups"])
        states = cast("dict[int, dict[str, object]]", optimizer_state["state"])
        parameters = cast("list[int]", groups[0]["params"])
        first_state = states[next(iter(states))]
        if damage == "group-count":
            second_group = copy.deepcopy(groups[0])
            groups[0]["params"] = parameters[:1]
            second_group["params"] = parameters[1:]
            groups.append(second_group)
        elif damage == "identifier-order":
            parameters[0], parameters[1] = parameters[1], parameters[0]
        elif damage == "unknown-state-identifier":
            states[len(parameters)] = copy.deepcopy(first_state)
        elif damage == "state-schema":
            momentum = first_state.pop("momentum_buffer")
            first_state["exp_avg"] = momentum
        elif damage == "momentum-shape":
            momentum = cast("torch.Tensor", first_state["momentum_buffer"])
            first_state["momentum_buffer"] = momentum.reshape(-1)[:1]
        else:
            groups[0]["betas"] = (0.9, 0.999)

    _rewrite_payload(path, mutate)
    target = _network(config)
    target_optimizer = _optimizer(target, config)
    before_model = _clone_state(target)
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(CheckpointError, match="optimizer"):
        load_checkpoint(
            path,
            network=target,
            optimizer=target_optimizer,
            config=config,
        )

    _assert_model_state(target, before_model)
    _assert_nested_equal(target_optimizer.state_dict(), before_optimizer)


def test_load_rejects_destination_optimizer_group_structure_without_mutation(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    target = _network(config)
    parameters = list(target.parameters())
    target_optimizer = torch.optim.SGD(
        [
            {"params": parameters[:1]},
            {"params": parameters[1:]},
        ],
        lr=config.learner.learning_rate,
        momentum=config.learner.momentum,
        weight_decay=config.learner.weight_decay,
    )
    before_model = _clone_state(target)
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())

    with pytest.raises(CheckpointError, match="group count"):
        load_checkpoint(
            path,
            network=target,
            optimizer=target_optimizer,
            config=config,
        )

    _assert_model_state(target, before_model)
    _assert_nested_equal(target_optimizer.state_dict(), before_optimizer)


def test_inference_load_rejects_tampered_optimizer_hyperparameters(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = cast("dict[str, object]", payload["optimizer_state"])
        groups = cast("list[dict[str, object]]", optimizer_state["param_groups"])
        groups[0]["lr"] = 0.02

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="lr"):
        load_checkpoint(path, network=_network(config), config=config)


def test_load_rejects_nonfinite_optimizer_tensor(tmp_path: Path, config: AppConfig) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = cast("dict[str, object]", payload["optimizer_state"])
        states = cast("dict[int, dict[str, object]]", optimizer_state["state"])
        first = next(iter(states.values()))
        momentum = cast("torch.Tensor", first["momentum_buffer"]).clone()
        momentum.reshape(-1)[0] = torch.inf
        first["momentum_buffer"] = momentum

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="non-finite"):
        load_checkpoint(path, network=_network(config), config=config)


@pytest.mark.parametrize("field", ["config", "compatibility"])
def test_load_rejects_nonfinite_primitive_metadata(
    tmp_path: Path,
    config: AppConfig,
    field: str,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    def mutate(payload: dict[str, object]) -> None:
        mapping = cast("dict[str, object]", payload[field])
        mapping["bad"] = float("nan")

    _rewrite_payload(path, mutate)

    with pytest.raises(CheckpointError, match="non-finite"):
        load_checkpoint(path, network=_network(config), config=config)


def test_save_rejects_nonfinite_model_without_touching_target(
    tmp_path: Path,
    config: AppConfig,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"old checkpoint")
    network = _network(config)
    with torch.no_grad():
        next(network.parameters()).reshape(-1)[0] = torch.inf

    with pytest.raises(CheckpointError, match="non-finite"):
        save_checkpoint(
            path,
            network=network,
            optimizer=_optimizer(network, config),
            step=0,
            config=config,
        )

    assert path.read_bytes() == b"old checkpoint"


@pytest.mark.parametrize("failure", ["save", "replace"])
def test_atomic_save_failure_preserves_target_and_cleans_temporary_file(
    tmp_path: Path,
    config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"existing checkpoint")
    network = _network(config)
    optimizer = _optimizer(network, config)

    if failure == "save":
        def fail_save(*args: object, **kwargs: object) -> None:
            raise OSError("save failed")

        monkeypatch.setattr("azgo.checkpoint.torch.save", fail_save)
    else:
        def fail_replace(*args: object, **kwargs: object) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr("azgo.checkpoint.os.replace", fail_replace)

    with pytest.raises(CheckpointError, match="could not save"):
        save_checkpoint(path, network=network, optimizer=optimizer, step=0, config=config)

    assert path.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.glob(".checkpoint.pt.*.tmp")) == []


def test_optimizer_apply_failure_rolls_back_model_optimizer_and_rng(
    tmp_path: Path,
    config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)
    target = _network(config)
    target_optimizer = _optimizer(target, config)
    before_model = _clone_state(target)
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    torch.manual_seed(8080)
    before_rng = torch.get_rng_state().clone()
    original_load = target_optimizer.load_state_dict
    calls = 0

    def fail_once(state: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            target_optimizer.param_groups[0]["lr"] = 999.0
            raise RuntimeError("partial optimizer failure")
        original_load(state)

    monkeypatch.setattr(target_optimizer, "load_state_dict", fail_once)

    with pytest.raises(CheckpointError, match="could not apply"):
        load_checkpoint(
            path,
            network=target,
            optimizer=target_optimizer,
            config=config,
        )

    assert calls == 2
    _assert_model_state(target, before_model)
    _assert_nested_equal(target_optimizer.state_dict(), before_optimizer)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("restore_rng", [0, "false", object()])
def test_load_rejects_non_boolean_restore_rng(
    tmp_path: Path,
    config: AppConfig,
    restore_rng: object,
) -> None:
    path = tmp_path / "checkpoint.pt"
    _saved_checkpoint(path, config)

    with pytest.raises(CheckpointError, match="restore_rng"):
        load_checkpoint(
            path,
            network=_network(config),
            config=config,
            restore_rng=restore_rng,  # type: ignore[arg-type]
        )
