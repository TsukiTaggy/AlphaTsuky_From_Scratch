"""Configuration composition and validation tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from azgo.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_config(path: Path, *, board_size: int = 5, extra: str = "") -> Path:
    path.write_text(
        f"""\
game:
  board_size: {board_size}
  komi: 5.5
  rules:
    ruleset: tromp_taylor
    scoring: area
    suicide: illegal
    superko: positional
    pass_repetition_exempt: true
zobrist:
  seed: 1234
benchmark:
  seed: 5678
  games: 3
  max_moves_per_game: 100
model:
  history_length: 8
  channels: 64
  residual_blocks: 4
  value_hidden_size: 64
search:
  simulations: 100
  c_puct: 1.5
  seed: 4242
  dirichlet_alpha: 0.3
  dirichlet_fraction: 0.25
self_play:
  seed: 4343
  games: 2
  workers: 2
  max_moves: 256
  temperature: 1.0
  temperature_moves: 10
  root_noise: true
inference:
  max_batch_size: 16
replay:
  capacity: 10000
learner:
  seed: 5301
  batch_size: 32
  steps: 10
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  value_loss_weight: 1.0
  gradient_clip_norm: 5.0
  checkpoint_interval: 5
  augment: true
arena:
  seed: 5401
  games: 4
  opening_moves: 4
  max_moves: 128
  promotion_threshold: 0.55
training_run:
  cycles: 1
{extra}""",
        encoding="utf-8",
    )
    return path


def test_load_config_composes_and_validates_yaml(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path / "engine.yaml"))

    assert config.game.board_size == 5
    assert config.game.komi == 5.5
    assert config.zobrist.seed == 1234
    assert config.benchmark.games == 3
    assert config.model.history_length == 8
    assert config.model.channels == 64
    assert config.model.residual_blocks == 4
    assert config.model.value_hidden_size == 64
    assert config.search.simulations == 100
    assert config.search.c_puct == 1.5
    assert config.search.seed == 4242
    assert config.search.dirichlet_alpha == 0.3
    assert config.search.dirichlet_fraction == 0.25
    assert config.self_play.seed == 4343
    assert config.self_play.games == 2
    assert config.self_play.workers == 2
    assert config.self_play.max_moves == 256
    assert config.self_play.temperature == 1.0
    assert config.self_play.temperature_moves == 10
    assert config.self_play.root_noise is True
    assert config.inference.max_batch_size == 16
    assert config.replay.capacity == 10000
    assert config.learner.seed == 5301
    assert config.learner.batch_size == 32
    assert config.learner.steps == 10
    assert config.learner.learning_rate == 0.01
    assert config.learner.momentum == 0.9
    assert config.learner.weight_decay == 0.0001
    assert config.learner.value_loss_weight == 1.0
    assert config.learner.gradient_clip_norm == 5.0
    assert config.learner.checkpoint_interval == 5
    assert config.learner.augment is True
    assert config.arena.seed == 5401
    assert config.arena.games == 4
    assert config.arena.opening_moves == 4
    assert config.arena.max_moves == 128
    assert config.arena.promotion_threshold == 0.55
    assert config.training_run.cycles == 1


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_checked_in_engine_configurations_are_valid(board_size: int) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "engine" / f"go{board_size}.yaml")

    assert config.game.board_size == board_size
    assert config.model.history_length == 8
    assert config.model.channels == 64
    assert config.model.residual_blocks == 4
    assert config.model.value_hidden_size == 64
    assert config.search.simulations == 100
    assert config.search.dirichlet_alpha == (0.3 if board_size in {5, 9} else 0.03)
    assert config.search.dirichlet_fraction == 0.25
    assert config.self_play.seed == board_size * 1000 + 201
    assert config.self_play.games == 1
    assert config.self_play.workers == 1
    assert config.self_play.max_moves == {5: 256, 9: 512, 13: 768, 19: 1024}[board_size]
    assert config.self_play.temperature == 1.0
    assert config.self_play.temperature_moves == {5: 10, 9: 20, 13: 30, 19: 30}[board_size]
    assert config.self_play.root_noise is True
    assert config.inference.max_batch_size == 16
    assert config.replay.capacity == 10000
    assert config.learner.seed == board_size * 1000 + 301
    assert config.learner.batch_size == 32
    assert config.learner.steps == 10
    assert config.learner.learning_rate == 0.01
    assert config.learner.momentum == 0.9
    assert config.learner.weight_decay == 0.0001
    assert config.learner.value_loss_weight == 1.0
    assert config.learner.gradient_clip_norm == 5.0
    assert config.learner.checkpoint_interval == 5
    assert config.learner.augment is True
    assert config.arena.seed == board_size * 1000 + 401
    assert config.arena.games == 4
    assert config.arena.opening_moves == {5: 4, 9: 8, 13: 12, 19: 16}[board_size]
    assert config.arena.max_moves == {5: 256, 9: 512, 13: 768, 19: 1024}[board_size]
    assert config.arena.promotion_threshold == 0.55
    assert config.training_run.cycles == 1


def test_hydra_override_is_validated(tmp_path: Path) -> None:
    config = load_config(
        _write_config(tmp_path / "engine.yaml"),
        overrides=("game.board_size=9", "game.komi=7.5"),
    )

    assert config.game.board_size == 9
    assert config.game.komi == 7.5


@pytest.mark.parametrize("board_size", [0, 4, 6, 8, 10, 18, 20])
def test_unsupported_board_size_is_rejected(tmp_path: Path, board_size: int) -> None:
    with pytest.raises(ValidationError):
        load_config(_write_config(tmp_path / "invalid.yaml", board_size=board_size))


@pytest.mark.parametrize("komi", [".inf", "-.inf", ".nan"])
def test_non_finite_komi_is_rejected(tmp_path: Path, komi: str) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(path.read_text(encoding="utf-8").replace("komi: 5.5", f"komi: {komi}"))

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("board_size: 5", "board_size: 5.0"),
        ("board_size: 5", 'board_size: "5"'),
        ("seed: 1234", 'seed: "1234"'),
    ],
)
def test_configuration_does_not_coerce_wrong_scalar_types(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement, 1),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(_write_config(tmp_path / "invalid.yaml", extra="unknown: true\n"))


@pytest.mark.parametrize(
    ("field", "configured_value"),
    [
        ("history_length", 8),
        ("channels", 64),
        ("residual_blocks", 4),
        ("value_hidden_size", 64),
    ],
)
def test_model_dimensions_must_be_positive(
    tmp_path: Path,
    field: str,
    configured_value: int,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f"{field}: {configured_value}",
            f"{field}: 0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("invalid_value", ["8.0", '"8"', "true"])
def test_model_dimensions_do_not_coerce_non_integers(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "history_length: 8",
            f"history_length: {invalid_value}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_model_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  value_hidden_size: 64",
            "  value_hidden_size: 64\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_model_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("model:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("simulations: 100", "simulations: 0"),
        ("simulations: 100", "simulations: -1"),
        ("simulations: 100", "simulations: 100.0"),
        ("simulations: 100", 'simulations: "100"'),
        ("simulations: 100", "simulations: true"),
        ("c_puct: 1.5", "c_puct: 0.0"),
        ("c_puct: 1.5", "c_puct: -1.0"),
        ("c_puct: 1.5", "c_puct: .inf"),
        ("c_puct: 1.5", "c_puct: .nan"),
        ("c_puct: 1.5", "c_puct: 1"),
        ("c_puct: 1.5", 'c_puct: "1.5"'),
        ("seed: 4242", "seed: -1"),
        ("seed: 4242", "seed: 18446744073709551616"),
        ("seed: 4242", "seed: 4242.0"),
        ("seed: 4242", "seed: true"),
        ("dirichlet_alpha: 0.3", "dirichlet_alpha: 0.0"),
        ("dirichlet_alpha: 0.3", "dirichlet_alpha: -0.1"),
        ("dirichlet_alpha: 0.3", "dirichlet_alpha: .inf"),
        ("dirichlet_alpha: 0.3", "dirichlet_alpha: 1"),
        ("dirichlet_fraction: 0.25", "dirichlet_fraction: -0.01"),
        ("dirichlet_fraction: 0.25", "dirichlet_fraction: 1.01"),
        ("dirichlet_fraction: 0.25", "dirichlet_fraction: .nan"),
        ("dirichlet_fraction: 0.25", "dirichlet_fraction: 0"),
    ],
)
def test_invalid_search_configuration_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_dirichlet_fraction_includes_unit_interval_endpoints(
    tmp_path: Path,
    fraction: float,
) -> None:
    path = _write_config(tmp_path / "valid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "dirichlet_fraction: 0.25",
            f"dirichlet_fraction: {fraction}",
        ),
        encoding="utf-8",
    )

    assert load_config(path).search.dirichlet_fraction == fraction


def test_unknown_search_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  dirichlet_fraction: 0.25",
            "  dirichlet_fraction: 0.25\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_search_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("search:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("seed: 4343", "seed: -1"),
        ("seed: 4343", "seed: 18446744073709551616"),
        ("seed: 4343", "seed: 4343.0"),
        ("seed: 4343", "seed: true"),
        ("games: 2", "games: 0"),
        ("games: 2", "games: -1"),
        ("games: 2", "games: 2.0"),
        ("games: 2", "games: true"),
        ("workers: 2", "workers: 0"),
        ("workers: 2", "workers: -1"),
        ("workers: 2", "workers: 2.0"),
        ("workers: 2", 'workers: "2"'),
        ("workers: 2", "workers: true"),
        ("max_moves: 256", "max_moves: 1"),
        ("max_moves: 256", "max_moves: 256.0"),
        ("max_moves: 256", "max_moves: true"),
        ("temperature: 1.0", "temperature: 0.0"),
        ("temperature: 1.0", "temperature: -1.0"),
        ("temperature: 1.0", "temperature: .inf"),
        ("temperature: 1.0", "temperature: .nan"),
        ("temperature: 1.0", "temperature: 1"),
        ("temperature: 1.0", 'temperature: "1.0"'),
        ("temperature_moves: 10", "temperature_moves: -1"),
        ("temperature_moves: 10", "temperature_moves: 10.0"),
        ("temperature_moves: 10", "temperature_moves: true"),
        ("root_noise: true", "root_noise: 1"),
        ("root_noise: true", 'root_noise: "true"'),
    ],
)
def test_invalid_self_play_configuration_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("temperature_moves", [0, 10])
def test_temperature_moves_accepts_nonnegative_integers(
    tmp_path: Path,
    temperature_moves: int,
) -> None:
    path = _write_config(tmp_path / "valid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "temperature_moves: 10",
            f"temperature_moves: {temperature_moves}",
        ),
        encoding="utf-8",
    )

    assert load_config(path).self_play.temperature_moves == temperature_moves


def test_self_play_workers_cannot_exceed_games(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace("workers: 2", "workers: 3"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="workers"):
        load_config(path)


def test_unknown_self_play_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  root_noise: true",
            "  root_noise: true\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_self_play_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    start = contents.index("self_play:\n")
    end = contents.index("replay:\n")
    path.write_text(contents[:start] + contents[end:], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("max_batch_size", ["0", "-1", "16.0", '"16"', "true"])
def test_invalid_inference_configuration_is_rejected(
    tmp_path: Path,
    max_batch_size: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "max_batch_size: 16",
            f"max_batch_size: {max_batch_size}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_inference_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  max_batch_size: 16",
            "  max_batch_size: 16\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_inference_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    start = contents.index("inference:\n")
    end = contents.index("replay:\n")
    path.write_text(contents[:start] + contents[end:], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("capacity", ["0", "-1", "10000.0", '"10000"', "true"])
def test_invalid_replay_capacity_is_rejected(tmp_path: Path, capacity: str) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace("capacity: 10000", f"capacity: {capacity}"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_replay_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  capacity: 10000",
            "  capacity: 10000\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_replay_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("replay:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_replay_capacity_must_hold_one_learner_batch(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace("capacity: 10000", "capacity: 31"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"learner\.batch_size"):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("seed: 5301", "seed: -1"),
        ("seed: 5301", "seed: 18446744073709551616"),
        ("seed: 5301", "seed: 5301.0"),
        ("seed: 5301", "seed: true"),
        ("batch_size: 32", "batch_size: 0"),
        ("batch_size: 32", "batch_size: 32.0"),
        ("batch_size: 32", "batch_size: true"),
        ("steps: 10", "steps: 0"),
        ("steps: 10", "steps: 10.0"),
        ("learning_rate: 0.01", "learning_rate: 0.0"),
        ("learning_rate: 0.01", "learning_rate: .inf"),
        ("learning_rate: 0.01", "learning_rate: 1"),
        ("momentum: 0.9", "momentum: -0.1"),
        ("momentum: 0.9", "momentum: 1.0"),
        ("momentum: 0.9", "momentum: .nan"),
        ("momentum: 0.9", "momentum: 0"),
        ("weight_decay: 0.0001", "weight_decay: -0.1"),
        ("weight_decay: 0.0001", "weight_decay: .inf"),
        ("weight_decay: 0.0001", "weight_decay: 0"),
        ("value_loss_weight: 1.0", "value_loss_weight: 0.0"),
        ("value_loss_weight: 1.0", "value_loss_weight: .nan"),
        ("value_loss_weight: 1.0", "value_loss_weight: 1"),
        ("gradient_clip_norm: 5.0", "gradient_clip_norm: 0.0"),
        ("gradient_clip_norm: 5.0", "gradient_clip_norm: .inf"),
        ("gradient_clip_norm: 5.0", "gradient_clip_norm: 5"),
        ("checkpoint_interval: 5", "checkpoint_interval: 0"),
        ("checkpoint_interval: 5", "checkpoint_interval: 5.0"),
        ("augment: true", "augment: 1"),
        ("augment: true", 'augment: "true"'),
    ],
)
def test_invalid_learner_configuration_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("cycles", ["0", "-1", "1.0", '"1"', "true"])
def test_invalid_training_run_cycles_are_rejected(tmp_path: Path, cycles: str) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace("cycles: 1", f"cycles: {cycles}"),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_unknown_training_run_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  cycles: 1",
            "  cycles: 1\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_training_run_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("training_run:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement", "expected"),
    [
        ("momentum: 0.9", "momentum: 0.0", 0.0),
        ("weight_decay: 0.0001", "weight_decay: 0.0", 0.0),
    ],
)
def test_learner_nonnegative_float_endpoints_are_valid(
    tmp_path: Path,
    original: str,
    replacement: str,
    expected: float,
) -> None:
    path = _write_config(tmp_path / "valid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    config = load_config(path)
    field = replacement.split(":", maxsplit=1)[0]
    assert getattr(config.learner, field) == expected


def test_unknown_learner_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  augment: true",
            "  augment: true\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_learner_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("learner:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("seed: 5401", "seed: -1"),
        ("seed: 5401", "seed: 18446744073709551616"),
        ("seed: 5401", "seed: 5401.0"),
        ("seed: 5401", 'seed: "5401"'),
        ("seed: 5401", "seed: true"),
        ("games: 4", "games: 0"),
        ("games: 4", "games: -2"),
        ("games: 4", "games: 1"),
        ("games: 4", "games: 3"),
        ("games: 4", "games: 4.0"),
        ("games: 4", 'games: "4"'),
        ("games: 4", "games: true"),
        ("opening_moves: 4", "opening_moves: -1"),
        ("opening_moves: 4", "opening_moves: 4.0"),
        ("opening_moves: 4", 'opening_moves: "4"'),
        ("opening_moves: 4", "opening_moves: true"),
        ("max_moves: 128", "max_moves: 0"),
        ("max_moves: 128", "max_moves: 1"),
        ("max_moves: 128", "max_moves: 128.0"),
        ("max_moves: 128", 'max_moves: "128"'),
        ("max_moves: 128", "max_moves: true"),
        ("promotion_threshold: 0.55", "promotion_threshold: 0.5"),
        ("promotion_threshold: 0.55", "promotion_threshold: 0.49"),
        ("promotion_threshold: 0.55", "promotion_threshold: 1.01"),
        ("promotion_threshold: 0.55", "promotion_threshold: .inf"),
        ("promotion_threshold: 0.55", "promotion_threshold: -.inf"),
        ("promotion_threshold: 0.55", "promotion_threshold: .nan"),
        ("promotion_threshold: 0.55", "promotion_threshold: 1"),
        ("promotion_threshold: 0.55", 'promotion_threshold: "0.55"'),
        ("promotion_threshold: 0.55", "promotion_threshold: true"),
    ],
)
def test_invalid_arena_configuration_is_rejected(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("opening_moves", [128, 129])
def test_arena_opening_must_be_below_move_limit(
    tmp_path: Path,
    opening_moves: int,
) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "opening_moves: 4",
            f"opening_moves: {opening_moves}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


@pytest.mark.parametrize("promotion_threshold", [0.5000001, 1.0])
def test_arena_promotion_threshold_accepts_valid_endpoints(
    tmp_path: Path,
    promotion_threshold: float,
) -> None:
    path = _write_config(tmp_path / "valid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "promotion_threshold: 0.55",
            f"promotion_threshold: {promotion_threshold}",
        ),
        encoding="utf-8",
    )

    assert load_config(path).arena.promotion_threshold == promotion_threshold


def test_arena_integer_boundaries_are_valid(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "valid.yaml")
    contents = path.read_text(encoding="utf-8")
    contents = contents.replace("seed: 5401", "seed: 18446744073709551615")
    contents = contents.replace("games: 4", "games: 2")
    contents = contents.replace("opening_moves: 4", "opening_moves: 0")
    contents = contents.replace("max_moves: 128", "max_moves: 2")
    path.write_text(contents, encoding="utf-8")

    config = load_config(path)
    assert config.arena.seed == (2**64) - 1
    assert config.arena.games == 2
    assert config.arena.opening_moves == 0
    assert config.arena.max_moves == 2


def test_unknown_arena_configuration_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "  promotion_threshold: 0.55",
            "  promotion_threshold: 0.55\n  unknown: true",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path)


def test_arena_configuration_is_required(tmp_path: Path) -> None:
    path = _write_config(tmp_path / "invalid.yaml")
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents[: contents.index("arena:\n")], encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does-not-exist.yaml"))
