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


def test_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does-not-exist.yaml"))
