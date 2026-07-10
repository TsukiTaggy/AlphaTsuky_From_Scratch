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


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_checked_in_engine_configurations_are_valid(board_size: int) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "engine" / f"go{board_size}.yaml")

    assert config.game.board_size == board_size


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


def test_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does-not-exist.yaml"))
