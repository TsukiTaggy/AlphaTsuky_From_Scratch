"""Command-line integration tests."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from azgo.cli import app


def _config(path: Path) -> Path:
    path.write_text(
        """\
game:
  board_size: 5
  komi: 5.5
  rules:
    ruleset: tromp_taylor
    scoring: area
    suicide: illegal
    superko: positional
    pass_repetition_exempt: true
zobrist:
  seed: 7
benchmark:
  seed: 11
  games: 1
  max_moves_per_game: 2
model:
  history_length: 8
  channels: 64
  residual_blocks: 4
  value_hidden_size: 64
search:
  simulations: 8
  c_puct: 1.5
  seed: 4242
  dirichlet_alpha: 0.3
  dirichlet_fraction: 0.25
""",
        encoding="utf-8",
    )
    return path


def test_validate_config_command(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(app, ["validate-config", str(config)])

    assert result.exit_code == 0, result.output
    assert '"board_size": 5' in result.output
    assert '"history_length": 8' in result.output
    assert '"residual_blocks": 4' in result.output
    assert '"simulations": 8' in result.output
    assert '"dirichlet_fraction": 0.25' in result.output
    assert "Configuration is valid" in result.output


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("channels: 64", "channels: 0"),
        ("history_length: 8", "history_length: 8.0"),
        ("value_hidden_size: 64", "value_hidden_size: 64\n  unknown: true"),
    ],
)
def test_validate_config_command_reports_invalid_model_settings(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    config = _config(tmp_path / "test.yaml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(original, replacement),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["validate-config", str(config)])

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output


def test_benchmark_engine_command(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(app, ["benchmark-engine", "--config", str(config)])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["board_size"] == 5
    assert report["games_requested"] == 1
    assert report["moves"] == 2
    assert report["seed"] == 11


def test_search_move_command_reports_uniform_search(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(app, ["search-move", "--config", str(config)])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["applied_moves"] == []
    assert report["board_size"] == 5
    assert report["root_noise"] is False
    assert report["simulations"] == 8
    assert len(report["visit_counts"]) == 26
    assert sum(report["visit_counts"]) == 8
    assert len(report["visit_policy"]) == 26
    assert sum(report["visit_policy"]) == pytest.approx(1.0)
    assert report["selected_action"] == 0
    assert report["selected_coordinate"] == [0, 0]
    assert report["selected_is_pass"] is False
    assert report["root_value"] == 0.0


def test_search_move_command_applies_repeatable_moves(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(
        app,
        ["search-move", "-c", str(config), "-m", "0", "-m", "1"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["applied_moves"] == [0, 1]
    assert report["selected_action"] not in {0, 1}
    assert sum(report["visit_counts"]) == 8


@pytest.mark.parametrize("moves", [["0", "0"], ["26"]])
def test_search_move_command_reports_invalid_move(
    tmp_path: Path,
    moves: list[str],
) -> None:
    config = _config(tmp_path / "test.yaml")
    arguments = ["search-move", "-c", str(config)]
    for move in moves:
        arguments.extend(("-m", move))

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert "Search failed:" in result.output


def test_search_move_command_rejects_terminal_root(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(
        app,
        ["search-move", "-c", str(config), "-m", "25", "-m", "25"],
    )

    assert result.exit_code == 2
    assert "Search failed:" in result.output


def test_search_move_seeded_root_noise_is_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")
    arguments = ["search-move", "-c", str(config), "--root-noise"]

    first = CliRunner().invoke(app, arguments)
    second = CliRunner().invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == json.loads(second.output)
    assert json.loads(first.output)["root_noise"] is True


def test_search_move_command_reports_invalid_search_settings(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")
    config.write_text(
        config.read_text(encoding="utf-8").replace("simulations: 8", "simulations: 0"),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["search-move", "-c", str(config)])

    assert result.exit_code == 2
    assert "Invalid configuration:" in result.output

