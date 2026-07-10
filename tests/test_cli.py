"""Command-line integration tests."""

import json
from pathlib import Path

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
""",
        encoding="utf-8",
    )
    return path


def test_validate_config_command(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(app, ["validate-config", str(config)])

    assert result.exit_code == 0, result.output
    assert '"board_size": 5' in result.output
    assert "Configuration is valid" in result.output


def test_benchmark_engine_command(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")

    result = CliRunner().invoke(app, ["benchmark-engine", "--config", str(config)])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["board_size"] == 5
    assert report["games_requested"] == 1
    assert report["moves"] == 2
    assert report["seed"] == 11

