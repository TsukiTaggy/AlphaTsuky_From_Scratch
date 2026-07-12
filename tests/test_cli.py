"""Command-line integration tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from azgo.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    from azgo.game import Color
    from azgo.self_play import SelfPlayGame


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
self_play:
  seed: 4343
  games: 1
  max_moves: 256
  temperature: 1.0
  temperature_moves: 10
  root_noise: true
replay:
  capacity: 10000
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
    assert '"max_moves": 256' in result.output
    assert '"root_noise": true' in result.output
    assert '"capacity": 10000' in result.output
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


def _self_play_game(game_index: int, winner: Color | None) -> SelfPlayGame:
    from azgo.game import Color, Score
    from azgo.self_play import SelfPlayGame, TrainingSample

    if winner is Color.BLACK:
        black_stones, white_stones = 1, 0
    elif winner is Color.WHITE:
        black_stones, white_stones = 0, 1
    else:
        black_stones, white_stones = 0, 0
    score = Score(
        black_stones=black_stones,
        white_stones=white_stones,
        black_territory=0,
        white_territory=0,
        neutral_points=24,
        komi=0.0,
    )

    samples = []
    for move_number, to_play in enumerate((Color.BLACK, Color.WHITE)):
        action = move_number
        policy = np.zeros(26, dtype=np.float32)
        policy[action] = 1.0
        value = float(score.outcome(to_play))
        samples.append(
            TrainingSample(
                features=np.zeros((17, 5, 5), dtype=np.float32),
                policy=policy,
                value=value,
                to_play=to_play,
                move_number=move_number,
                selected_action=action,
                game_index=game_index,
            )
        )
    return SelfPlayGame(
        samples=tuple(samples),
        actions=(0, 1),
        final_score=score,
        winner=winner,
        game_index=game_index,
    )


def _install_fake_self_play_runner(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    from azgo import self_play
    from azgo.game import Color

    calls: list[int] = []

    class FakeRunner:
        def __init__(self, evaluator: object, config: object) -> None:
            del evaluator, config

        def play_game(self, game_index: int) -> SelfPlayGame:
            calls.append(game_index)
            winners = (Color.BLACK, Color.WHITE, None)
            return _self_play_game(game_index, winners[game_index % len(winners)])

    monkeypatch.setattr(self_play, "SelfPlayRunner", FakeRunner)
    return calls


def test_generate_self_play_creates_snapshot_and_reports_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo.replay import ReplayBuffer

    config = _config(tmp_path / "test.yaml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  games: 1\n  max_moves: 256",
            "  games: 3\n  max_moves: 256",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "nested" / "replay.npz"
    calls = _install_fake_self_play_runner(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["generate-self-play", "-c", str(config), "-o", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert calls == [0, 1, 2]
    report = json.loads(result.output)
    assert report == {
        "black_wins": 1,
        "board_size": 5,
        "draws": 1,
        "games_generated": 3,
        "next_game_index": 3,
        "output": str(output.resolve()),
        "positions_generated": 6,
        "replay_capacity": 10000,
        "replay_size": 6,
        "white_wins": 1,
    }
    loaded = ReplayBuffer.load(output)
    assert len(loaded) == 6
    assert loaded.next_game_index == 3


def test_generate_self_play_appends_and_overwrite_restarts_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo.replay import ReplayBuffer

    config = _config(tmp_path / "test.yaml")
    output = tmp_path / "replay.npz"
    calls = _install_fake_self_play_runner(monkeypatch)
    arguments = ["generate-self-play", "-c", str(config), "-o", str(output)]

    first = CliRunner().invoke(app, arguments)
    appended = CliRunner().invoke(app, [*arguments, "--no-overwrite"])

    assert first.exit_code == 0, first.output
    assert appended.exit_code == 0, appended.output
    assert calls == [0, 1]
    assert json.loads(appended.output)["next_game_index"] == 2
    assert len(ReplayBuffer.load(output)) == 4

    replaced = CliRunner().invoke(app, [*arguments, "--overwrite"])

    assert replaced.exit_code == 0, replaced.output
    assert calls == [0, 1, 0]
    report = json.loads(replaced.output)
    assert report["next_game_index"] == 1
    assert report["replay_size"] == 2
    assert len(ReplayBuffer.load(output)) == 2


def test_generate_self_play_rejects_incompatible_existing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo.replay import ReplayBuffer

    config = _config(tmp_path / "test.yaml")
    output = tmp_path / "replay.npz"
    ReplayBuffer(board_size=9, history_length=8, capacity=10000).save(output)
    original = output.read_bytes()
    calls = _install_fake_self_play_runner(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["generate-self-play", "-c", str(config), "-o", str(output)],
    )

    assert result.exit_code == 2
    assert "Self-play failed:" in result.output
    assert "does not match configuration" in result.output
    assert calls == []
    assert output.read_bytes() == original


def test_generate_self_play_reports_corrupt_snapshot_without_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    output = tmp_path / "replay.npz"
    original = b"not an npz snapshot"
    output.write_bytes(original)
    calls = _install_fake_self_play_runner(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["generate-self-play", "-c", str(config), "-o", str(output)],
    )

    assert result.exit_code == 2
    assert "Self-play failed:" in result.output
    assert calls == []
    assert output.read_bytes() == original


def test_generate_self_play_failure_does_not_mutate_existing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo import self_play

    config = _config(tmp_path / "test.yaml")
    output = tmp_path / "replay.npz"
    _install_fake_self_play_runner(monkeypatch)
    arguments = ["generate-self-play", "-c", str(config), "-o", str(output)]
    created = CliRunner().invoke(app, arguments)
    assert created.exit_code == 0, created.output
    original = output.read_bytes()

    class FailingRunner:
        def __init__(self, evaluator: object, settings: object) -> None:
            del evaluator, settings

        def play_game(self, game_index: int) -> SelfPlayGame:
            raise self_play.SelfPlayLimitError(f"game {game_index} reached max_moves")

    monkeypatch.setattr(self_play, "SelfPlayRunner", FailingRunner)

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2
    assert "Self-play failed:" in result.output
    assert "reached max_moves" in result.output
    assert output.read_bytes() == original

