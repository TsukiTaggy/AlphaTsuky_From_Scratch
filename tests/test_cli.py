"""Command-line integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from azgo.cli import app

if TYPE_CHECKING:
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
learner:
  seed: 5301
  batch_size: 2
  steps: 2
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  value_loss_weight: 1.0
  gradient_clip_norm: 5.0
  checkpoint_interval: 1
  augment: true
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
    assert '"batch_size": 2' in result.output
    assert '"checkpoint_interval": 1' in result.output
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
    assert report["checkpoint_step"] is None
    assert report["evaluator"] == "uniform"
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
        "checkpoint_step": None,
        "draws": 1,
        "evaluator": "uniform",
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


def _write_training_replay(
    path: Path,
    *,
    board_size: int = 5,
    games: int = 1,
) -> Path:
    from azgo.game import Color
    from azgo.replay import ReplayBuffer

    buffer = ReplayBuffer(board_size=board_size, history_length=8, capacity=10000)
    if board_size == 5:
        for game_index in range(games):
            buffer.add_game(_self_play_game(game_index, Color.BLACK))
    buffer.save(path)
    return path


def _install_fake_training(
    monkeypatch: pytest.MonkeyPatch,
    *,
    restored_step: int = 0,
) -> tuple[list[int], list[dict[str, object]]]:
    from azgo import checkpoint as checkpoint_module
    from azgo import learner as learner_module
    from azgo.checkpoint import CheckpointMetadata
    from azgo.learner import TrainingMetrics

    saves: list[int] = []
    loads: list[dict[str, object]] = []

    class FakeLearner:
        def __init__(self, network: object, config: object) -> None:
            del network, config
            self.optimizer = object()
            self.step = 0

        def restore_step(self, step: int) -> None:
            self.step = step

        def train_step(self, batch: object) -> TrainingMetrics:
            del batch
            self.step += 1
            value = float(self.step)
            return TrainingMetrics(
                step=self.step,
                batch_size=2,
                policy_loss=value,
                value_loss=value + 0.25,
                total_loss=value + 0.5,
                gradient_norm=value + 0.75,
            )

    def fake_save_checkpoint(
        path: str | Path,
        *,
        network: object,
        optimizer: object,
        step: int,
        config: object,
    ) -> None:
        del network, optimizer, config
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(step), encoding="utf-8")
        saves.append(step)

    def fake_load_checkpoint(
        path: str | Path,
        *,
        network: object,
        config: object,
        optimizer: object | None = None,
        restore_rng: bool | None = None,
    ) -> CheckpointMetadata:
        del network, config
        loads.append(
            {
                "optimizer": optimizer,
                "path": Path(path),
                "restore_rng": restore_rng,
            }
        )
        return CheckpointMetadata(step=restored_step, config={})

    monkeypatch.setattr(learner_module, "Learner", FakeLearner)
    monkeypatch.setattr(checkpoint_module, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(checkpoint_module, "load_checkpoint", fake_load_checkpoint)
    return saves, loads


def test_train_network_creates_periodic_and_final_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")
    checkpoint = tmp_path / "nested" / "model.pt"
    saves, loads = _install_fake_training(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
        ],
    )

    assert result.exit_code == 0, result.output
    assert saves == [1, 2, 2]
    assert loads == []
    report = json.loads(result.output)
    assert report == {
        "board_size": 5,
        "checkpoint": str(checkpoint.resolve()),
        "end_step": 2,
        "final_gradient_norm": 2.75,
        "final_policy_loss": 2.0,
        "final_total_loss": 2.5,
        "final_value_loss": 2.25,
        "mean_policy_loss": 1.5,
        "mean_total_loss": 2.0,
        "mean_value_loss": 1.75,
        "replay_size": 2,
        "resumed": False,
        "start_step": 0,
        "steps_completed": 2,
    }
    assert checkpoint.read_text(encoding="utf-8") == "2"


def test_train_network_refuses_existing_checkpoint_without_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing")
    saves, loads = _install_fake_training(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
        ],
    )

    assert result.exit_code == 2
    assert "Training failed:" in result.output
    assert "already exists" in result.output
    assert checkpoint.read_bytes() == b"existing"
    assert saves == []
    assert loads == []


def test_train_network_overwrite_starts_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing")
    saves, loads = _install_fake_training(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["start_step"] == 0
    assert saves == [1, 2, 2]
    assert loads == []


def test_train_network_resume_restores_and_runs_additional_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing")
    saves, loads = _install_fake_training(monkeypatch, restored_step=7)

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["resumed"] is True
    assert report["start_step"] == 7
    assert report["end_step"] == 9
    assert report["steps_completed"] == 2
    assert saves == [8, 9, 9]
    assert len(loads) == 1
    assert loads[0]["path"] == checkpoint.resolve()
    assert loads[0]["optimizer"] is not None
    assert loads[0]["restore_rng"] is True


def test_train_network_rejects_resume_with_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"existing")

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--resume",
            "--overwrite",
        ],
    )

    assert result.exit_code == 2
    assert "Training failed:" in result.output
    assert "cannot be used together" in result.output


def test_train_network_resume_requires_existing_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(tmp_path / "replay.npz")

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--resume",
        ],
    )

    assert result.exit_code == 2
    assert "Training failed:" in result.output
    assert "requires an existing checkpoint" in result.output


def test_train_network_rejects_incompatible_checkpoint(tmp_path: Path) -> None:
    import torch

    from azgo.checkpoint import save_checkpoint
    from azgo.config import load_config
    from azgo.learner import Learner
    from azgo.network import PolicyValueNetwork

    config = _config(tmp_path / "test.yaml")
    settings = load_config(config)
    torch.manual_seed(settings.learner.seed)
    network = PolicyValueNetwork(
        board_size=settings.game.board_size,
        history_length=settings.model.history_length,
        channels=settings.model.channels,
        residual_blocks=settings.model.residual_blocks,
        value_hidden_size=settings.model.value_hidden_size,
    )
    learner = Learner(network, settings)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        network=network,
        optimizer=learner.optimizer,
        step=3,
        config=settings,
    )
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "learning_rate: 0.01",
            "learning_rate: 0.02",
        ),
        encoding="utf-8",
    )
    replay = _write_training_replay(tmp_path / "replay.npz")

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(checkpoint),
            "--resume",
        ],
    )

    assert result.exit_code == 2
    assert "Training failed:" in result.output
    assert "incompatible" in result.output


@pytest.mark.parametrize(
    ("board_size", "expected"),
    [(9, "metadata does not match"), (5, "smaller than batch_size")],
)
def test_train_network_rejects_incompatible_or_insufficient_replay(
    tmp_path: Path,
    board_size: int,
    expected: str,
) -> None:
    config = _config(tmp_path / "test.yaml")
    replay = _write_training_replay(
        tmp_path / "replay.npz",
        board_size=board_size,
        games=0,
    )

    result = CliRunner().invoke(
        app,
        [
            "train-network",
            "-c",
            str(config),
            "--replay",
            str(replay),
            "--checkpoint",
            str(tmp_path / "model.pt"),
        ],
    )

    assert result.exit_code == 2
    assert "Training failed:" in result.output
    assert expected in result.output


def test_checkpoint_evaluator_is_shared_by_search_and_self_play(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo import checkpoint as checkpoint_module
    from azgo import evaluator as evaluator_module
    from azgo.checkpoint import CheckpointMetadata

    config = _config(tmp_path / "test.yaml")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stub")
    output = tmp_path / "replay.npz"
    calls: list[dict[str, object]] = []

    def fake_load_checkpoint(
        path: str | Path,
        *,
        network: object,
        config: object,
        optimizer: object | None = None,
        restore_rng: bool | None = None,
    ) -> CheckpointMetadata:
        del network, config
        calls.append(
            {
                "optimizer": optimizer,
                "path": Path(path),
                "restore_rng": restore_rng,
            }
        )
        return CheckpointMetadata(step=12, config={})

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        evaluator_module,
        "TorchEvaluator",
        lambda network: evaluator_module.UniformEvaluator(),
    )
    _install_fake_self_play_runner(monkeypatch)

    searched = CliRunner().invoke(
        app,
        [
            "search-move",
            "-c",
            str(config),
            "--checkpoint",
            str(checkpoint),
        ],
    )
    generated = CliRunner().invoke(
        app,
        [
            "generate-self-play",
            "-c",
            str(config),
            "-o",
            str(output),
            "--checkpoint",
            str(checkpoint),
        ],
    )

    assert searched.exit_code == 0, searched.output
    assert generated.exit_code == 0, generated.output
    for result in (searched, generated):
        report = json.loads(result.output)
        assert report["evaluator"] == "checkpoint"
        assert report["checkpoint_step"] == 12
    assert len(calls) == 2
    assert all(call["path"] == checkpoint.resolve() for call in calls)
    assert all(call["optimizer"] is None for call in calls)
    assert all(call["restore_rng"] is False for call in calls)


@pytest.mark.parametrize(
    ("arguments", "prefix"),
    [
        (["search-move"], "Search failed:"),
        (["generate-self-play"], "Self-play failed:"),
    ],
)
def test_commands_report_malformed_checkpoint(
    tmp_path: Path,
    arguments: list[str],
    prefix: str,
) -> None:
    config = _config(tmp_path / "test.yaml")
    checkpoint = tmp_path / "malformed.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    command = [*arguments, "-c", str(config)]
    if arguments == ["generate-self-play"]:
        command.extend(("-o", str(tmp_path / "replay.npz")))
    command.extend(("--checkpoint", str(checkpoint)))

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 2
    assert prefix in result.output

