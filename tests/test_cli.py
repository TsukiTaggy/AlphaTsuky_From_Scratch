"""Command-line integration tests."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from azgo.cli import app

if TYPE_CHECKING:
    from collections.abc import Callable

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
inference:
  max_batch_size: 16
search:
  simulations: 8
  c_puct: 1.5
  seed: 4242
  dirichlet_alpha: 0.3
  dirichlet_fraction: 0.25
self_play:
  seed: 4343
  games: 1
  workers: 1
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
arena:
  seed: 5401
  games: 4
  opening_moves: 4
  max_moves: 256
  promotion_threshold: 0.55
training_run:
  cycles: 1
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
    assert '"workers": 1' in result.output
    assert '"max_batch_size": 16' in result.output
    assert '"capacity": 10000' in result.output
    assert '"batch_size": 2' in result.output
    assert '"checkpoint_interval": 1' in result.output
    assert '"cycles": 1' in result.output
    assert "Configuration is valid" in result.output


def test_run_training_cycle_forwards_modes_and_reports_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo import training_run as training_run_module

    config = _config(tmp_path / "test.yaml")
    run_directory = tmp_path / "managed"
    calls: list[dict[str, object]] = []

    class FakeResult:
        def report(self) -> dict[str, object]:
            return {
                "completed_cycles": 1,
                "run_directory": str(run_directory.resolve()),
            }

    class FakeRunner:
        def __init__(
            self,
            settings: object,
            directory: Path,
            *,
            workers: int | None,
        ) -> None:
            calls.append(
                {
                    "directory": directory,
                    "settings": settings,
                    "workers": workers,
                }
            )

        def run(self, *, resume: bool) -> FakeResult:
            calls[-1]["resume"] = resume
            return FakeResult()

    monkeypatch.setattr(training_run_module, "TrainingRunRunner", FakeRunner)

    result = CliRunner().invoke(
        app,
        [
            "run-training-cycle",
            "-c",
            str(config),
            "--run-dir",
            str(run_directory),
            "--resume",
            "--workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "completed_cycles": 1,
        "run_directory": str(run_directory.resolve()),
    }
    assert len(calls) == 1
    assert calls[0]["directory"] == run_directory.resolve()
    assert calls[0]["workers"] == 1
    assert calls[0]["resume"] is True


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


def _pass_self_play_game(game_index: int) -> SelfPlayGame:
    from azgo.game import Color, GameState, Rules
    from azgo.self_play import SelfPlayGame, TrainingSample

    rules = Rules(board_size=5, komi=5.5)
    actions = (rules.pass_action, rules.pass_action)
    state = GameState.new(rules, zobrist_seed=7)
    for action in actions:
        state = state.apply(action)
    score = state.score()
    samples = []
    for move_number, (to_play, action) in enumerate(
        zip((Color.BLACK, Color.WHITE), actions, strict=True)
    ):
        policy = np.zeros(rules.action_size, dtype=np.float32)
        policy[action] = 1.0
        samples.append(
            TrainingSample(
                features=np.zeros((17, 5, 5), dtype=np.float32),
                policy=policy,
                value=float(score.outcome(to_play)),
                to_play=to_play,
                move_number=move_number,
                selected_action=action,
                game_index=game_index,
            )
        )
    return SelfPlayGame(
        samples=tuple(samples),
        actions=actions,
        final_score=score,
        winner=score.winner,
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
            "  games: 1\n  workers: 1\n  max_moves: 256",
            "  games: 3\n  workers: 1\n  max_moves: 256",
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
        "inference_batches": 0,
        "inference_max_batch_size": 0,
        "inference_mean_batch_size": 0.0,
        "inference_mode": "direct",
        "inference_positions": 0,
        "inference_requests": 0,
        "next_game_index": 3,
        "output": str(output.resolve()),
        "positions_generated": 6,
        "replay_capacity": 10000,
        "replay_size": 6,
        "self_play_workers": 1,
        "white_wins": 1,
    }
    loaded = ReplayBuffer.load(output)
    assert len(loaded) == 6
    assert loaded.next_game_index == 3


def test_generate_self_play_writes_sgf_and_inspect_command_validates_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo import self_play

    class PassRunner:
        def __init__(self, evaluator: object, config: object) -> None:
            del evaluator, config

        def play_game(self, game_index: int) -> SelfPlayGame:
            return _pass_self_play_game(game_index)

    monkeypatch.setattr(self_play, "SelfPlayRunner", PassRunner)
    config = _config(tmp_path / "test.yaml")
    replay = tmp_path / "replay.npz"
    sgf = tmp_path / "records" / "self-play.sgf"

    generated = CliRunner().invoke(
        app,
        [
            "generate-self-play",
            "-c",
            str(config),
            "-o",
            str(replay),
            "--sgf-output",
            str(sgf),
        ],
    )

    assert generated.exit_code == 0, generated.output
    report = json.loads(generated.output)
    assert report["sgf_output"] == str(sgf.resolve())
    assert report["sgf_sha256"] == sha256(sgf.read_bytes()).hexdigest()
    inspected = CliRunner().invoke(
        app,
        ["inspect-sgf", "-c", str(config), "--input", str(sgf)],
    )
    assert inspected.exit_code == 0, inspected.output
    inspection = json.loads(inspected.output)
    assert inspection["games_count"] == 1
    assert inspection["games"][0]["move_count"] == 2
    assert inspection["games"][0]["winner"] == "white"
    assert inspection["games"][0]["game_name"] == "self-play-00000000000000000000"


def test_self_play_sgf_bytes_are_independent_of_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo import self_play

    class PassRunner:
        def __init__(self, evaluator: object, config: object) -> None:
            del evaluator, config

        def play_game(self, game_index: int) -> SelfPlayGame:
            return _pass_self_play_game(game_index)

    monkeypatch.setattr(self_play, "SelfPlayRunner", PassRunner)
    config = _config(tmp_path / "test.yaml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  games: 1\n  workers: 1\n  max_moves: 256",
            "  games: 3\n  workers: 1\n  max_moves: 256",
        ),
        encoding="utf-8",
    )
    sgf_paths = (tmp_path / "single.sgf", tmp_path / "parallel.sgf")

    for index, workers in enumerate((1, 2)):
        result = CliRunner().invoke(
            app,
            [
                "generate-self-play",
                "-c",
                str(config),
                "-o",
                str(tmp_path / f"replay-{index}.npz"),
                "--workers",
                str(workers),
                "--sgf-output",
                str(sgf_paths[index]),
            ],
        )
        assert result.exit_code == 0, result.output

    assert sgf_paths[0].read_bytes() == sgf_paths[1].read_bytes()


def test_generate_self_play_worker_override_uses_parallel_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "  games: 1\n  workers: 1\n  max_moves: 256",
            "  games: 3\n  workers: 1\n  max_moves: 256",
        ),
        encoding="utf-8",
    )
    output = tmp_path / "parallel.npz"
    calls = _install_fake_self_play_runner(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "generate-self-play",
            "-c",
            str(config),
            "-o",
            str(output),
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(calls) == [0, 1, 2]
    report = json.loads(result.output)
    assert report["self_play_workers"] == 2
    assert report["inference_mode"] == "deterministic_batch"
    assert report["inference_requests"] == 0
    assert report["inference_positions"] == 0
    assert report["inference_batches"] == 0
    assert report["inference_mean_batch_size"] == 0.0
    assert report["inference_max_batch_size"] == 0


def test_generate_self_play_rejects_worker_override_above_game_count(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "test.yaml")
    output = tmp_path / "replay.npz"

    result = CliRunner().invoke(
        app,
        [
            "generate-self-play",
            "-c",
            str(config),
            "-o",
            str(output),
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 2
    assert "Self-play failed:" in result.output
    assert "workers must be no greater than self_play.games" in result.output
    assert not output.exists()


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


def _fake_arena_result(
    *,
    eligible: bool = True,
    candidate_score: float = 0.625,
    promotion_threshold: float = 0.55,
) -> SimpleNamespace:
    from azgo.game import Color, Score

    black_win = Score(1, 0, 0, 0, 24, 0.0)
    white_win = Score(0, 1, 0, 0, 24, 0.0)
    draw = Score(0, 0, 0, 0, 25, 0.0)
    games = (
        SimpleNamespace(
            pair_index=0,
            game_index=0,
            candidate_color=Color.BLACK,
            opening_actions=(0, 1),
            move_count=10,
            final_score=black_win,
            winner=Color.BLACK,
            candidate_outcome="win",
        ),
        SimpleNamespace(
            pair_index=0,
            game_index=1,
            candidate_color=Color.WHITE,
            opening_actions=(0, 1),
            move_count=11,
            final_score=draw,
            winner=None,
            candidate_outcome="draw",
        ),
        SimpleNamespace(
            pair_index=1,
            game_index=2,
            candidate_color=Color.BLACK,
            opening_actions=(2, 3),
            move_count=12,
            final_score=white_win,
            winner=Color.WHITE,
            candidate_outcome="loss",
        ),
        SimpleNamespace(
            pair_index=1,
            game_index=3,
            candidate_color=Color.WHITE,
            opening_actions=(2, 3),
            move_count=13,
            final_score=white_win,
            winner=Color.WHITE,
            candidate_outcome="win",
        ),
    )
    return SimpleNamespace(
        games=games,
        candidate_wins=2,
        incumbent_wins=1,
        draws=1,
        candidate_points=2.5,
        candidate_score=candidate_score,
        promotion_threshold=promotion_threshold,
        promotion_eligible=eligible,
    )


def _install_fake_arena(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: object | None = None,
    load_error: Exception | None = None,
    on_run: Callable[[], object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from azgo import arena as arena_module
    from azgo import checkpoint as checkpoint_module
    from azgo import cli as cli_module
    from azgo import evaluator as evaluator_module
    from azgo.checkpoint import CheckpointMetadata

    loads: list[dict[str, object]] = []
    runners: list[dict[str, object]] = []
    networks: list[object] = []
    arena_result = _fake_arena_result() if result is None else result

    def fake_build_network(settings: object) -> object:
        del settings
        network = object()
        networks.append(network)
        return network

    def fake_load_checkpoint(
        path: str | Path,
        *,
        network: object,
        config: object,
        optimizer: object | None = None,
        restore_rng: bool | None = None,
    ) -> CheckpointMetadata:
        del config
        loads.append(
            {
                "network": network,
                "optimizer": optimizer,
                "path": Path(path),
                "restore_rng": restore_rng,
            }
        )
        if load_error is not None:
            raise load_error
        return CheckpointMetadata(step=10 + len(loads), config={})

    class FakeArenaRunner:
        def __init__(
            self,
            candidate_evaluator: object,
            incumbent_evaluator: object,
            config: object,
        ) -> None:
            runners.append(
                {
                    "candidate": candidate_evaluator,
                    "config": config,
                    "incumbent": incumbent_evaluator,
                }
            )

        def run(self) -> object:
            if on_run is not None:
                on_run()
            return arena_result

    monkeypatch.setattr(cli_module, "_build_network", fake_build_network)
    monkeypatch.setattr(checkpoint_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(evaluator_module, "TorchEvaluator", lambda network: network)
    monkeypatch.setattr(arena_module, "ArenaRunner", FakeArenaRunner)
    return loads, runners


def _arena_command(
    config: Path,
    candidate: Path,
    incumbent: Path,
    promote_to: Path | None = None,
) -> list[str]:
    command = [
        "evaluate-arena",
        "-c",
        str(config),
        "--candidate",
        str(candidate),
        "--incumbent",
        str(incumbent),
    ]
    if promote_to is not None:
        command.extend(("--promote-to", str(promote_to)))
    return command


def test_evaluate_arena_reports_identities_results_and_compact_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    loads, runners = _install_fake_arena(monkeypatch)

    result = CliRunner().invoke(app, _arena_command(config, candidate, incumbent))

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["candidate"] == str(candidate.resolve())
    assert report["candidate_sha256"] == sha256(candidate.read_bytes()).hexdigest()
    assert report["candidate_step"] == 11
    assert report["incumbent"] == str(incumbent.resolve())
    assert report["incumbent_sha256"] == sha256(incumbent.read_bytes()).hexdigest()
    assert report["incumbent_step"] == 12
    assert report["candidate_wins"] == 2
    assert report["incumbent_wins"] == 1
    assert report["draws"] == 1
    assert report["candidate_points"] == 2.5
    assert report["candidate_score"] == 0.625
    assert report["promotion_threshold"] == 0.55
    assert report["promotion_eligible"] is True
    assert report["promotion_requested"] is False
    assert report["promoted"] is False
    assert report["promoted_to"] is None
    assert report["games_played"] == 4
    assert report["games"][0] == {
        "black_score": 1.0,
        "candidate_color": "black",
        "candidate_outcome": "win",
        "game_index": 0,
        "move_count": 10,
        "opening_actions": [0, 1],
        "pair_index": 0,
        "white_score": 0.0,
        "winner": "black",
    }
    assert report["games"][1]["winner"] is None
    assert len(loads) == 2
    assert loads[0]["network"] is not loads[1]["network"]
    assert all(load["optimizer"] is None for load in loads)
    assert all(load["restore_rng"] is False for load in loads)
    assert len(runners) == 1
    assert runners[0]["candidate"] is loads[0]["network"]
    assert runners[0]["incumbent"] is loads[1]["network"]


def test_evaluate_arena_writes_complete_color_swapped_sgf_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo.arena import ArenaGameResult, ArenaResult
    from azgo.game import Color, GameState, Rules
    from azgo.sgf import load_sgf_collection

    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    sgf = tmp_path / "arena.sgf"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    rules = Rules(board_size=5, komi=5.5)
    state = GameState.new(rules, zobrist_seed=7).apply(25).apply(25)
    score = state.score()
    games = (
        ArenaGameResult(
            pair_index=0,
            game_index=0,
            candidate_color=Color.BLACK,
            opening_actions=(),
            actions=(25, 25),
            move_count=2,
            final_score=score,
            winner=Color.WHITE,
            candidate_outcome="loss",
        ),
        ArenaGameResult(
            pair_index=0,
            game_index=1,
            candidate_color=Color.WHITE,
            opening_actions=(),
            actions=(25, 25),
            move_count=2,
            final_score=score,
            winner=Color.WHITE,
            candidate_outcome="win",
        ),
    )
    _install_fake_arena(monkeypatch, result=ArenaResult(games, 0.55))

    result = CliRunner().invoke(
        app,
        [
            *_arena_command(config, candidate, incumbent),
            "--sgf-output",
            str(sgf),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["sgf_output"] == str(sgf.resolve())
    assert report["sgf_sha256"] == sha256(sgf.read_bytes()).hexdigest()
    records = load_sgf_collection(sgf, expected_rules=rules, zobrist_seed=7)
    assert len(records) == 2
    assert records[0].actions == records[1].actions == (25, 25)
    assert records[0].black_player.startswith("candidate-")
    assert records[1].white_player.startswith("candidate-")


@pytest.mark.parametrize(
    ("eligible", "score", "promoted"),
    [(False, 0.5, False), (True, 0.55, True)],
)
def test_evaluate_arena_promotes_only_when_eligible_including_threshold_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    eligible: bool,
    score: float,
    promoted: bool,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    destination = tmp_path / "nested" / "best.pt"
    candidate_bytes = b"candidate checkpoint"
    original = b"original destination"
    candidate.write_bytes(candidate_bytes)
    incumbent.write_bytes(b"incumbent checkpoint")
    destination.parent.mkdir()
    destination.write_bytes(original)
    arena_result = _fake_arena_result(
        eligible=eligible,
        candidate_score=score,
        promotion_threshold=0.55,
    )
    _install_fake_arena(monkeypatch, result=arena_result)

    result = CliRunner().invoke(
        app,
        _arena_command(config, candidate, incumbent, destination),
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["promotion_eligible"] is eligible
    assert report["promotion_requested"] is True
    assert report["promoted"] is promoted
    assert report["promoted_to"] == (str(destination.resolve()) if promoted else None)
    assert destination.read_bytes() == (candidate_bytes if promoted else original)
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_evaluate_arena_allows_promotion_over_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    _install_fake_arena(monkeypatch)

    result = CliRunner().invoke(
        app,
        _arena_command(config, candidate, incumbent, incumbent),
    )

    assert result.exit_code == 0, result.output
    assert incumbent.read_bytes() == candidate.read_bytes()
    assert json.loads(result.output)["promoted_to"] == str(incumbent.resolve())


@pytest.mark.parametrize("same_role", ["incumbent", "promotion"])
def test_evaluate_arena_rejects_candidate_path_reuse(
    tmp_path: Path,
    same_role: str,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    command = _arena_command(
        config,
        candidate,
        candidate if same_role == "incumbent" else incumbent,
        candidate if same_role == "promotion" else None,
    )

    result = CliRunner().invoke(app, command)

    assert result.exit_code == 2
    assert "Arena evaluation failed:" in result.output
    assert "candidate" in result.output


def test_evaluate_arena_reports_missing_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path / "test.yaml")
    incumbent = tmp_path / "incumbent.pt"
    incumbent.write_bytes(b"incumbent checkpoint")

    result = CliRunner().invoke(
        app,
        _arena_command(config, tmp_path / "missing.pt", incumbent),
    )

    assert result.exit_code == 2
    assert "Arena evaluation failed:" in result.output


@pytest.mark.parametrize("message", ["malformed checkpoint", "incompatible checkpoint"])
def test_evaluate_arena_reports_checkpoint_load_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    from azgo.checkpoint import CheckpointError

    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    _install_fake_arena(monkeypatch, load_error=CheckpointError(message))

    result = CliRunner().invoke(app, _arena_command(config, candidate, incumbent))

    assert result.exit_code == 2
    assert "Arena evaluation failed:" in result.output
    assert message in result.output


def test_evaluate_arena_rejects_candidate_changed_during_arena(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    destination = tmp_path / "best.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    destination.write_bytes(b"original destination")
    _install_fake_arena(
        monkeypatch,
        on_run=lambda: candidate.write_bytes(b"changed candidate"),
    )

    result = CliRunner().invoke(
        app,
        _arena_command(config, candidate, incumbent, destination),
    )

    assert result.exit_code == 2
    assert "changed after arena evaluation" in result.output
    assert destination.read_bytes() == b"original destination"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_evaluate_arena_failure_does_not_touch_promotion_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azgo.arena import ArenaLimitError

    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    destination = tmp_path / "best.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    destination.write_bytes(b"original destination")

    def fail_arena() -> None:
        raise ArenaLimitError("synthetic move-limit failure")

    _install_fake_arena(monkeypatch, on_run=fail_arena)

    result = CliRunner().invoke(
        app,
        _arena_command(config, candidate, incumbent, destination),
    )

    assert result.exit_code == 2
    assert "synthetic move-limit failure" in result.output
    assert destination.read_bytes() == b"original destination"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


@pytest.mark.parametrize("failure", ["fsync", "replace"])
def test_evaluate_arena_promotion_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = _config(tmp_path / "test.yaml")
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    destination = tmp_path / "best.pt"
    candidate.write_bytes(b"candidate checkpoint")
    incumbent.write_bytes(b"incumbent checkpoint")
    destination.write_bytes(b"original destination")
    _install_fake_arena(monkeypatch)

    def fail(*args: object) -> None:
        del args
        raise OSError(f"synthetic {failure} failure")

    monkeypatch.setattr(os, failure, fail)

    result = CliRunner().invoke(
        app,
        _arena_command(config, candidate, incumbent, destination),
    )

    assert result.exit_code == 2
    assert f"synthetic {failure} failure" in result.output
    assert destination.read_bytes() == b"original destination"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []

