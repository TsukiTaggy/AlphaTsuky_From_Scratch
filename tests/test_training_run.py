"""Tests for deterministic, crash-safe managed AlphaZero training runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from azgo import operations
from azgo.config import AppConfig, load_config
from azgo.inference import InferenceMetrics
from azgo.learner import TrainingError, TrainingMetrics
from azgo.operations import (
    CheckpointIdentity,
    SelfPlayOperationResult,
    SgfArtifactIdentity,
    TrainingOperationResult,
)
from azgo.training_run import TrainingRunError, TrainingRunRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(*, cycles: int = 2, workers: int = 2) -> AppConfig:
    raw = load_config(PROJECT_ROOT / "configs" / "engine" / "go5.yaml").model_dump(
        mode="json"
    )
    cast("dict[str, object]", raw["model"]).update(
        channels=2,
        residual_blocks=1,
        value_hidden_size=2,
    )
    cast("dict[str, object]", raw["search"])["simulations"] = 1
    cast("dict[str, object]", raw["self_play"]).update(
        games=2,
        workers=workers,
        max_moves=128,
        root_noise=False,
    )
    cast("dict[str, object]", raw["replay"])["capacity"] = 32
    cast("dict[str, object]", raw["learner"]).update(
        batch_size=2,
        steps=1,
        checkpoint_interval=1,
        augment=False,
    )
    cast("dict[str, object]", raw["arena"]).update(
        games=2,
        opening_moves=0,
        max_moves=128,
    )
    cast("dict[str, object]", raw["training_run"])["cycles"] = cycles
    return AppConfig.model_validate(raw)


class _FakeArenaOperation:
    def __init__(
        self,
        candidate: CheckpointIdentity,
        incumbent: CheckpointIdentity,
        *,
        promoted: bool,
        score: float,
        sgf: SgfArtifactIdentity | None,
    ) -> None:
        self.candidate = candidate
        self.incumbent = incumbent
        self.arena = SimpleNamespace(
            promotion_eligible=promoted,
            candidate_score=score,
        )
        self.sgf = sgf

    def report(self) -> dict[str, object]:
        report: dict[str, object] = {
            "candidate": str(self.candidate.path),
            "candidate_score": self.arena.candidate_score,
            "candidate_sha256": self.candidate.sha256,
            "candidate_step": self.candidate.step,
            "incumbent": str(self.incumbent.path),
            "incumbent_sha256": self.incumbent.sha256,
            "incumbent_step": self.incumbent.step,
            "promotion_eligible": self.arena.promotion_eligible,
        }
        if self.sgf is not None:
            report["sgf_output"] = str(self.sgf.path)
            report["sgf_sha256"] = self.sgf.sha256
        return report


def _install_fake_operations(
    monkeypatch: pytest.MonkeyPatch,
    *,
    promotions: tuple[bool, ...],
    fail_once: str | None = None,
) -> dict[str, list[object]]:
    calls: dict[str, list[object]] = {
        "bootstrap": [],
        "self_play": [],
        "training": [],
        "arena": [],
    }
    failed: set[str] = set()

    def maybe_fail(stage: str) -> None:
        if fail_once == stage and stage not in failed:
            failed.add(stage)
            raise TrainingError(f"synthetic {stage} interruption")

    def fake_bootstrap(
        config: AppConfig,
        destination: Path,
        **kwargs: object,
    ) -> CheckpointIdentity:
        del config, kwargs
        calls["bootstrap"].append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bootstrap")
        return CheckpointIdentity(
            destination.resolve(),
            operations.sha256_file(destination),
            0,
        )

    def fake_self_play(
        config: AppConfig,
        output: Path,
        *,
        checkpoint: Path | None,
        base_replay: Path | None = None,
        workers: int | None = None,
        minimum_positions: int = 0,
        sgf_output: Path | None = None,
        **kwargs: object,
    ) -> SelfPlayOperationResult:
        del kwargs
        call_index = len(calls["self_play"]) + 1
        calls["self_play"].append(
            (output, checkpoint, base_replay, workers, minimum_positions)
        )
        maybe_fail("self_play")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(f"replay-{call_index}".encode())
        sgf: SgfArtifactIdentity | None = None
        if sgf_output is not None:
            sgf_output.write_bytes(f"self-play-sgf-{call_index}".encode())
            sgf = SgfArtifactIdentity(
                sgf_output.resolve(),
                operations.sha256_file(sgf_output),
                config.self_play.games,
            )
        return SelfPlayOperationResult(
            output=output.resolve(),
            board_size=config.game.board_size,
            evaluator="checkpoint",
            checkpoint_step=(call_index - 1) * config.learner.steps,
            games_generated=config.self_play.games,
            generation_batches=1,
            positions_generated=2,
            black_wins=1,
            white_wins=1,
            draws=0,
            replay_capacity=config.replay.capacity,
            replay_size=call_index * 2,
            next_game_index=call_index * config.self_play.games,
            effective_workers=workers or config.self_play.workers,
            inference_mode="deterministic_batch",
            inference_metrics=InferenceMetrics(0, 0, 0, 0, 0.0),
            sgf=sgf,
        )

    def fake_training(
        config: AppConfig,
        replay_path: Path,
        destination: Path,
        *,
        source_checkpoint: Path | None,
        **kwargs: object,
    ) -> TrainingOperationResult:
        del kwargs
        call_index = len(calls["training"]) + 1
        calls["training"].append((replay_path, destination, source_checkpoint))
        maybe_fail("training")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"candidate-{call_index}".encode())
        metric = TrainingMetrics(
            step=call_index,
            batch_size=config.learner.batch_size,
            policy_loss=1.0,
            value_loss=0.5,
            total_loss=1.5,
            gradient_norm=2.0,
        )
        return TrainingOperationResult(
            checkpoint=destination.resolve(),
            board_size=config.game.board_size,
            replay_size=call_index * 2,
            resumed=True,
            start_step=call_index - 1,
            end_step=call_index,
            metrics=(metric,),
        )

    def fake_arena(
        config: AppConfig,
        *,
        candidate: Path,
        incumbent: Path,
        sgf_output: Path | None = None,
        **kwargs: object,
    ) -> _FakeArenaOperation:
        del kwargs
        call_index = len(calls["arena"])
        calls["arena"].append((candidate, incumbent))
        maybe_fail("arena")
        candidate_step = int(candidate.read_text(encoding="utf-8").split("-")[-1])
        incumbent_step = (
            0
            if incumbent.name == "bootstrap.pt"
            else int(incumbent.read_text(encoding="utf-8").split("-")[-1])
        )
        candidate_identity = CheckpointIdentity(
            candidate.resolve(),
            operations.sha256_file(candidate),
            candidate_step,
        )
        incumbent_identity = CheckpointIdentity(
            incumbent.resolve(),
            operations.sha256_file(incumbent),
            incumbent_step,
        )
        promoted = promotions[min(call_index, len(promotions) - 1)]
        sgf: SgfArtifactIdentity | None = None
        if sgf_output is not None:
            sgf_output.write_bytes(f"arena-sgf-{call_index}".encode())
            sgf = SgfArtifactIdentity(
                sgf_output.resolve(),
                operations.sha256_file(sgf_output),
                config.arena.games,
            )
        return _FakeArenaOperation(
            candidate_identity,
            incumbent_identity,
            promoted=promoted,
            score=0.75 if promoted else 0.25,
            sgf=sgf,
        )

    monkeypatch.setattr(operations, "bootstrap_checkpoint", fake_bootstrap)
    monkeypatch.setattr(operations, "generate_self_play", fake_self_play)
    monkeypatch.setattr(operations, "train_network", fake_training)
    monkeypatch.setattr(operations, "evaluate_checkpoints", fake_arena)
    return calls


def _disable_resume_semantic_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TrainingRunRunner, "_verify_checkpoint", lambda *args: None)
    monkeypatch.setattr(TrainingRunRunner, "_verify_replay", lambda *args: None)
    monkeypatch.setattr(TrainingRunRunner, "_verify_sgf", lambda *args: None)


def test_two_cycle_run_tracks_acceptance_rejection_and_artifact_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_operations(monkeypatch, promotions=(True, False))
    run_directory = tmp_path / "run"

    result = TrainingRunRunner(_config(), run_directory).run()

    assert result.completed_cycles == 2
    assert result.cycles_completed_this_invocation == 2
    assert result.promotion_count == 1
    assert result.rejection_count == 1
    assert result.incumbent == run_directory / "cycles" / "000001" / "candidate.pt"
    assert result.replay == run_directory / "replays" / "replay-1.npz"
    training_calls = calls["training"]
    assert cast("tuple[Path, Path, Path]", training_calls[0])[2].name == "bootstrap.pt"
    assert cast("tuple[Path, Path, Path]", training_calls[1])[2] == result.incumbent
    self_play_calls = calls["self_play"]
    assert cast("tuple[Path, Path, Path | None, int, int]", self_play_calls[0])[2] is None
    assert cast("tuple[Path, Path, Path | None, int, int]", self_play_calls[1])[2] == (
        run_directory / "replays" / "replay-0.npz"
    )
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["incumbent"]["path"] == "cycles/000001/candidate.pt"
    assert manifest["active_replay"]["slot"] == 1
    assert [cycle["promoted"] for cycle in manifest["cycles"]] == [True, False]
    assert not any(str(run_directory) in path for path in _manifest_paths(manifest))


def test_rejected_candidate_is_not_parent_of_next_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_operations(monkeypatch, promotions=(False, False))
    run_directory = tmp_path / "run"

    result = TrainingRunRunner(_config(), run_directory).run()

    sources = [
        cast("tuple[Path, Path, Path]", call)[2]
        for call in calls["training"]
    ]
    assert sources == [run_directory / "bootstrap.pt", run_directory / "bootstrap.pt"]
    assert result.incumbent == run_directory / "bootstrap.pt"
    assert result.rejection_count == 2


@pytest.mark.parametrize(
    ("failed_stage", "expected_calls"),
    [
        ("self_play", (2, 1, 1)),
        ("training", (1, 2, 1)),
        ("arena", (1, 1, 2)),
    ],
)
def test_operation_interruption_resumes_from_last_committed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    expected_calls: tuple[int, int, int],
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    calls = _install_fake_operations(
        monkeypatch,
        promotions=(True,),
        fail_once=failed_stage,
    )
    run_directory = tmp_path / "run"
    runner = TrainingRunRunner(_config(cycles=1), run_directory)

    with pytest.raises(TrainingRunError, match="interruption"):
        runner.run()
    resumed = TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    assert resumed.completed_cycles == 1
    assert resumed.cycles_completed_this_invocation == 1
    assert tuple(len(calls[name]) for name in ("self_play", "training", "arena")) == (
        expected_calls
    )


@pytest.mark.parametrize(
    ("failed_write", "expected_calls"),
    [
        (3, (2, 1, 1)),
        (4, (1, 2, 1)),
        (5, (1, 1, 2)),
    ],
)
def test_manifest_commit_interruption_regenerates_only_uncommitted_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_write: int,
    expected_calls: tuple[int, int, int],
) -> None:
    from azgo import training_run as training_run_module

    _disable_resume_semantic_checks(monkeypatch)
    calls = _install_fake_operations(monkeypatch, promotions=(True, True))
    original_write = training_run_module._write_manifest
    writes = 0

    def fail_one_write(run_directory: Path, manifest: object) -> None:
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("synthetic manifest interruption")
        original_write(run_directory, manifest)  # type: ignore[arg-type]

    monkeypatch.setattr(training_run_module, "_write_manifest", fail_one_write)
    run_directory = tmp_path / "run"

    with pytest.raises(TrainingRunError, match="manifest interruption"):
        TrainingRunRunner(_config(cycles=1), run_directory).run()
    resumed = TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    assert resumed.completed_cycles == 1
    assert tuple(len(calls[name]) for name in ("self_play", "training", "arena")) == (
        expected_calls
    )


def test_completed_resume_is_validated_noop_and_reuses_recorded_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    calls = _install_fake_operations(monkeypatch, promotions=(True,))
    run_directory = tmp_path / "run"
    TrainingRunRunner(_config(cycles=1), run_directory, workers=2).run()
    counts = {name: len(values) for name, values in calls.items()}

    result = TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    assert result.resumed is True
    assert result.start_cycle == 2
    assert result.cycles_completed_this_invocation == 0
    assert result.effective_workers == 2
    assert {name: len(values) for name, values in calls.items()} == counts


def test_resume_rejects_conflicting_workers_and_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    _install_fake_operations(monkeypatch, promotions=(True,))
    run_directory = tmp_path / "run"
    TrainingRunRunner(_config(cycles=1), run_directory, workers=2).run()

    with pytest.raises(TrainingRunError, match="worker override"):
        TrainingRunRunner(_config(cycles=1), run_directory, workers=1).run(resume=True)

    raw = _config(cycles=1).model_dump(mode="json")
    cast("dict[str, object]", raw["learner"])["steps"] = 2
    changed = AppConfig.model_validate(raw)
    with pytest.raises(TrainingRunError, match="configuration"):
        TrainingRunRunner(changed, run_directory).run(resume=True)


def test_resume_rejects_path_escape_and_hash_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    _install_fake_operations(monkeypatch, promotions=(True,))
    run_directory = tmp_path / "run"
    TrainingRunRunner(_config(cycles=1), run_directory).run()
    manifest_path = run_directory / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["bootstrap"]["path"] = "../outside.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TrainingRunError, match="manifest is invalid"):
        TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    manifest_path.write_bytes(original)
    candidate = run_directory / "cycles" / "000001" / "candidate.pt"
    candidate.write_bytes(b"corrupted")

    def verify_hash(runner: TrainingRunRunner, record: object) -> None:
        path = runner._artifact_path(record.path)  # type: ignore[attr-defined]
        if operations.sha256_file(path) != record.sha256:  # type: ignore[attr-defined]
            raise TrainingRunError("checkpoint hash mismatch")

    monkeypatch.setattr(TrainingRunRunner, "_verify_checkpoint", verify_hash)
    with pytest.raises(TrainingRunError, match="checkpoint hash mismatch"):
        TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)


def test_resume_rejects_committed_sgf_hash_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    _install_fake_operations(monkeypatch, promotions=(True,))
    run_directory = tmp_path / "run"
    TrainingRunRunner(_config(cycles=1), run_directory).run()
    sgf = run_directory / "cycles" / "000001" / "self-play.sgf"
    sgf.write_bytes(b"corrupted")

    def verify_hash(runner: TrainingRunRunner, record: object) -> None:
        path = runner._artifact_path(record.path)  # type: ignore[attr-defined]
        if operations.sha256_file(path) != record.sha256:  # type: ignore[attr-defined]
            raise TrainingRunError("SGF hash mismatch")

    monkeypatch.setattr(TrainingRunRunner, "_verify_sgf", verify_hash)
    with pytest.raises(TrainingRunError, match="SGF hash mismatch"):
        TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)


def test_fresh_and_resume_modes_refuse_wrong_directory_state(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(TrainingRunError, match="already exists"):
        TrainingRunRunner(_config(cycles=1), existing).run()
    with pytest.raises(TrainingRunError, match="requires an existing"):
        TrainingRunRunner(_config(cycles=1), tmp_path / "missing").run(resume=True)


def test_operating_system_lock_rejects_concurrent_run_writer(tmp_path: Path) -> None:
    from azgo.training_run import _RunLock

    run_directory = tmp_path / "run"
    with (
        _RunLock(run_directory),
        pytest.raises(TrainingRunError, match="locked"),
        _RunLock(run_directory),
    ):
        pytest.fail("a second writer must not acquire the same run lock")


def test_real_small_run_completes_and_resume_is_a_validated_noop(tmp_path: Path) -> None:
    config = _config(cycles=1)
    run_directory = tmp_path / "real-run"

    created = TrainingRunRunner(config, run_directory).run()
    resumed = TrainingRunRunner(config, run_directory).run(resume=True)

    assert created.completed_cycles == 1
    assert created.replay_size >= config.learner.batch_size
    assert created.next_game_index >= config.self_play.games
    assert created.incumbent.is_file()
    assert created.replay.is_file()
    self_play_sgf = run_directory / "cycles" / "000001" / "self-play.sgf"
    arena_sgf = run_directory / "cycles" / "000001" / "arena.sgf"
    assert self_play_sgf.is_file()
    assert arena_sgf.is_file()
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
    assert manifest["sgf_start_cycle"] == 1
    assert manifest["cycles"][0]["self_play_sgf"]["path"] == (
        "cycles/000001/self-play.sgf"
    )
    assert manifest["cycles"][0]["arena_sgf"]["path"] == "cycles/000001/arena.sgf"
    assert resumed.cycles_completed_this_invocation == 0
    assert resumed.incumbent_sha256 == created.incumbent_sha256
    assert resumed.replay_sha256 == created.replay_sha256


def test_phase9_manifest_migrates_without_requiring_historical_sgf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    calls = _install_fake_operations(monkeypatch, promotions=(True,))
    run_directory = tmp_path / "legacy"
    TrainingRunRunner(_config(cycles=1), run_directory).run()
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest.pop("sgf_start_cycle")
    cycle = manifest["cycles"][0]
    cycle.pop("self_play_sgf")
    cycle.pop("arena_sgf")
    cycle["self_play_report"].pop("sgf_output", None)
    cycle["self_play_report"].pop("sgf_sha256", None)
    cycle["arena_report"].pop("sgf_output", None)
    cycle["arena_report"].pop("sgf_sha256", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("self-play.sgf", "arena.sgf"):
        (run_directory / "cycles" / "000001" / name).unlink()
    counts = {name: len(values) for name, values in calls.items()}

    result = TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    assert result.cycles_completed_this_invocation == 0
    assert {name: len(values) for name, values in calls.items()} == counts
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["format_version"] == 2
    assert migrated["sgf_start_cycle"] == 2
    assert migrated["cycles"][0]["self_play_sgf"] is None
    assert migrated["cycles"][0]["arena_sgf"] is None


def test_phase9_mid_cycle_migration_starts_recording_after_unrecoverable_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_resume_semantic_checks(monkeypatch)
    _install_fake_operations(monkeypatch, promotions=(True,), fail_once="training")
    run_directory = tmp_path / "legacy-interrupted"
    with pytest.raises(TrainingRunError, match="interruption"):
        TrainingRunRunner(_config(cycles=1), run_directory).run()

    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest.pop("sgf_start_cycle")
    progress = manifest["in_progress"]
    progress.pop("self_play_sgf")
    progress["self_play_report"].pop("sgf_output", None)
    progress["self_play_report"].pop("sgf_sha256", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_directory / "cycles" / "000001" / "self-play.sgf").unlink()

    result = TrainingRunRunner(_config(cycles=1), run_directory).run(resume=True)

    assert result.completed_cycles == 1
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["sgf_start_cycle"] == 2
    assert migrated["cycles"][0]["self_play_sgf"] is None
    assert migrated["cycles"][0]["arena_sgf"] is None


def _manifest_paths(value: object) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "path" and isinstance(item, str):
                paths.append(item)
            else:
                paths.extend(_manifest_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_manifest_paths(item))
    return paths
