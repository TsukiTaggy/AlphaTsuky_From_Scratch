"""Crash-safe, deterministic orchestration for bounded AlphaZero training runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO, Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from azgo import operations
from azgo.checkpoint import load_checkpoint
from azgo.config import AppConfig
from azgo.game import Rules, Ruleset
from azgo.replay import ReplayBuffer
from azgo.sgf import load_sgf_collection

if TYPE_CHECKING:
    from azgo.operations import ArenaOperationResult

_MANIFEST_VERSION: Literal[2] = 2
_MANIFEST_NAME = "manifest.json"
_HEX_LENGTH = 64


class TrainingRunError(ValueError):
    """Raised when a managed training run cannot progress safely."""


def _validate_json_mapping(value: dict[str, object], name: str) -> dict[str, object]:
    def validate(item: object, location: str) -> None:
        if item is None or type(item) in {bool, int, str}:
            return
        if type(item) is float:
            if not isfinite(item):
                raise ValueError(f"{location} must not contain non-finite numbers")
            return
        if type(item) is list:
            for index, child in enumerate(cast("list[object]", item)):
                validate(child, f"{location}[{index}]")
            return
        if type(item) is dict:
            mapping = cast("dict[object, object]", item)
            if any(type(key) is not str for key in mapping):
                raise ValueError(f"{location} object keys must be strings")
            for key, child in mapping.items():
                validate(child, f"{location}.{key}")
            return
        raise ValueError(f"{location} contains a non-JSON value")

    validate(value, name)
    return value


class _ArtifactRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str
    sha256: str = Field(min_length=_HEX_LENGTH, max_length=_HEX_LENGTH)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if type(value) is not str or not value:
            raise ValueError("artifact path must be a nonempty relative POSIX path")
        path = PurePosixPath(value)
        if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
            raise ValueError("artifact path must be a contained relative POSIX path")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("artifact path must not contain empty or current components")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if type(value) is not str or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("artifact sha256 must be 64 lowercase hexadecimal characters")
        return value


class _CheckpointRecord(_ArtifactRecord):
    step: int = Field(strict=True, ge=0)


class _ReplayRecord(_ArtifactRecord):
    slot: Literal[0, 1]
    size: int = Field(strict=True, ge=0)
    next_game_index: int = Field(strict=True, ge=0)


class _SgfRecord(_ArtifactRecord):
    games: int = Field(strict=True, ge=1)


class _InProgressRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cycle: int = Field(strict=True, ge=1)
    stage: Literal["self_play", "training", "arena"]
    replay: _ReplayRecord | None = None
    candidate: _CheckpointRecord | None = None
    self_play_sgf: _SgfRecord | None = None
    self_play_report: dict[str, object] | None = None
    training_report: dict[str, object] | None = None

    @field_validator("self_play_report", "training_report")
    @classmethod
    def validate_reports(
        cls,
        value: dict[str, object] | None,
    ) -> dict[str, object] | None:
        return None if value is None else _validate_json_mapping(value, "stage report")

    @model_validator(mode="after")
    def validate_stage_payload(self) -> Self:
        if self.stage == "self_play" and any(
            value is not None
            for value in (
                self.replay,
                self.candidate,
                self.self_play_sgf,
                self.self_play_report,
                self.training_report,
            )
        ):
            raise ValueError("self_play stage cannot contain completed stage outputs")
        if self.stage == "training":
            if self.replay is None or self.self_play_report is None:
                raise ValueError("training stage requires committed self-play output")
            if self.candidate is not None or self.training_report is not None:
                raise ValueError("training stage cannot contain candidate output")
        if self.stage == "arena" and any(
            value is None
            for value in (
                self.replay,
                self.candidate,
                self.self_play_report,
                self.training_report,
            )
        ):
            raise ValueError("arena stage requires committed replay and candidate outputs")
        return self


class _CycleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cycle: int = Field(strict=True, ge=1)
    candidate: _CheckpointRecord
    replay_sha256: str = Field(min_length=_HEX_LENGTH, max_length=_HEX_LENGTH)
    replay_size: int = Field(strict=True, ge=0)
    next_game_index: int = Field(strict=True, ge=0)
    incumbent_before_sha256: str = Field(min_length=_HEX_LENGTH, max_length=_HEX_LENGTH)
    incumbent_after_sha256: str = Field(min_length=_HEX_LENGTH, max_length=_HEX_LENGTH)
    promoted: bool
    candidate_score: float
    self_play_report: dict[str, object]
    training_report: dict[str, object]
    arena_report: dict[str, object]
    self_play_sgf: _SgfRecord | None = None
    arena_sgf: _SgfRecord | None = None

    @field_validator("self_play_report", "training_report", "arena_report")
    @classmethod
    def validate_reports(cls, value: dict[str, object]) -> dict[str, object]:
        return _validate_json_mapping(value, "cycle report")

    @field_validator(
        "replay_sha256",
        "incumbent_before_sha256",
        "incumbent_after_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str) -> str:
        return _ArtifactRecord.validate_sha256(value)

    @field_validator("candidate_score")
    @classmethod
    def validate_candidate_score(cls, value: float) -> float:
        if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("candidate_score must be a finite float in [0, 1]")
        return value


class _RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format_version: Literal[2]
    settings: dict[str, object]
    config_sha256: str = Field(min_length=_HEX_LENGTH, max_length=_HEX_LENGTH)
    target_cycles: int = Field(strict=True, ge=1)
    sgf_start_cycle: int = Field(strict=True, ge=1)
    effective_workers: int = Field(strict=True, ge=1)
    bootstrap: _CheckpointRecord
    incumbent: _CheckpointRecord
    active_replay: _ReplayRecord | None
    cycles: tuple[_CycleRecord, ...]
    in_progress: _InProgressRecord | None

    @field_validator("settings")
    @classmethod
    def validate_settings(cls, value: dict[str, object]) -> dict[str, object]:
        return _validate_json_mapping(value, "settings")

    @field_validator("config_sha256")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        return _ArtifactRecord.validate_sha256(value)

    @model_validator(mode="after")
    def validate_progression(self) -> Self:
        if len(self.cycles) > self.target_cycles:
            raise ValueError("completed cycles cannot exceed target_cycles")
        if self.sgf_start_cycle > self.target_cycles + 1:
            raise ValueError("sgf_start_cycle cannot exceed target_cycles + 1")
        current = self.bootstrap
        for expected_cycle, cycle in enumerate(self.cycles, start=1):
            if cycle.cycle != expected_cycle:
                raise ValueError("completed cycles must be contiguous and one-based")
            if cycle.incumbent_before_sha256 != current.sha256:
                raise ValueError("cycle incumbent lineage is inconsistent")
            current = cycle.candidate if cycle.promoted else current
            if cycle.incumbent_after_sha256 != current.sha256:
                raise ValueError("cycle incumbent result is inconsistent")
            records = (cycle.self_play_sgf, cycle.arena_sgf)
            if expected_cycle >= self.sgf_start_cycle and any(
                record is None for record in records
            ):
                raise ValueError("recorded cycles require self-play and arena SGF artifacts")
            if expected_cycle < self.sgf_start_cycle and any(
                record is not None for record in records
            ):
                raise ValueError("legacy cycles cannot contain SGF artifacts")
        if self.incumbent != current:
            raise ValueError("current incumbent does not match completed cycle lineage")
        if self.cycles and self.active_replay is None:
            raise ValueError("completed cycles require an active replay artifact")
        if self.in_progress is not None:
            if self.in_progress.cycle != len(self.cycles) + 1:
                raise ValueError("in-progress cycle must follow completed cycles")
            if self.in_progress.cycle > self.target_cycles:
                raise ValueError("in-progress cycle cannot exceed target_cycles")
            requires_sgf = self.in_progress.cycle >= self.sgf_start_cycle
            has_committed_self_play = self.in_progress.stage in {"training", "arena"}
            if requires_sgf and has_committed_self_play:
                if self.in_progress.self_play_sgf is None:
                    raise ValueError("recorded self-play stages require an SGF artifact")
            elif self.in_progress.self_play_sgf is not None:
                raise ValueError("this in-progress cycle cannot contain an SGF artifact")
        return self


@dataclass(frozen=True, slots=True)
class TrainingCycleResult:
    """Compact immutable evidence from one completed managed cycle."""

    cycle: int
    promoted: bool
    candidate_score: float
    candidate_step: int
    replay_size: int
    next_game_index: int


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Final state and invocation delta for one managed training run."""

    run_directory: Path
    resumed: bool
    target_cycles: int
    start_cycle: int
    completed_cycles: int
    cycles_completed_this_invocation: int
    effective_workers: int
    incumbent: Path
    incumbent_sha256: str
    incumbent_step: int
    replay: Path
    replay_sha256: str
    replay_size: int
    next_game_index: int
    promotion_count: int
    rejection_count: int
    cycles: tuple[TrainingCycleResult, ...]

    def report(self) -> dict[str, object]:
        """Return a stable JSON-compatible run summary."""

        return {
            "completed_cycles": self.completed_cycles,
            "current_incumbent": str(self.incumbent),
            "current_replay": str(self.replay),
            "cycles_completed_this_invocation": self.cycles_completed_this_invocation,
            "effective_workers": self.effective_workers,
            "incumbent_sha256": self.incumbent_sha256,
            "incumbent_step": self.incumbent_step,
            "next_game_index": self.next_game_index,
            "promotion_count": self.promotion_count,
            "rejection_count": self.rejection_count,
            "replay_sha256": self.replay_sha256,
            "replay_size": self.replay_size,
            "resumed": self.resumed,
            "run_directory": str(self.run_directory),
            "start_cycle": self.start_cycle,
            "target_cycles": self.target_cycles,
        }


class TrainingRunRunner:
    """Execute or resume an atomically journaled local AlphaZero training run."""

    def __init__(
        self,
        config: AppConfig,
        run_directory: str | Path,
        *,
        workers: int | None = None,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        if not isinstance(run_directory, (str, Path)):
            raise TypeError("run_directory must be a string or pathlib.Path")
        self._config = config
        self._run_directory = Path(run_directory).expanduser().resolve()
        self._workers_supplied = workers is not None
        self._worker_override = _validate_workers(workers, config)

    def run(self, *, resume: bool = False) -> TrainingRunResult:
        """Create or resume the run until its immutable target cycle count."""

        if type(resume) is not bool:
            raise TrainingRunError("resume must be a boolean")
        self._run_directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _RunLock(self._run_directory):
                manifest = self._load_resume() if resume else self._initialize()
                initial_completed = len(manifest.cycles)
                start_cycle = initial_completed + 1
                while len(manifest.cycles) < manifest.target_cycles:
                    manifest = self._run_next_stage(manifest)
                return self._result(
                    manifest,
                    resumed=resume,
                    start_cycle=start_cycle,
                    initial_completed=initial_completed,
                )
        except TrainingRunError:
            raise
        except (OSError, ValueError) as exc:
            raise TrainingRunError(f"training run failed: {exc}") from exc
        except Exception as exc:
            raise TrainingRunError(f"training run failed unexpectedly: {exc}") from exc

    def _initialize(self) -> _RunManifest:
        if self._run_directory.exists():
            raise TrainingRunError(
                "fresh run directory already exists; use --resume for an existing run"
            )
        staging = Path(
            tempfile.mkdtemp(
                dir=self._run_directory.parent,
                prefix=f".{self._run_directory.name}.",
                suffix=".tmp",
            )
        )
        moved = False
        try:
            bootstrap_path = staging / "bootstrap.pt"
            identity = operations.bootstrap_checkpoint(self._config, bootstrap_path)
            checkpoint = _CheckpointRecord(
                path="bootstrap.pt",
                sha256=identity.sha256,
                step=identity.step,
            )
            settings = cast(
                "dict[str, object]",
                self._config.model_dump(mode="json"),
            )
            manifest = _RunManifest(
                format_version=_MANIFEST_VERSION,
                settings=settings,
                config_sha256=_config_sha256(self._config),
                target_cycles=self._config.training_run.cycles,
                sgf_start_cycle=1,
                effective_workers=self._worker_override,
                bootstrap=checkpoint,
                incumbent=checkpoint,
                active_replay=None,
                cycles=(),
                in_progress=None,
            )
            _write_manifest(staging, manifest)
            os.replace(staging, self._run_directory)  # noqa: PTH105
            moved = True
            return manifest
        finally:
            if not moved:
                with suppress(OSError):
                    shutil.rmtree(staging)

    def _load_resume(self) -> _RunManifest:
        if not self._run_directory.is_dir():
            raise TrainingRunError("--resume requires an existing run directory")
        manifest = _read_manifest(self._run_directory)
        expected_settings = self._config.model_dump(mode="json")
        if manifest.settings != expected_settings:
            raise TrainingRunError("run configuration does not match the manifest")
        if manifest.config_sha256 != _config_sha256(self._config):
            raise TrainingRunError("run configuration hash does not match the manifest")
        if manifest.target_cycles != self._config.training_run.cycles:
            raise TrainingRunError("training_run.cycles cannot change after initialization")
        if manifest.effective_workers > self._config.self_play.games:
            raise TrainingRunError("manifest worker count exceeds self_play.games")
        if self._workers_supplied and self._worker_override != manifest.effective_workers:
            raise TrainingRunError("worker override does not match the initialized run")
        if not self._workers_supplied:
            self._worker_override = manifest.effective_workers
        self._verify_manifest_artifacts(manifest)
        _write_manifest(self._run_directory, manifest)
        return manifest

    def _verify_manifest_artifacts(self, manifest: _RunManifest) -> None:
        checkpoints = [manifest.bootstrap, *[cycle.candidate for cycle in manifest.cycles]]
        if manifest.in_progress is not None and manifest.in_progress.candidate is not None:
            checkpoints.append(manifest.in_progress.candidate)
        seen: set[tuple[str, str]] = set()
        for checkpoint in checkpoints:
            identity = (checkpoint.path, checkpoint.sha256)
            if identity in seen:
                continue
            seen.add(identity)
            self._verify_checkpoint(checkpoint)

        replays = [manifest.active_replay]
        if manifest.in_progress is not None:
            replays.append(manifest.in_progress.replay)
        replay_seen: set[tuple[str, str]] = set()
        for replay in replays:
            if replay is None:
                continue
            identity = (replay.path, replay.sha256)
            if identity in replay_seen:
                continue
            replay_seen.add(identity)
            self._verify_replay(replay)

        sgf_records = [
            record
            for cycle in manifest.cycles
            for record in (cycle.self_play_sgf, cycle.arena_sgf)
            if record is not None
        ]
        if (
            manifest.in_progress is not None
            and manifest.in_progress.self_play_sgf is not None
        ):
            sgf_records.append(manifest.in_progress.self_play_sgf)
        sgf_seen: set[tuple[str, str]] = set()
        for record in sgf_records:
            identity = (record.path, record.sha256)
            if identity in sgf_seen:
                continue
            sgf_seen.add(identity)
            self._verify_sgf(record)

    def _verify_checkpoint(self, record: _CheckpointRecord) -> None:
        path = self._artifact_path(record.path)
        if operations.sha256_file(path) != record.sha256:
            raise TrainingRunError(f"checkpoint hash mismatch: {record.path}")
        original_rng = torch.get_rng_state().clone()
        try:
            network = operations.build_network(self._config)
            metadata = load_checkpoint(
                path,
                network=network,
                config=self._config,
                optimizer=None,
                restore_rng=False,
            )
        finally:
            torch.set_rng_state(original_rng)
        if metadata.step != record.step:
            raise TrainingRunError(f"checkpoint step mismatch: {record.path}")

    def _verify_replay(self, record: _ReplayRecord) -> None:
        path = self._artifact_path(record.path)
        if operations.sha256_file(path) != record.sha256:
            raise TrainingRunError(f"replay hash mismatch: {record.path}")
        replay = ReplayBuffer.load(path)
        expected = (
            self._config.game.board_size,
            self._config.model.history_length,
            self._config.replay.capacity,
            record.size,
            record.next_game_index,
        )
        actual = (
            replay.board_size,
            replay.history_length,
            replay.capacity,
            len(replay),
            replay.next_game_index,
        )
        if actual != expected:
            raise TrainingRunError(f"replay metadata mismatch: {record.path}")

    def _verify_sgf(self, record: _SgfRecord) -> None:
        path = self._artifact_path(record.path)
        if operations.sha256_file(path) != record.sha256:
            raise TrainingRunError(f"SGF hash mismatch: {record.path}")
        rules = Rules(
            board_size=self._config.game.board_size,
            komi=self._config.game.komi,
            ruleset=Ruleset(self._config.game.rules.ruleset),
        )
        games = load_sgf_collection(
            path,
            expected_rules=rules,
            zobrist_seed=self._config.zobrist.seed,
        )
        if len(games) != record.games:
            raise TrainingRunError(f"SGF game count mismatch: {record.path}")

    def _run_next_stage(self, manifest: _RunManifest) -> _RunManifest:
        if manifest.in_progress is None:
            cycle = len(manifest.cycles) + 1
            (self._run_directory / "cycles" / f"{cycle:06d}").mkdir(
                parents=True,
                exist_ok=True,
            )
            manifest = _updated_manifest(
                manifest,
                in_progress=_InProgressRecord(cycle=cycle, stage="self_play"),
            )
            _write_manifest(self._run_directory, manifest)

        progress = manifest.in_progress
        if progress is None:
            raise TrainingRunError("manifest lost its in-progress cycle")
        if progress.stage == "self_play":
            return self._run_self_play_stage(manifest, progress)
        if progress.stage == "training":
            return self._run_training_stage(manifest, progress)
        return self._run_arena_stage(manifest, progress)

    def _run_self_play_stage(
        self,
        manifest: _RunManifest,
        progress: _InProgressRecord,
    ) -> _RunManifest:
        active = manifest.active_replay
        slot: Literal[0, 1] = 0 if active is None or active.slot == 1 else 1
        relative = f"replays/replay-{slot}.npz"
        output = self._artifact_path(relative)
        output.parent.mkdir(parents=True, exist_ok=True)
        base = None if active is None else self._artifact_path(active.path)
        incumbent = self._artifact_path(manifest.incumbent.path)
        sgf_relative = f"cycles/{progress.cycle:06d}/self-play.sgf"
        sgf_output = (
            self._artifact_path(sgf_relative)
            if progress.cycle >= manifest.sgf_start_cycle
            else None
        )
        result = operations.generate_self_play(
            self._config,
            output,
            checkpoint=incumbent,
            base_replay=base,
            workers=manifest.effective_workers,
            minimum_positions=self._config.learner.batch_size,
            sgf_output=sgf_output,
        )
        replay = _ReplayRecord(
            path=relative,
            sha256=operations.sha256_file(output),
            slot=slot,
            size=result.replay_size,
            next_game_index=result.next_game_index,
        )
        report = result.report()
        report["output"] = relative
        self_play_sgf: _SgfRecord | None = None
        if sgf_output is not None:
            if result.sgf is None:
                raise TrainingRunError("self-play operation did not return its SGF artifact")
            self_play_sgf = _SgfRecord(
                path=sgf_relative,
                sha256=result.sgf.sha256,
                games=result.sgf.games,
            )
            report["sgf_output"] = sgf_relative
        updated_progress = _InProgressRecord(
            cycle=progress.cycle,
            stage="training",
            replay=replay,
            self_play_sgf=self_play_sgf,
            self_play_report=report,
        )
        updated = _updated_manifest(manifest, in_progress=updated_progress)
        _write_manifest(self._run_directory, updated)
        return updated

    def _run_training_stage(
        self,
        manifest: _RunManifest,
        progress: _InProgressRecord,
    ) -> _RunManifest:
        if progress.replay is None or progress.self_play_report is None:
            raise TrainingRunError("training stage is missing committed self-play state")
        relative = f"cycles/{progress.cycle:06d}/candidate.pt"
        destination = self._artifact_path(relative)
        replay = self._artifact_path(progress.replay.path)
        incumbent = self._artifact_path(manifest.incumbent.path)
        result = operations.train_network(
            self._config,
            replay,
            destination,
            source_checkpoint=incumbent,
        )
        candidate = _CheckpointRecord(
            path=relative,
            sha256=operations.sha256_file(destination),
            step=result.end_step,
        )
        report = result.report()
        report["checkpoint"] = relative
        updated_progress = _InProgressRecord(
            cycle=progress.cycle,
            stage="arena",
            replay=progress.replay,
            candidate=candidate,
            self_play_sgf=progress.self_play_sgf,
            self_play_report=progress.self_play_report,
            training_report=report,
        )
        updated = _updated_manifest(manifest, in_progress=updated_progress)
        _write_manifest(self._run_directory, updated)
        return updated

    def _run_arena_stage(
        self,
        manifest: _RunManifest,
        progress: _InProgressRecord,
    ) -> _RunManifest:
        if (
            progress.replay is None
            or progress.candidate is None
            or progress.self_play_report is None
            or progress.training_report is None
        ):
            raise TrainingRunError("arena stage is missing committed candidate state")
        candidate_path = self._artifact_path(progress.candidate.path)
        incumbent_path = self._artifact_path(manifest.incumbent.path)
        sgf_relative = f"cycles/{progress.cycle:06d}/arena.sgf"
        sgf_output = (
            self._artifact_path(sgf_relative)
            if progress.cycle >= manifest.sgf_start_cycle
            else None
        )
        evaluation = operations.evaluate_checkpoints(
            self._config,
            candidate=candidate_path,
            incumbent=incumbent_path,
            sgf_output=sgf_output,
        )
        if (
            evaluation.candidate.sha256 != progress.candidate.sha256
            or evaluation.candidate.step != progress.candidate.step
        ):
            raise TrainingRunError("candidate identity changed before arena completion")
        if (
            evaluation.incumbent.sha256 != manifest.incumbent.sha256
            or evaluation.incumbent.step != manifest.incumbent.step
        ):
            raise TrainingRunError("incumbent identity changed before arena completion")
        promoted = evaluation.arena.promotion_eligible
        incumbent = progress.candidate if promoted else manifest.incumbent
        arena_report = _relative_arena_report(
            evaluation,
            progress.candidate.path,
            manifest.incumbent.path,
        )
        arena_sgf: _SgfRecord | None = None
        if sgf_output is not None:
            if evaluation.sgf is None:
                raise TrainingRunError("arena operation did not return its SGF artifact")
            arena_sgf = _SgfRecord(
                path=sgf_relative,
                sha256=evaluation.sgf.sha256,
                games=evaluation.sgf.games,
            )
            arena_report["sgf_output"] = sgf_relative
        cycle = _CycleRecord(
            cycle=progress.cycle,
            candidate=progress.candidate,
            replay_sha256=progress.replay.sha256,
            replay_size=progress.replay.size,
            next_game_index=progress.replay.next_game_index,
            incumbent_before_sha256=manifest.incumbent.sha256,
            incumbent_after_sha256=incumbent.sha256,
            promoted=promoted,
            candidate_score=float(evaluation.arena.candidate_score),
            self_play_report=progress.self_play_report,
            training_report=progress.training_report,
            arena_report=arena_report,
            self_play_sgf=progress.self_play_sgf,
            arena_sgf=arena_sgf,
        )
        updated = _updated_manifest(
            manifest,
            incumbent=incumbent,
            active_replay=progress.replay,
            cycles=(*manifest.cycles, cycle),
            in_progress=None,
        )
        _write_manifest(self._run_directory, updated)
        return updated

    def _artifact_path(self, relative: str) -> Path:
        try:
            validated = _ArtifactRecord(path=relative, sha256="0" * _HEX_LENGTH).path
        except ValidationError as exc:
            raise TrainingRunError(f"invalid artifact path in manifest: {relative!r}") from exc
        path = (self._run_directory / Path(PurePosixPath(validated))).resolve()
        if not path.is_relative_to(self._run_directory):
            raise TrainingRunError(f"artifact path escapes run directory: {relative}")
        return path

    def _result(
        self,
        manifest: _RunManifest,
        *,
        resumed: bool,
        start_cycle: int,
        initial_completed: int,
    ) -> TrainingRunResult:
        replay = manifest.active_replay
        if replay is None:
            raise TrainingRunError("completed training run has no replay artifact")
        cycles = tuple(
            TrainingCycleResult(
                cycle=cycle.cycle,
                promoted=cycle.promoted,
                candidate_score=cycle.candidate_score,
                candidate_step=cycle.candidate.step,
                replay_size=cycle.replay_size,
                next_game_index=cycle.next_game_index,
            )
            for cycle in manifest.cycles
        )
        promotions = sum(cycle.promoted for cycle in manifest.cycles)
        completed = len(manifest.cycles)
        return TrainingRunResult(
            run_directory=self._run_directory,
            resumed=resumed,
            target_cycles=manifest.target_cycles,
            start_cycle=start_cycle,
            completed_cycles=completed,
            cycles_completed_this_invocation=completed - initial_completed,
            effective_workers=manifest.effective_workers,
            incumbent=self._artifact_path(manifest.incumbent.path),
            incumbent_sha256=manifest.incumbent.sha256,
            incumbent_step=manifest.incumbent.step,
            replay=self._artifact_path(replay.path),
            replay_sha256=replay.sha256,
            replay_size=replay.size,
            next_game_index=replay.next_game_index,
            promotion_count=promotions,
            rejection_count=completed - promotions,
            cycles=cycles,
        )


class _RunLock:
    """One-byte operating-system lock that is released automatically on process exit."""

    def __init__(self, run_directory: Path) -> None:
        self._path = run_directory.parent / f".{run_directory.name}.lock"
        self._stream: BinaryIO | None = None

    def __enter__(self) -> Self:
        stream = self._path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, PermissionError) as exc:
            stream.close()
            raise TrainingRunError("training run directory is locked by another process") from exc
        self._stream = stream
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc_type, exc_value, traceback
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                descriptor = stream.fileno()
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl = importlib.import_module("fcntl")
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                stream.close()
        return False


def _validate_workers(workers: int | None, config: AppConfig) -> int:
    effective = config.self_play.workers if workers is None else workers
    if type(effective) is not int or effective <= 0:
        raise TrainingRunError("workers must be a positive integer")
    if effective > config.self_play.games:
        raise TrainingRunError("workers must be no greater than self_play.games")
    return effective


def _config_sha256(config: AppConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _updated_manifest(manifest: _RunManifest, **changes: object) -> _RunManifest:
    payload = manifest.model_dump(mode="python")
    payload.update(changes)
    try:
        return _RunManifest.model_validate(payload)
    except ValidationError as exc:
        raise TrainingRunError(f"could not construct valid run manifest: {exc}") from exc


def _write_manifest(run_directory: Path, manifest: _RunManifest) -> None:
    target = run_directory / _MANIFEST_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                manifest.model_dump(mode="json"),
                stream,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)  # noqa: PTH105
        temporary = None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def _read_manifest(run_directory: Path) -> _RunManifest:
    source = run_directory / _MANIFEST_NAME
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if type(payload) is not dict:
            raise TrainingRunError("run manifest root must be an object")
        mapping = cast("dict[str, object]", payload)
        if type(mapping.get("format_version")) is int and mapping["format_version"] == 1:
            mapping = _migrate_phase9_manifest(mapping)
        return _RunManifest.model_validate_json(
            json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )
    except FileNotFoundError as exc:
        raise TrainingRunError("run manifest is missing") from exc
    except TrainingRunError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValidationError) as exc:
        raise TrainingRunError(f"run manifest is invalid: {exc}") from exc


def _migrate_phase9_manifest(payload: dict[str, object]) -> dict[str, object]:
    """Normalize one strict Phase 9 manifest into the Phase 10 SGF schema."""

    cycles_value = payload.get("cycles")
    if type(cycles_value) is not list:
        raise TrainingRunError("legacy run manifest cycles must be an array")
    cycles = cast("list[object]", cycles_value)
    for value in cycles:
        if type(value) is not dict:
            raise TrainingRunError("legacy run manifest cycle must be an object")
        cycle = cast("dict[str, object]", value)
        cycle["self_play_sgf"] = None
        cycle["arena_sgf"] = None

    start_cycle = len(cycles) + 1
    progress_value = payload.get("in_progress")
    if progress_value is not None:
        if type(progress_value) is not dict:
            raise TrainingRunError("legacy in-progress state must be an object")
        progress = cast("dict[str, object]", progress_value)
        stage = progress.get("stage")
        if stage in {"training", "arena"}:
            start_cycle += 1
        progress["self_play_sgf"] = None

    payload["format_version"] = _MANIFEST_VERSION
    payload["sgf_start_cycle"] = start_cycle
    return payload


def _relative_arena_report(
    evaluation: ArenaOperationResult,
    candidate: str,
    incumbent: str,
) -> dict[str, object]:
    report = evaluation.report()
    report["candidate"] = candidate
    report["incumbent"] = incumbent
    return report


__all__ = [
    "TrainingCycleResult",
    "TrainingRunError",
    "TrainingRunResult",
    "TrainingRunRunner",
]
