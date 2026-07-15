"""Typed, reusable operations shared by the CLI and managed training runs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from azgo import arena as arena_module
from azgo import checkpoint as checkpoint_module
from azgo import evaluator as evaluator_module
from azgo import learner as learner_module
from azgo.config import AppConfig
from azgo.game import Color, Rules, Ruleset
from azgo.inference import InferenceMetrics
from azgo.learner import TrainingError, TrainingMetrics
from azgo.network import PolicyValueNetwork
from azgo.replay import ReplayBuffer, ReplayError
from azgo.self_play import ParallelSelfPlayRunner
from azgo.sgf import SgfGameRecord, save_sgf_collection

if TYPE_CHECKING:
    from collections.abc import Callable

    from azgo.arena import ArenaGameResult, ArenaResult
    from azgo.evaluator import Evaluator
    from azgo.self_play import SelfPlayGame


class OperationError(ValueError):
    """Raised when a reusable workflow operation violates its contract."""


type NetworkFactory = Callable[[AppConfig], PolicyValueNetwork]
type EvaluatorBuilder = Callable[
    [AppConfig, Path | None],
    tuple[Evaluator, str, int | None],
]


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Stable identity and learner step for one checkpoint artifact."""

    path: Path
    sha256: str
    step: int


@dataclass(frozen=True, slots=True)
class SgfArtifactIdentity:
    """Stable identity and game count for one SGF collection artifact."""

    path: Path
    sha256: str
    games: int


@dataclass(frozen=True, slots=True)
class SelfPlayOperationResult:
    """Complete replay update and aggregate self-play measurements."""

    output: Path
    board_size: int
    evaluator: str
    checkpoint_step: int | None
    games_generated: int
    generation_batches: int
    positions_generated: int
    black_wins: int
    white_wins: int
    draws: int
    replay_capacity: int
    replay_size: int
    next_game_index: int
    effective_workers: int
    inference_mode: str
    inference_metrics: InferenceMetrics
    sgf: SgfArtifactIdentity | None = None

    def report(self) -> dict[str, object]:
        """Return the stable JSON-compatible command representation."""

        inference = self.inference_metrics
        report: dict[str, object] = {
            "black_wins": self.black_wins,
            "board_size": self.board_size,
            "checkpoint_step": self.checkpoint_step,
            "draws": self.draws,
            "evaluator": self.evaluator,
            "games_generated": self.games_generated,
            "inference_batches": inference.batches,
            "inference_max_batch_size": inference.max_batch_size,
            "inference_mean_batch_size": inference.mean_batch_size,
            "inference_mode": self.inference_mode,
            "inference_positions": inference.positions,
            "inference_requests": inference.requests,
            "next_game_index": self.next_game_index,
            "output": str(self.output),
            "positions_generated": self.positions_generated,
            "replay_capacity": self.replay_capacity,
            "replay_size": self.replay_size,
            "self_play_workers": self.effective_workers,
            "white_wins": self.white_wins,
        }
        if self.sgf is not None:
            report["sgf_output"] = str(self.sgf.path)
            report["sgf_sha256"] = self.sgf.sha256
        return report


@dataclass(frozen=True, slots=True)
class TrainingOperationResult:
    """Complete deterministic learner update sequence."""

    checkpoint: Path
    board_size: int
    replay_size: int
    resumed: bool
    start_step: int
    end_step: int
    metrics: tuple[TrainingMetrics, ...]

    def report(self) -> dict[str, object]:
        """Return the stable JSON-compatible command representation."""

        if not self.metrics:
            raise OperationError("training result must contain at least one metric")
        final = self.metrics[-1]
        count = len(self.metrics)
        return {
            "board_size": self.board_size,
            "checkpoint": str(self.checkpoint),
            "end_step": self.end_step,
            "final_gradient_norm": final.gradient_norm,
            "final_policy_loss": final.policy_loss,
            "final_total_loss": final.total_loss,
            "final_value_loss": final.value_loss,
            "mean_policy_loss": sum(item.policy_loss for item in self.metrics) / count,
            "mean_total_loss": sum(item.total_loss for item in self.metrics) / count,
            "mean_value_loss": sum(item.value_loss for item in self.metrics) / count,
            "replay_size": self.replay_size,
            "resumed": self.resumed,
            "start_step": self.start_step,
            "steps_completed": self.end_step - self.start_step,
        }


@dataclass(frozen=True, slots=True)
class ArenaOperationResult:
    """Checkpoint identities plus a complete paired arena decision."""

    candidate: CheckpointIdentity
    incumbent: CheckpointIdentity
    arena: ArenaResult
    promotion_requested: bool
    promoted: bool
    promoted_to: Path | None
    sgf: SgfArtifactIdentity | None = None

    def report(self) -> dict[str, object]:
        """Return the stable JSON-compatible command representation."""

        result = self.arena
        report: dict[str, object] = {
            "candidate": str(self.candidate.path),
            "candidate_points": float(result.candidate_points),
            "candidate_score": float(result.candidate_score),
            "candidate_sha256": self.candidate.sha256,
            "candidate_step": self.candidate.step,
            "candidate_wins": int(result.candidate_wins),
            "draws": int(result.draws),
            "games": [arena_game_record(game) for game in result.games],
            "games_played": len(result.games),
            "incumbent": str(self.incumbent.path),
            "incumbent_sha256": self.incumbent.sha256,
            "incumbent_step": self.incumbent.step,
            "incumbent_wins": int(result.incumbent_wins),
            "promoted": self.promoted,
            "promoted_to": str(self.promoted_to) if self.promoted_to is not None else None,
            "promotion_eligible": bool(result.promotion_eligible),
            "promotion_requested": self.promotion_requested,
            "promotion_threshold": float(result.promotion_threshold),
        }
        if self.sgf is not None:
            report["sgf_output"] = str(self.sgf.path)
            report["sgf_sha256"] = self.sgf.sha256
        return report


def build_network(config: AppConfig) -> PolicyValueNetwork:
    """Construct the configured CPU policy-value network."""

    return PolicyValueNetwork(
        board_size=config.game.board_size,
        history_length=config.model.history_length,
        channels=config.model.channels,
        residual_blocks=config.model.residual_blocks,
        value_hidden_size=config.model.value_hidden_size,
    )


def build_evaluator(
    config: AppConfig,
    checkpoint: Path | None,
    *,
    network_factory: NetworkFactory = build_network,
) -> tuple[Evaluator, str, int | None]:
    """Build uniform evaluation or load one trusted compatible checkpoint."""

    if checkpoint is None:
        return evaluator_module.UniformEvaluator(), "uniform", None
    source = checkpoint.expanduser().resolve()
    network = network_factory(config)
    metadata = checkpoint_module.load_checkpoint(
        source,
        network=network,
        config=config,
        optimizer=None,
        restore_rng=False,
    )
    return evaluator_module.TorchEvaluator(network), "checkpoint", metadata.step


def bootstrap_checkpoint(
    config: AppConfig,
    destination: Path,
    *,
    network_factory: NetworkFactory = build_network,
) -> CheckpointIdentity:
    """Create a deterministic seeded step-zero incumbent checkpoint."""

    target = destination.expanduser().resolve()
    torch.manual_seed(config.learner.seed)
    network = network_factory(config)
    learner = learner_module.Learner(network, config)
    checkpoint_module.save_checkpoint(
        target,
        network=network,
        optimizer=learner.optimizer,
        step=0,
        config=config,
    )
    return CheckpointIdentity(target, sha256_file(target), 0)


def generate_self_play(
    config: AppConfig,
    output: Path,
    *,
    checkpoint: Path | None,
    base_replay: Path | None = None,
    workers: int | None = None,
    minimum_positions: int = 0,
    sgf_output: Path | None = None,
    evaluator_builder: EvaluatorBuilder | None = None,
) -> SelfPlayOperationResult:
    """Generate one or more complete batches and atomically save their replay."""

    if type(minimum_positions) is not int or minimum_positions < 0:
        raise OperationError("minimum_positions must be a nonnegative integer")
    if minimum_positions > config.replay.capacity:
        raise OperationError("minimum_positions cannot exceed replay capacity")

    destination = output.expanduser().resolve()
    sgf_destination = None if sgf_output is None else sgf_output.expanduser().resolve()
    if sgf_destination == destination:
        raise OperationError("SGF output must not resolve to the replay output")
    buffer = (
        _new_replay(config)
        if base_replay is None
        else ReplayBuffer.load(base_replay.expanduser().resolve())
    )
    _validate_replay_metadata(buffer, config, require_capacity=True)
    builder = evaluator_builder or build_evaluator
    evaluator, evaluator_name, checkpoint_step = builder(config, checkpoint)

    games_generated = 0
    generation_batches = 0
    positions_generated = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    requests = 0
    positions = 0
    batches = 0
    maximum = 0
    effective_workers = 0
    inference_mode = "direct"
    completed_games: list[SelfPlayGame] = []

    while generation_batches == 0 or len(buffer) < minimum_positions:
        result = ParallelSelfPlayRunner(
            evaluator,
            config,
            workers=workers,
        ).play_games(buffer.next_game_index)
        effective_workers = result.effective_workers
        inference_mode = result.inference_mode
        inference = result.inference_metrics
        requests += inference.requests
        positions += inference.positions
        batches += inference.batches
        maximum = max(maximum, inference.max_batch_size)
        generation_batches += 1
        games_generated += len(result.games)
        completed_games.extend(result.games)
        for game in result.games:
            positions_generated += len(game.samples)
            black_wins += game.winner is Color.BLACK
            white_wins += game.winner is Color.WHITE
            draws += game.winner is None
            buffer.add_game(game)

    mean = positions / batches if batches else 0.0
    metrics = InferenceMetrics(requests, positions, batches, maximum, mean)
    sgf_identity: SgfArtifactIdentity | None = None
    if sgf_destination is not None:
        records = _self_play_sgf_records(
            tuple(completed_games),
            config,
            evaluator_name=evaluator_name,
            checkpoint_step=checkpoint_step,
        )
        save_sgf_collection(sgf_destination, records)
        sgf_identity = SgfArtifactIdentity(
            sgf_destination,
            sha256_file(sgf_destination),
            len(records),
        )
    buffer.save(destination)
    return SelfPlayOperationResult(
        output=destination,
        board_size=config.game.board_size,
        evaluator=evaluator_name,
        checkpoint_step=checkpoint_step,
        games_generated=games_generated,
        generation_batches=generation_batches,
        positions_generated=positions_generated,
        black_wins=black_wins,
        white_wins=white_wins,
        draws=draws,
        replay_capacity=buffer.capacity,
        replay_size=len(buffer),
        next_game_index=buffer.next_game_index,
        effective_workers=effective_workers,
        inference_mode=inference_mode,
        inference_metrics=metrics,
        sgf=sgf_identity,
    )


def train_network(
    config: AppConfig,
    replay_path: Path,
    destination: Path,
    *,
    source_checkpoint: Path | None,
    network_factory: NetworkFactory = build_network,
) -> TrainingOperationResult:
    """Train from scratch or from a distinct source checkpoint into a destination."""

    replay_source = replay_path.expanduser().resolve()
    target = destination.expanduser().resolve()
    replay = ReplayBuffer.load(replay_source)
    _validate_replay_metadata(replay, config, require_capacity=False)
    if len(replay) < config.learner.batch_size:
        raise TrainingError(
            f"replay size {len(replay)} is smaller than batch_size "
            f"{config.learner.batch_size}"
        )

    torch.manual_seed(config.learner.seed)
    network = network_factory(config)
    learner = learner_module.Learner(network, config)
    source = (
        None if source_checkpoint is None else source_checkpoint.expanduser().resolve()
    )
    if source is not None:
        metadata = checkpoint_module.load_checkpoint(
            source,
            network=network,
            config=config,
            optimizer=learner.optimizer,
            restore_rng=True,
        )
        learner.restore_step(metadata.step)

    start_step = learner.step
    metrics: list[TrainingMetrics] = []
    for _ in range(config.learner.steps):
        sample_seed = int(
            np.random.SeedSequence([config.learner.seed, learner.step]).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        batch = replay.sample(
            config.learner.batch_size,
            sample_seed,
            augment=config.learner.augment,
        )
        metric = learner.train_step(batch)
        metrics.append(metric)
        if metric.step % config.learner.checkpoint_interval == 0:
            checkpoint_module.save_checkpoint(
                target,
                network=network,
                optimizer=learner.optimizer,
                step=learner.step,
                config=config,
            )

    checkpoint_module.save_checkpoint(
        target,
        network=network,
        optimizer=learner.optimizer,
        step=learner.step,
        config=config,
    )
    return TrainingOperationResult(
        checkpoint=target,
        board_size=config.game.board_size,
        replay_size=len(replay),
        resumed=source is not None,
        start_step=start_step,
        end_step=learner.step,
        metrics=tuple(metrics),
    )


def evaluate_checkpoints(
    config: AppConfig,
    *,
    candidate: Path,
    incumbent: Path,
    promote_to: Path | None = None,
    sgf_output: Path | None = None,
    evaluator_builder: EvaluatorBuilder | None = None,
) -> ArenaOperationResult:
    """Evaluate immutable checkpoint identities and optionally promote safely."""

    candidate_path = candidate.expanduser().resolve()
    incumbent_path = incumbent.expanduser().resolve()
    destination = None if promote_to is None else promote_to.expanduser().resolve()
    sgf_destination = None if sgf_output is None else sgf_output.expanduser().resolve()
    if candidate_path == incumbent_path:
        raise OperationError(
            "candidate and incumbent must resolve to different checkpoint paths"
        )
    if destination == candidate_path:
        raise OperationError("promotion destination must not resolve to the candidate path")
    protected_paths = {candidate_path, incumbent_path}
    if destination is not None:
        protected_paths.add(destination)
    if sgf_destination in protected_paths:
        raise OperationError("SGF output must not overwrite a checkpoint artifact")

    candidate_sha256 = sha256_file(candidate_path)
    incumbent_sha256 = sha256_file(incumbent_path)
    builder = evaluator_builder or build_evaluator
    candidate_evaluator, _, candidate_step = builder(config, candidate_path)
    incumbent_evaluator, _, incumbent_step = builder(config, incumbent_path)
    if candidate_step is None or incumbent_step is None:
        raise OperationError("arena checkpoints must provide learner steps")
    if sha256_file(candidate_path) != candidate_sha256:
        raise OperationError("candidate checkpoint changed while it was being loaded")
    if sha256_file(incumbent_path) != incumbent_sha256:
        raise OperationError("incumbent checkpoint changed while it was being loaded")

    arena = arena_module.ArenaRunner(
        candidate_evaluator,
        incumbent_evaluator,
        config,
    ).run()
    sgf_identity: SgfArtifactIdentity | None = None
    if sgf_destination is not None:
        records = _arena_sgf_records(
            arena.games,
            config,
            candidate_sha256=candidate_sha256,
            incumbent_sha256=incumbent_sha256,
        )
        save_sgf_collection(sgf_destination, records)
        sgf_identity = SgfArtifactIdentity(
            sgf_destination,
            sha256_file(sgf_destination),
            len(records),
        )
    promoted = False
    if arena.promotion_eligible and destination is not None:
        atomic_promote_checkpoint(
            candidate_path,
            destination,
            expected_sha256=candidate_sha256,
        )
        promoted = True
    return ArenaOperationResult(
        candidate=CheckpointIdentity(candidate_path, candidate_sha256, candidate_step),
        incumbent=CheckpointIdentity(incumbent_path, incumbent_sha256, incumbent_step),
        arena=arena,
        promotion_requested=destination is not None,
        promoted=promoted,
        promoted_to=destination if promoted else None,
        sgf=sgf_identity,
    )


def _self_play_sgf_records(
    games: tuple[SelfPlayGame, ...],
    config: AppConfig,
    *,
    evaluator_name: str,
    checkpoint_step: int | None,
) -> tuple[SgfGameRecord, ...]:
    rules = _rules(config)
    player = (
        evaluator_name
        if checkpoint_step is None
        else f"{evaluator_name}-step-{checkpoint_step}"
    )
    return tuple(
        SgfGameRecord(
            rules=rules,
            actions=game.actions,
            final_score=game.final_score,
            game_name=f"self-play-{game.game_index:020d}",
            black_player=player,
            white_player=player,
        )
        for game in games
    )


def _arena_sgf_records(
    games: tuple[ArenaGameResult, ...],
    config: AppConfig,
    *,
    candidate_sha256: str,
    incumbent_sha256: str,
) -> tuple[SgfGameRecord, ...]:
    rules = _rules(config)
    candidate = f"candidate-{candidate_sha256}"
    incumbent = f"incumbent-{incumbent_sha256}"
    return tuple(
        SgfGameRecord(
            rules=rules,
            actions=game.actions,
            final_score=game.final_score,
            game_name=f"arena-pair-{game.pair_index:06d}-game-{game.game_index:06d}",
            black_player=candidate if game.candidate_color is Color.BLACK else incumbent,
            white_player=candidate if game.candidate_color is Color.WHITE else incumbent,
        )
        for game in games
    )


def arena_game_record(game: ArenaGameResult) -> dict[str, object]:
    """Convert one arena game to its stable compact JSON representation."""

    return {
        "black_score": float(game.final_score.black_score),
        "candidate_color": game.candidate_color.name.lower(),
        "candidate_outcome": game.candidate_outcome,
        "game_index": int(game.game_index),
        "move_count": int(game.move_count),
        "opening_actions": [int(action) for action in game.opening_actions],
        "pair_index": int(game.pair_index),
        "white_score": float(game.final_score.white_score),
        "winner": None if game.winner is None else game.winner.name.lower(),
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one regular artifact file."""

    source = path.expanduser().resolve()
    if not source.is_file():
        raise OperationError(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_promote_checkpoint(
    candidate: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Atomically byte-copy a candidate after confirming its evaluated identity."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        digest = hashlib.sha256()
        with (
            candidate.open("rb") as source,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if digest.hexdigest() != expected_sha256:
            raise OperationError(
                "candidate checkpoint changed after arena evaluation; promotion aborted"
            )
        os.replace(temporary_path, destination)  # noqa: PTH105
        temporary_path = None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


def _new_replay(config: AppConfig) -> ReplayBuffer:
    return ReplayBuffer(
        board_size=config.game.board_size,
        history_length=config.model.history_length,
        capacity=config.replay.capacity,
    )


def _rules(config: AppConfig) -> Rules:
    return Rules(
        board_size=config.game.board_size,
        komi=config.game.komi,
        ruleset=Ruleset(config.game.rules.ruleset),
    )


def _validate_replay_metadata(
    replay: ReplayBuffer,
    config: AppConfig,
    *,
    require_capacity: bool,
) -> None:
    expected = (config.game.board_size, config.model.history_length)
    actual = (replay.board_size, replay.history_length)
    if actual != expected:
        raise ReplayError(
            "replay metadata does not match configuration: expected "
            f"board_size/history_length {expected}, got {actual}"
        )
    if require_capacity and replay.capacity != config.replay.capacity:
        raise ReplayError(
            "replay metadata does not match configuration: expected capacity "
            f"{config.replay.capacity}, got {replay.capacity}"
        )


__all__ = [
    "ArenaOperationResult",
    "CheckpointIdentity",
    "OperationError",
    "SelfPlayOperationResult",
    "SgfArtifactIdentity",
    "TrainingOperationResult",
    "arena_game_record",
    "atomic_promote_checkpoint",
    "bootstrap_checkpoint",
    "build_evaluator",
    "build_network",
    "evaluate_checkpoints",
    "generate_self_play",
    "sha256_file",
    "train_network",
]
