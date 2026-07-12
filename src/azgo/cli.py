"""Command-line tools for configuration and the Go rules engine."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from azgo.config import AppConfig, load_config

if TYPE_CHECKING:
    from azgo.arena import ArenaGameResult
    from azgo.evaluator import Evaluator
    from azgo.network import PolicyValueNetwork

app = typer.Typer(
    name="azgo",
    help="Correctness-first Phase 1-7 tools for the AlphaZero Go project.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

ConfigArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a Phase 1-7 YAML configuration.",
    ),
]
ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a Phase 1-7 YAML configuration.",
    ),
]
MovesOption = Annotated[
    list[int] | None,
    typer.Option(
        "--move",
        "-m",
        help="Row-major action to apply before search; repeat for multiple moves.",
    ),
]
RootNoiseOption = Annotated[
    bool,
    typer.Option(
        "--root-noise/--no-root-noise",
        help="Mix seeded Dirichlet noise into legal root priors.",
    ),
]
OutputOption = Annotated[
    Path,
    typer.Option(
        "--output",
        "-o",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Compressed NPZ replay snapshot to create or append.",
    ),
]
OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite/--no-overwrite",
        help="Replace an existing replay snapshot instead of appending to it.",
    ),
]
CheckpointOption = Annotated[
    Path | None,
    typer.Option(
        "--checkpoint",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional trusted Phase 6 model checkpoint used for evaluation.",
    ),
]
ReplayInputOption = Annotated[
    Path,
    typer.Option(
        "--replay",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Compressed NPZ replay snapshot used for training.",
    ),
]
TrainingCheckpointOption = Annotated[
    Path,
    typer.Option(
        "--checkpoint",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Trusted Phase 6 checkpoint to create, replace, or resume.",
    ),
]
ResumeOption = Annotated[
    bool,
    typer.Option(
        "--resume/--no-resume",
        help="Resume optimizer, step, and random state from an existing checkpoint.",
    ),
]
TrainingOverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite/--no-overwrite",
        help="Start fresh and explicitly replace an existing checkpoint.",
    ),
]
ArenaCandidateOption = Annotated[
    Path,
    typer.Option(
        "--candidate",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Trusted compatible candidate checkpoint to evaluate.",
    ),
]
ArenaIncumbentOption = Annotated[
    Path,
    typer.Option(
        "--incumbent",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Trusted compatible incumbent checkpoint to evaluate.",
    ),
]
PromotionOption = Annotated[
    Path | None,
    typer.Option(
        "--promote-to",
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Atomically copy an eligible candidate to this explicit destination.",
    ),
]


class ArenaCommandError(ValueError):
    """Raised when arena command inputs or promotion safety checks fail."""


def _load_or_exit(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except (
        FileNotFoundError,
        HydraException,
        OmegaConfBaseException,
        ValidationError,
        ValueError,
    ) as exc:
        typer.echo(f"Invalid configuration: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command("validate-config")
def validate_config_command(config: ConfigArgument) -> None:
    """Compose and strictly validate a Phase 1-7 configuration."""

    settings = _load_or_exit(config)
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
    typer.echo(f"Configuration is valid: {config}")


@app.command("benchmark-engine")
def benchmark_engine(config: ConfigOption) -> None:
    """Benchmark deterministic random legal games using the configured engine."""

    settings = _load_or_exit(config)
    typer.echo(json.dumps(_run_engine_benchmark(settings), indent=2, sort_keys=True))


@app.command("search-move")
def search_move(
    config: ConfigOption,
    moves: MovesOption = None,
    root_noise: RootNoiseOption = False,
    checkpoint: CheckpointOption = None,
) -> None:
    """Analyze a reconstructed position with uniform or checkpoint evaluation."""

    settings = _load_or_exit(config)
    try:
        report = _run_search(
            settings,
            () if moves is None else moves,
            root_noise=root_noise,
            checkpoint=checkpoint,
        )
    except _search_failures() as exc:
        typer.echo(f"Search failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("generate-self-play")
def generate_self_play(
    config: ConfigOption,
    output: OutputOption,
    overwrite: OverwriteOption = False,
    checkpoint: CheckpointOption = None,
) -> None:
    """Generate deterministic games into a replay snapshot."""

    settings = _load_or_exit(config)
    try:
        report = _run_self_play(
            settings,
            output,
            overwrite=overwrite,
            checkpoint=checkpoint,
        )
    except _self_play_failures() as exc:
        typer.echo(f"Self-play failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("train-network")
def train_network(
    config: ConfigOption,
    replay: ReplayInputOption,
    checkpoint: TrainingCheckpointOption,
    resume: ResumeOption = False,
    overwrite: TrainingOverwriteOption = False,
) -> None:
    """Train a CPU policy-value network from replay and save a checkpoint."""

    settings = _load_or_exit(config)
    try:
        report = _run_training(
            settings,
            replay,
            checkpoint,
            resume=resume,
            overwrite=overwrite,
        )
    except _training_failures() as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("evaluate-arena")
def evaluate_arena(
    config: ConfigOption,
    candidate: ArenaCandidateOption,
    incumbent: ArenaIncumbentOption,
    promote_to: PromotionOption = None,
) -> None:
    """Evaluate candidate and incumbent checkpoints and optionally promote."""

    settings = _load_or_exit(config)
    try:
        report = _run_arena(
            settings,
            candidate=candidate,
            incumbent=incumbent,
            promote_to=promote_to,
        )
    except _arena_failures() as exc:
        typer.echo(f"Arena evaluation failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


def _run_engine_benchmark(settings: AppConfig) -> dict[str, float | int]:
    """Run the benchmark with a command-local game-engine import."""

    from azgo.game import GameState, Rules, Ruleset

    rules = Rules(
        board_size=settings.game.board_size,
        komi=settings.game.komi,
        ruleset=Ruleset(settings.game.rules.ruleset),
    )
    rng = random.Random(settings.benchmark.seed)

    total_moves = 0
    completed_games = 0
    started = time.perf_counter()
    for _ in range(settings.benchmark.games):
        state = GameState.new(rules, zobrist_seed=settings.zobrist.seed)
        for _ in range(settings.benchmark.max_moves_per_game):
            if state.is_terminal:
                completed_games += 1
                break
            state = state.apply(rng.choice(state.legal_actions()))
            total_moves += 1

    elapsed_seconds = time.perf_counter() - started
    moves_per_second = total_moves / elapsed_seconds if elapsed_seconds else 0.0
    return {
        "board_size": settings.game.board_size,
        "completed_games": completed_games,
        "elapsed_seconds": elapsed_seconds,
        "games_requested": settings.benchmark.games,
        "moves": total_moves,
        "moves_per_second": moves_per_second,
        "seed": settings.benchmark.seed,
    }


def _search_failures() -> tuple[type[Exception], ...]:
    """Return command-local failure types without importing search for other commands."""

    from azgo.checkpoint import CheckpointError
    from azgo.game import GoEngineError
    from azgo.search import SearchError

    return (GoEngineError, SearchError, CheckpointError, OSError, ValueError)


def _build_network(settings: AppConfig) -> PolicyValueNetwork:
    """Construct the configured policy-value architecture on CPU."""

    from azgo.network import PolicyValueNetwork

    return PolicyValueNetwork(
        board_size=settings.game.board_size,
        history_length=settings.model.history_length,
        channels=settings.model.channels,
        residual_blocks=settings.model.residual_blocks,
        value_hidden_size=settings.model.value_hidden_size,
    )


def _build_evaluator(
    settings: AppConfig,
    checkpoint: Path | None,
) -> tuple[Evaluator, str, int | None]:
    """Build a uniform evaluator or load a trusted compatible checkpoint."""

    from azgo.checkpoint import load_checkpoint
    from azgo.evaluator import TorchEvaluator, UniformEvaluator

    if checkpoint is None:
        return UniformEvaluator(), "uniform", None

    network = _build_network(settings)
    metadata = load_checkpoint(
        checkpoint.expanduser().resolve(),
        network=network,
        config=settings,
        optimizer=None,
        restore_rng=False,
    )
    return TorchEvaluator(network), "checkpoint", metadata.step


def _arena_failures() -> tuple[type[Exception], ...]:
    """Return arena command failures without loading arena for other commands."""

    from azgo.arena import ArenaError
    from azgo.checkpoint import CheckpointError

    return (ArenaError, CheckpointError, ArenaCommandError, OSError)


def _run_arena(
    settings: AppConfig,
    *,
    candidate: Path,
    incumbent: Path,
    promote_to: Path | None,
) -> dict[str, object]:
    """Evaluate two immutable checkpoint identities and optionally promote."""

    from azgo.arena import ArenaRunner

    candidate = candidate.expanduser().resolve()
    incumbent = incumbent.expanduser().resolve()
    destination = (
        None if promote_to is None else promote_to.expanduser().resolve()
    )
    if candidate == incumbent:
        raise ArenaCommandError(
            "candidate and incumbent must resolve to different checkpoint paths"
        )
    if destination == candidate:
        raise ArenaCommandError("--promote-to must not resolve to the candidate path")

    candidate_sha256 = _sha256_file(candidate)
    incumbent_sha256 = _sha256_file(incumbent)
    candidate_evaluator, _, candidate_step = _build_evaluator(settings, candidate)
    incumbent_evaluator, _, incumbent_step = _build_evaluator(settings, incumbent)
    if _sha256_file(candidate) != candidate_sha256:
        raise ArenaCommandError("candidate checkpoint changed while it was being loaded")
    if _sha256_file(incumbent) != incumbent_sha256:
        raise ArenaCommandError("incumbent checkpoint changed while it was being loaded")

    result = ArenaRunner(
        candidate_evaluator,
        incumbent_evaluator,
        settings,
    ).run()
    promotion_requested = destination is not None
    promoted = False
    if result.promotion_eligible and destination is not None:
        _atomic_promote_checkpoint(
            candidate,
            destination,
            expected_sha256=candidate_sha256,
        )
        promoted = True

    return {
        "candidate": str(candidate),
        "candidate_points": float(result.candidate_points),
        "candidate_score": float(result.candidate_score),
        "candidate_sha256": candidate_sha256,
        "candidate_step": candidate_step,
        "candidate_wins": int(result.candidate_wins),
        "draws": int(result.draws),
        "games": [_arena_game_record(game) for game in result.games],
        "games_played": len(result.games),
        "incumbent": str(incumbent),
        "incumbent_sha256": incumbent_sha256,
        "incumbent_step": incumbent_step,
        "incumbent_wins": int(result.incumbent_wins),
        "promoted": promoted,
        "promoted_to": str(destination) if promoted else None,
        "promotion_eligible": bool(result.promotion_eligible),
        "promotion_requested": promotion_requested,
        "promotion_threshold": float(result.promotion_threshold),
    }


def _arena_game_record(game: ArenaGameResult) -> dict[str, object]:
    """Convert one arena game to the stable compact JSON representation."""

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


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of a regular checkpoint file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_promote_checkpoint(
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
            raise ArenaCommandError(
                "candidate checkpoint changed after arena evaluation; promotion aborted"
            )
        os.replace(temporary_path, destination)  # noqa: PTH105
        temporary_path = None
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink()


def _run_search(
    settings: AppConfig,
    moves: list[int] | tuple[int, ...],
    *,
    root_noise: bool,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    """Reconstruct one game and produce a JSON-compatible search report."""

    from azgo.game import GameState, Rules, Ruleset
    from azgo.search import MCTS

    rules = Rules(
        board_size=settings.game.board_size,
        komi=settings.game.komi,
        ruleset=Ruleset(settings.game.rules.ruleset),
    )
    state = GameState.new(rules, zobrist_seed=settings.zobrist.seed)
    applied_moves: list[int] = []
    for action in moves:
        state = state.apply(action)
        applied_moves.append(action)

    evaluator, evaluator_name, checkpoint_step = _build_evaluator(settings, checkpoint)
    search = MCTS(
        evaluator,
        simulations=settings.search.simulations,
        c_puct=settings.search.c_puct,
        seed=settings.search.seed,
        dirichlet_alpha=settings.search.dirichlet_alpha,
        dirichlet_fraction=settings.search.dirichlet_fraction,
    )
    result = search.run(state, add_root_noise=root_noise)
    coordinate = state.action_to_coord(result.selected_action)
    return {
        "applied_moves": applied_moves,
        "board_size": state.board_size,
        "checkpoint_step": checkpoint_step,
        "evaluator": evaluator_name,
        "root_noise": root_noise,
        "root_value": float(result.root_value),
        "selected_action": int(result.selected_action),
        "selected_coordinate": None if coordinate is None else list(coordinate),
        "selected_is_pass": result.selected_action == state.pass_action,
        "simulations": int(result.simulations),
        "visit_counts": [int(count) for count in result.visit_counts],
        "visit_policy": [float(probability) for probability in result.visit_policy],
    }


def _self_play_failures() -> tuple[type[Exception], ...]:
    """Return command-local self-play failures without loading them for other commands."""

    from azgo.checkpoint import CheckpointError
    from azgo.game import GoEngineError
    from azgo.replay import ReplayError
    from azgo.self_play import SelfPlayError

    return (
        GoEngineError,
        ReplayError,
        SelfPlayError,
        CheckpointError,
        OSError,
        ValueError,
    )


def _run_self_play(
    settings: AppConfig,
    output: Path,
    *,
    overwrite: bool,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    """Generate a complete game batch, then atomically update replay storage."""

    from azgo.game import Color
    from azgo.replay import ReplayBuffer, ReplayError
    from azgo.self_play import SelfPlayRunner

    output = output.expanduser().resolve()
    if output.exists() and not overwrite:
        buffer = ReplayBuffer.load(output)
        expected = (
            settings.game.board_size,
            settings.model.history_length,
            settings.replay.capacity,
        )
        actual = (buffer.board_size, buffer.history_length, buffer.capacity)
        if actual != expected:
            raise ReplayError(
                "existing replay metadata does not match configuration: "
                f"expected board_size/history_length/capacity {expected}, got {actual}"
            )
    else:
        buffer = ReplayBuffer(
            board_size=settings.game.board_size,
            history_length=settings.model.history_length,
            capacity=settings.replay.capacity,
        )

    evaluator, evaluator_name, checkpoint_step = _build_evaluator(settings, checkpoint)
    runner = SelfPlayRunner(evaluator, settings)
    first_game_index = buffer.next_game_index
    games = [
        runner.play_game(first_game_index + offset)
        for offset in range(settings.self_play.games)
    ]

    positions_generated = sum(len(game.samples) for game in games)
    black_wins = sum(game.winner is Color.BLACK for game in games)
    white_wins = sum(game.winner is Color.WHITE for game in games)
    draws = sum(game.winner is None for game in games)

    for game in games:
        buffer.add_game(game)
    buffer.save(output)

    return {
        "black_wins": black_wins,
        "board_size": settings.game.board_size,
        "checkpoint_step": checkpoint_step,
        "draws": draws,
        "evaluator": evaluator_name,
        "games_generated": len(games),
        "next_game_index": buffer.next_game_index,
        "output": str(output),
        "positions_generated": positions_generated,
        "replay_capacity": buffer.capacity,
        "replay_size": len(buffer),
        "white_wins": white_wins,
    }


def _training_failures() -> tuple[type[Exception], ...]:
    """Return command-local learner failures without loading training for other commands."""

    from azgo.checkpoint import CheckpointError
    from azgo.learner import TrainingError
    from azgo.replay import ReplayError

    return (TrainingError, CheckpointError, ReplayError, OSError, ValueError)


def _run_training(
    settings: AppConfig,
    replay_path: Path,
    checkpoint_path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, object]:
    """Run the configured number of deterministic CPU learner updates."""

    import numpy as np
    import torch

    from azgo.checkpoint import load_checkpoint, save_checkpoint
    from azgo.learner import Learner, TrainingError, TrainingMetrics
    from azgo.replay import ReplayBuffer

    replay_path = replay_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if resume and overwrite:
        raise TrainingError("--resume and --overwrite cannot be used together")
    if resume and not checkpoint_path.is_file():
        raise TrainingError("--resume requires an existing checkpoint file")
    if checkpoint_path.exists() and not resume and not overwrite:
        raise TrainingError(
            "checkpoint already exists; pass --resume to continue it or --overwrite to replace it"
        )

    replay = ReplayBuffer.load(replay_path)
    expected_replay = (settings.game.board_size, settings.model.history_length)
    actual_replay = (replay.board_size, replay.history_length)
    if actual_replay != expected_replay:
        raise TrainingError(
            "replay metadata does not match configuration: "
            f"expected board_size/history_length {expected_replay}, got {actual_replay}"
        )
    if len(replay) < settings.learner.batch_size:
        raise TrainingError(
            f"replay size {len(replay)} is smaller than batch_size "
            f"{settings.learner.batch_size}"
        )

    torch.manual_seed(settings.learner.seed)
    network = _build_network(settings)
    learner = Learner(network, settings)
    if resume:
        metadata = load_checkpoint(
            checkpoint_path,
            network=network,
            config=settings,
            optimizer=learner.optimizer,
            restore_rng=True,
        )
        learner.restore_step(metadata.step)

    start_step = learner.step
    metrics: list[TrainingMetrics] = []
    for _ in range(settings.learner.steps):
        sample_seed = int(
            np.random.SeedSequence([settings.learner.seed, learner.step]).generate_state(
                1,
                dtype=np.uint64,
            )[0]
        )
        batch = replay.sample(
            settings.learner.batch_size,
            sample_seed,
            augment=settings.learner.augment,
        )
        metric = learner.train_step(batch)
        metrics.append(metric)
        if metric.step % settings.learner.checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                network=network,
                optimizer=learner.optimizer,
                step=learner.step,
                config=settings,
            )

    save_checkpoint(
        checkpoint_path,
        network=network,
        optimizer=learner.optimizer,
        step=learner.step,
        config=settings,
    )
    final = metrics[-1]
    count = len(metrics)
    return {
        "board_size": settings.game.board_size,
        "checkpoint": str(checkpoint_path),
        "end_step": learner.step,
        "final_gradient_norm": final.gradient_norm,
        "final_policy_loss": final.policy_loss,
        "final_total_loss": final.total_loss,
        "final_value_loss": final.value_loss,
        "mean_policy_loss": sum(item.policy_loss for item in metrics) / count,
        "mean_total_loss": sum(item.total_loss for item in metrics) / count,
        "mean_value_loss": sum(item.value_loss for item in metrics) / count,
        "replay_size": len(replay),
        "resumed": resume,
        "start_step": start_step,
        "steps_completed": learner.step - start_step,
    }
