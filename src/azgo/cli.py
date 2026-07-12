"""Command-line tools for configuration and the Go rules engine."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Annotated

import typer
from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from azgo.config import AppConfig, load_config

app = typer.Typer(
    name="azgo",
    help="Correctness-first tools for the AlphaZero Go project.",
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
        help="Path to a Phase 1-5 YAML configuration.",
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
        help="Path to a Phase 1-5 YAML configuration.",
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
    """Compose and strictly validate a Phase 1-5 configuration."""

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
) -> None:
    """Analyze a reconstructed position with synchronous uniform-evaluator MCTS."""

    settings = _load_or_exit(config)
    try:
        report = _run_search(settings, () if moves is None else moves, root_noise=root_noise)
    except _search_failures() as exc:
        typer.echo(f"Search failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("generate-self-play")
def generate_self_play(
    config: ConfigOption,
    output: OutputOption,
    overwrite: OverwriteOption = False,
) -> None:
    """Generate deterministic uniform-evaluator games into a replay snapshot."""

    settings = _load_or_exit(config)
    try:
        report = _run_self_play(settings, output, overwrite=overwrite)
    except _self_play_failures() as exc:
        typer.echo(f"Self-play failed: {exc}", err=True)
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

    from azgo.game import GoEngineError
    from azgo.search import SearchError

    return (GoEngineError, SearchError, ValueError)


def _run_search(
    settings: AppConfig,
    moves: list[int] | tuple[int, ...],
    *,
    root_noise: bool,
) -> dict[str, bool | float | int | list[float] | list[int] | None]:
    """Reconstruct one game and produce a JSON-compatible search report."""

    from azgo.evaluator import UniformEvaluator
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

    search = MCTS(
        UniformEvaluator(),
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

    from azgo.game import GoEngineError
    from azgo.replay import ReplayError
    from azgo.self_play import SelfPlayError

    return (GoEngineError, ReplayError, SelfPlayError, OSError, ValueError)


def _run_self_play(
    settings: AppConfig,
    output: Path,
    *,
    overwrite: bool,
) -> dict[str, int | str]:
    """Generate a complete game batch, then atomically update replay storage."""

    from azgo.evaluator import UniformEvaluator
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

    runner = SelfPlayRunner(UniformEvaluator(), settings)
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
        "draws": draws,
        "games_generated": len(games),
        "next_game_index": buffer.next_game_index,
        "output": str(output),
        "positions_generated": positions_generated,
        "replay_capacity": buffer.capacity,
        "replay_size": len(buffer),
        "white_wins": white_wins,
    }
