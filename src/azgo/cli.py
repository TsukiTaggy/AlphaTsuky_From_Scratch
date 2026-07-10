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
        help="Path to a Phase 1-2 YAML configuration.",
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
        help="Path to a Phase 1-2 YAML configuration.",
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
    """Compose and strictly validate a Phase 1-2 configuration."""

    settings = _load_or_exit(config)
    typer.echo(json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True))
    typer.echo(f"Configuration is valid: {config}")


@app.command("benchmark-engine")
def benchmark_engine(config: ConfigOption) -> None:
    """Benchmark deterministic random legal games using the configured engine."""

    settings = _load_or_exit(config)
    typer.echo(json.dumps(_run_engine_benchmark(settings), indent=2, sort_keys=True))


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
