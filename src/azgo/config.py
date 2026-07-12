"""Composition and validation for the implemented project phases."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter, field_validator

BoardSize = Annotated[int, Field(strict=True)]
Seed = Annotated[int, Field(strict=True, ge=0, le=(2**64) - 1)]
PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
MoveLimitStrictInt = Annotated[int, Field(strict=True, ge=2)]
PositiveFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, gt=0.0, allow_inf_nan=False),
]
UnitFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]
NonNegativeFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, allow_inf_nan=False),
]
MomentumFiniteStrictFloat = Annotated[
    float,
    Field(strict=True, ge=0.0, lt=1.0, allow_inf_nan=False),
]
SUPPORTED_BOARD_SIZES = frozenset({5, 9, 13, 19})


class RulesConfig(BaseModel):
    """The fixed baseline Go rules semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ruleset: Literal["tromp_taylor"]
    scoring: Literal["area"]
    suicide: Literal["illegal"]
    superko: Literal["positional"]
    pass_repetition_exempt: Literal[True]


class GameConfig(BaseModel):
    """Validated settings needed to construct a Go game."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    board_size: BoardSize
    komi: float = Field(allow_inf_nan=False)
    rules: RulesConfig

    @field_validator("board_size")
    @classmethod
    def validate_board_size(cls, value: int) -> int:
        """Restrict games to the board sizes covered by the engine contract."""

        if value not in SUPPORTED_BOARD_SIZES:
            supported = ", ".join(str(size) for size in sorted(SUPPORTED_BOARD_SIZES))
            raise ValueError(f"board_size must be one of {{{supported}}}")
        return value


class ZobristConfig(BaseModel):
    """Settings for reproducible position hashing."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: Seed


class BenchmarkConfig(BaseModel):
    """A deterministic random-game engine benchmark workload."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: Seed
    games: int = Field(ge=1)
    max_moves_per_game: int = Field(ge=2)


class ModelConfig(BaseModel):
    """Validated dimensions for state encoding and the policy-value network."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    history_length: PositiveStrictInt
    channels: PositiveStrictInt
    residual_blocks: PositiveStrictInt
    value_hidden_size: PositiveStrictInt


class SearchConfig(BaseModel):
    """Validated settings for deterministic synchronous PUCT search."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    simulations: PositiveStrictInt
    c_puct: PositiveFiniteStrictFloat
    seed: Seed
    dirichlet_alpha: PositiveFiniteStrictFloat
    dirichlet_fraction: UnitFiniteStrictFloat

    @field_validator("c_puct", "dirichlet_alpha", "dirichlet_fraction", mode="before")
    @classmethod
    def require_float_scalars(cls, value: object) -> object:
        """Reject integers and booleans at the strict YAML configuration boundary."""

        if type(value) is not float:
            raise ValueError("value must be a floating-point number")
        return value


class SelfPlayConfig(BaseModel):
    """Validated settings for deterministic self-play generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: Seed
    games: PositiveStrictInt
    max_moves: MoveLimitStrictInt
    temperature: PositiveFiniteStrictFloat
    temperature_moves: NonNegativeStrictInt
    root_noise: StrictBool

    @field_validator("temperature", mode="before")
    @classmethod
    def require_float_temperature(cls, value: object) -> object:
        """Reject integers and booleans at the strict YAML boundary."""

        if type(value) is not float:
            raise ValueError("temperature must be a floating-point number")
        return value


class ReplayConfig(BaseModel):
    """Validated settings for bounded position replay storage."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capacity: PositiveStrictInt


class LearnerConfig(BaseModel):
    """Validated settings for deterministic CPU network optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    seed: Seed
    batch_size: PositiveStrictInt
    steps: PositiveStrictInt
    learning_rate: PositiveFiniteStrictFloat
    momentum: MomentumFiniteStrictFloat
    weight_decay: NonNegativeFiniteStrictFloat
    value_loss_weight: PositiveFiniteStrictFloat
    gradient_clip_norm: PositiveFiniteStrictFloat
    checkpoint_interval: PositiveStrictInt
    augment: StrictBool

    @field_validator(
        "learning_rate",
        "momentum",
        "weight_decay",
        "value_loss_weight",
        "gradient_clip_norm",
        mode="before",
    )
    @classmethod
    def require_float_scalars(cls, value: object) -> object:
        """Reject integers and booleans at the strict YAML configuration boundary."""

        if type(value) is not float:
            raise ValueError("value must be a floating-point number")
        return value


class AppConfig(BaseModel):
    """Complete validated configuration for Phases 1 through 6."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    game: GameConfig
    zobrist: ZobristConfig
    benchmark: BenchmarkConfig
    model: ModelConfig
    search: SearchConfig
    self_play: SelfPlayConfig
    replay: ReplayConfig
    learner: LearnerConfig


_APP_CONFIG_ADAPTER = TypeAdapter(AppConfig)


def compose_config(path: str | Path, overrides: tuple[str, ...] = ()) -> DictConfig:
    """Compose one YAML file with Hydra and resolve interpolation.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ValueError: If ``path`` is not YAML or its root is not a mapping.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        msg = f"configuration must be a YAML file: {config_path}"
        raise ValueError(msg)

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_path.parent),
        job_name="azgo",
    ):
        composed: object = compose(config_name=config_path.stem, overrides=list(overrides))

    if not isinstance(composed, DictConfig):
        msg = f"configuration root must be a mapping: {config_path}"
        raise ValueError(msg)
    OmegaConf.resolve(composed)
    return composed


def validate_config(config: DictConfig) -> AppConfig:
    """Validate a composed mapping and return an immutable model."""

    primitive = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return _APP_CONFIG_ADAPTER.validate_python(primitive)


def load_config(path: str | Path, overrides: tuple[str, ...] = ()) -> AppConfig:
    """Compose and validate a Phase 1-6 YAML configuration."""

    return validate_config(compose_config(path, overrides))
