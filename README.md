# alphazero-go

`alphazero-go` is a correctness-first, research-oriented implementation of an
AlphaZero-style Go system. The current Phase 1-2 milestone contains the Python
project foundation, validated engine configuration, and a PyTorch-independent
Go rules engine. It is usable on 5x5, 9x9, 13x13, and 19x19 boards.

The longer-term project is intended to learn only from self-play and the rules
of Go. Neural networks, MCTS, self-play actors, replay storage, training,
evaluation, and distributed execution are not part of this milestone and are
not represented by placeholder modules or configuration.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

## Quick start

From the repository root, create the locked development environment:

```console
uv sync --dev
```

Validate the default 5x5 engine configuration:

```console
uv run azgo validate-config configs/engine/go5.yaml
```

Run the deterministic random-play engine benchmark:

```console
uv run azgo benchmark-engine --config configs/engine/go5.yaml
```

The benchmark reports JSON containing the board size, requested and completed
games, seed, moves, elapsed time, and moves per second. Its configured maximum
move count is a benchmark safety bound, not a Go termination rule.

Equivalent validated configurations are provided at:

- `configs/engine/go5.yaml`
- `configs/engine/go9.yaml`
- `configs/engine/go13.yaml`
- `configs/engine/go19.yaml`

Use `uv run azgo --help` or `uv run azgo COMMAND --help` for command details.

## Configuration

Configuration is composed with Hydra/OmegaConf and validated with immutable
Pydantic models before an engine is constructed. Unknown fields and invalid
values are rejected.

The current YAML contract contains only Phase 1-2 settings:

- `game.board_size`: one of `5`, `9`, `13`, or `19`
- `game.komi`: a finite number added to White's score
- `game.rules`: the fixed Tromp-Taylor area-scoring, illegal-suicide,
  positional-superko rules declaration
- `zobrist.seed`: an unsigned 64-bit seed for deterministic position hashing
- `benchmark.seed`: an unsigned 64-bit seed for random benchmark play
- `benchmark.games`: a positive game count
- `benchmark.max_moves_per_game`: a safety bound of at least two moves

The fixed rule fields are deliberately explicit in YAML. Changing one to an
unsupported alternative fails validation instead of silently selecting
different semantics.

## Go engine

The public engine API is available from `azgo.game`:

```python
from azgo.game import GameState, Rules

state = GameState.new(Rules(board_size=5, komi=7.5), zobrist_seed=0)
action = 0  # row 0, column 0 on a 5x5 board
assert state.is_legal(action)
next_state = state.apply(action)
assert next_state is not state
```

Actions `0` through `N*N - 1` identify intersections in row-major order, and
action `N*N` is pass. There is no resignation action. Applying an action returns
a new immutable state; board and history storage owned by earlier states cannot
be changed by the child.

The baseline rules are:

- positional superko on stone placements, with pass exempt from repetition;
- illegal suicide, after resolving any captures;
- two consecutive passes end the game;
- Tromp-Taylor-style area scoring with configurable komi for White; and
- terminal outcome `+1`, `0`, or `-1` from an explicitly requested color's
  perspective.

Zobrist hashing is reproducible from its seed, but hashes never decide superko
alone. Every candidate repetition is confirmed by an exact immutable-board
comparison, so collisions cannot alter legality.

See [Go rules](docs/game_rules.md) for the normative rule contract and
[Architecture](docs/architecture.md) for package boundaries and data flow.

## Development and verification

Run every Phase 1-2 quality gate from the repository root:

```console
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The test suite covers groups and liberties, captures, suicide, simple ko and
longer superko, pass behavior, termination, scoring and komi, action encoding,
legal masks, immutable parent states, deterministic and collision-safe hashes,
random legal games, and property-based state invariants on every supported
board size.

The engine benchmark is intended for reproducible regression measurements.
Profile evidence should precede internal optimization, and optimizations must
preserve the public immutable-state and exact-legality contracts.
