# alphazero-go

`alphazero-go` is a correctness-first, research-oriented implementation of an
AlphaZero-style Go system. The current Phase 1-5 milestone contains the Python
project foundation, validated configuration, a PyTorch-independent Go rules
engine, deterministic state encoding and board symmetries, and a CPU-first
policy-value network with deterministic PUCT search, self-play generation, and
bounded replay storage. It is usable on 5x5, 9x9, 13x13, and 19x19 boards.

The longer-term project is intended to learn only from self-play and the rules
of Go. Inference queues, neural training, model evaluation, checkpoint
management, and distributed execution are not part of this milestone and are
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

Analyze the empty board with deterministic uniform-evaluator search:

```console
uv run azgo search-move --config configs/engine/go5.yaml
```

Use repeatable `--move` options to reconstruct a position and opt into seeded
root exploration noise when desired:

```console
uv run azgo search-move -c configs/engine/go5.yaml -m 0 -m 1 --root-noise
```

The command emits JSON containing the selected row-major action and coordinate,
root value, visit counts and policy, simulation count, and applied moves. It
uses the deterministic `UniformEvaluator`; model checkpoint loading is deferred.

Generate one configured batch of self-play games into a compressed replay
snapshot:

```console
uv run azgo generate-self-play --config configs/engine/go5.yaml --output data/go5.npz
```

Running the command again appends games with continuous deterministic game
indices. Pass `--overwrite` to start a new replay sequence at index zero. The
command generates the complete requested batch before changing the buffer and
saves through an atomic file replacement, so generation failures do not alter
an existing snapshot. It uses `UniformEvaluator` as a correctness smoke path;
Python callers may inject `TorchEvaluator` into `SelfPlayRunner`.

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

The current YAML contract contains only settings owned by implemented Phase
1-5 subsystems:

- `game.board_size`: one of `5`, `9`, `13`, or `19`
- `game.komi`: a finite number added to White's score
- `game.rules`: the fixed Tromp-Taylor area-scoring, illegal-suicide,
  positional-superko rules declaration
- `zobrist.seed`: an unsigned 64-bit seed for deterministic position hashing
- `benchmark.seed`: an unsigned 64-bit seed for random benchmark play
- `benchmark.games`: a positive game count
- `benchmark.max_moves_per_game`: a safety bound of at least two moves
- `model.history_length`: a positive number of encoded board positions
- `model.channels`: a positive residual-trunk channel count
- `model.residual_blocks`: a positive residual-block count
- `model.value_hidden_size`: a positive hidden size for the value head
- `search.simulations`: a positive number of root simulations
- `search.c_puct`: a finite positive PUCT exploration constant
- `search.seed`: an unsigned 64-bit seed for optional root noise
- `search.dirichlet_alpha`: a finite positive Dirichlet concentration
- `search.dirichlet_fraction`: root-prior noise weight in the inclusive range
  `[0, 1]`
- `self_play.seed`: an unsigned 64-bit seed for deterministic game streams
- `self_play.games`: a positive number of games generated per command batch
- `self_play.max_moves`: a safety bound of at least two moves; reaching it is
  an error rather than an adjudicated result
- `self_play.temperature`: a finite positive visit-sampling temperature
- `self_play.temperature_moves`: a nonnegative number of early sampled moves
- `self_play.root_noise`: a strict boolean controlling root noise at each move
- `replay.capacity`: a positive FIFO capacity counted in positions

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

## Encoding, symmetry, and network

`azgo.encoding.encode_state` converts a `GameState` into a contiguous `float32`
feature tensor shaped `[2H+1, N, N]`, where `H` is `history_length`.
Newest-to-oldest history entries contribute a current-player stone plane and
an opponent stone plane. Unavailable older history is zero-filled,
pass-created duplicate boards remain represented, and the final plane is all
ones for Black to play or all zeros for White. Tensor rows and columns use the
same orientation as the engine's row-major action layout.

`azgo.symmetry.Symmetry` exposes the eight D4 symmetries of a square board and
can transform encoded features, board actions, and policy vectors. The pass
action is invariant. Each transform has an exact inverse, so augmentation
preserves feature/action alignment and policy probability mass.

The CPU-first `azgo.network.PolicyValueNetwork` uses dimensions declared in
the `model` section. Its default architecture uses a 64-channel convolutional
stem, four residual blocks, and separate policy and value heads. Given a batch
shaped `[B, 2H+1, N, N]`, it returns raw policy logits shaped `[B, N*N+1]` and
`tanh` current-player values shaped `[B]`. Softmax, loss construction,
optimization, batched inference services, and checkpoint management are
intentionally left to later phases.

## Evaluation and search

`azgo.evaluator` defines a batch evaluator boundary returning policy logits and
current-player values. `UniformEvaluator` supplies zero logits and values for
deterministic engine-only searches, while `TorchEvaluator` adapts the Phase 3
network under inference mode.

`azgo.search.MCTS` performs synchronous, single-threaded PUCT search. It masks
illegal actions, expands the root before counting simulations, uses exact game
outcomes at terminal leaves, and backs values up with alternating perspective.
Selection ties are resolved by the smallest action. Optional seeded Dirichlet
noise is mixed into legal root priors only. Search results expose visit counts,
their normalized policy, the selected action, and the root value.

Search nodes retain complete immutable `GameState` histories because identical
stone arrangements can have different positional-superko histories. The tree
therefore does not merge transpositions. Call `advance(action)` to retain an
explored child subtree after a move, or `reset(state)` to start from a new root.
Concurrent inference queues, time-limited search, and checkpoint loading remain
later-phase work.

## Self-play and replay

`azgo.self_play.SelfPlayRunner` plays complete games with Phase 4 MCTS. When
configured, root noise is applied at every move. During the configured
temperature window, moves are sampled from powered root visit counts; later
moves use the maximum visit count with deterministic smallest-action
tie-breaking. Every
stored `TrainingSample` contains canonical encoded features, the normalized
root visit policy, the selected action and metadata, and a terminal value from
the player-to-move perspective of that feature tensor. Reaching `max_moves`
raises `SelfPlayLimitError`; partial games are never labeled with invented
outcomes.

`azgo.replay.ReplayBuffer` is a fixed-board, position-capacity FIFO buffer.
Canonical samples are stored once, while seeded sampling can apply a random D4
symmetry to each selected feature/policy pair without modifying storage.
Sampling is without replacement and returned arrays are contiguous and
read-only.

Portable snapshots use compressed NPZ with pickle disabled. Their fields are
`version`, `board_size`, `history_length`, `capacity`, `next_game_index`,
`features`, `policies`, `values`, `to_play`, `move_numbers`,
`selected_actions`, and `game_indices`. Loading validates metadata, shapes,
dtypes, finiteness, probability normalization, and value bounds. Saving writes
a temporary file in the target directory and atomically replaces the
destination.

See [Go rules](docs/game_rules.md) for the normative rule contract and
[Architecture](docs/architecture.md) for package boundaries and data flow.

## Development and verification

Run every Phase 1-5 quality gate from the repository root:

```console
uv run ruff check .
uv run mypy src tests
uv run pytest
```

The test suite covers groups and liberties, captures, suicide, simple ko and
longer superko, pass behavior, termination, scoring and komi, action encoding,
legal masks, immutable parent states, deterministic and collision-safe hashes,
random legal games, property-based state invariants, feature history and
perspective, symmetry round trips, and policy-value forward/backward behavior
on every supported board size. Phase 4 coverage adds evaluator validation,
legal-only priors, deterministic PUCT selection and backup, seeded root noise,
tree reuse, and search CLI behavior. Phase 5 coverage adds deterministic
self-play, temperature selection, root-noise streams, terminal value targets,
move-limit failure, FIFO eviction, replay sampling and augmentation, validated
NPZ round trips, atomic saving, and the self-play CLI workflow.

The engine benchmark is intended for reproducible regression measurements.
Profile evidence should precede internal optimization, and optimizations must
preserve the public immutable-state and exact-legality contracts.
