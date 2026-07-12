# Architecture

`alphazero-go` is being built in correctness-gated milestones. Phase 1 provides
the package, validated configuration, command-line entry points, and quality
tooling. Phase 2 provides an immutable Go rules engine and its tests. Phase 3
adds deterministic feature encoding, square-board symmetries, and a CPU-first
policy-value network. Phase 4 adds a validated evaluator boundary and
deterministic, synchronous PUCT search. Phase 5 adds complete-game self-play
generation and bounded, persistent replay storage. This document describes only
those implemented boundaries.

## Current package boundaries

```text
YAML ----> azgo.config <---- azgo.cli ----> azgo.replay
                 |              |               ^
                 |              v               |
                 |        azgo.self_play --------+
                 |          |       |
                 |          v       v
                 |     azgo.search  azgo.encoding ----> azgo.game
                 |          |                              ^
                 v          v                              |
            azgo.network <- azgo.evaluator ----------------+
                 ^
                 |
            azgo.symmetry <-------------------- azgo.replay
```

### `azgo.game`

`azgo.game` owns all Go rule behavior and is independent from PyTorch. Its core
public surface includes:

- `Color`, plus the semantic `Stone` and `Intersection` aliases
- `Rules` (`GameRules` is an alias) and the supported `Ruleset`
- `ZobristTable`
- `GameState`
- `Group` and `Score`
- typed engine errors for invalid actions, occupation, suicide, superko,
  unfinished games, and already-finished games
- `coord_to_action`, `action_to_coord`, and `pass_action`

A new game is constructed with validated rules and an explicit hashing seed:

```python
from azgo.game import GameState, Rules

state = GameState.new(Rules(board_size=5, komi=7.5), zobrist_seed=0)
```

`GameState` exposes legality checks, legal actions and masks, immutable action
application, group inspection, scoring, and terminal outcomes. Rules and
hashing dependencies are passed into construction rather than read from
process-global state.

### `azgo.config`

The configuration boundary composes YAML with Hydra/OmegaConf and then
validates the fully resolved mapping with Pydantic. Invalid board sizes,
non-finite komi, unsupported rulesets, malformed seeds, and invalid benchmark
workloads fail before engine construction. Code outside this boundary consumes
validated typed settings rather than unstructured dictionaries.

The Phase 1-5 schema covers only implemented behavior: board size, komi,
ruleset, Zobrist seed, benchmark workload, encoding history length, residual
trunk width and depth, value-head hidden size, and PUCT search settings. Search
configuration includes a simulation count, exploration constant, unsigned
64-bit random seed, and Dirichlet concentration and mixing fraction. Self-play
configuration adds a seed, game batch size, safety move limit, temperature
window, and root-noise switch. Replay configuration declares its position
capacity. Learner, checkpoint, arena, and distributed settings are intentionally
absent until their owning subsystems exist.

### `azgo.encoding` and `azgo.symmetry`

The Phase 3 encoding boundary consumes immutable `GameState` values without
changing the engine. For history length `H`, it emits a contiguous `float32`
tensor shaped `[2H+1, N, N]`. Newest-to-oldest positions are encoded as paired
current-player and opponent planes, missing positions are zero-filled, and the
last plane is one for Black to play and zero for White. History is interpreted
from the current state's perspective, so a player change also changes which
stones occupy the paired planes.

All eight elements of the square board's D4 symmetry group share one action
coordinate convention with feature and policy transforms. Stone actions move
with their intersections, pass remains `N*N`, and inverse transforms restore
features, actions, and policies exactly. These utilities provide deterministic
augmentation primitives; no random augmentation policy belongs to this phase.

### `azgo.network`

`PolicyValueNetwork` owns the PyTorch model and consumes batches shaped
`[B, 2H+1, N, N]`. A convolutional stem feeds a configurable residual trunk.
The policy head emits raw logits shaped `[B, N*N+1]`, including pass, while the
value head emits `tanh`-bounded scalars shaped `[B]` from the current player's
perspective. The CPU-first defaults are `H=8`, 64 channels, four residual
blocks, and a 64-unit value hidden layer.

The network validates its construction dimensions and input rank, channel
count, and board dimensions. It does not own softmax, loss calculation,
optimization, inference queues, checkpoints, or device orchestration.

### `azgo.evaluator`

The evaluator boundary decouples search from model execution. Evaluators accept
a non-empty homogeneous batch of `GameState` values and return policy logits
shaped `[B, N*N+1]` plus finite current-player values shaped `[B]`.
`UniformEvaluator` returns zero logits and values for deterministic tests and
the engine-only CLI. `TorchEvaluator` encodes state histories and runs a
compatible `PolicyValueNetwork` under PyTorch inference mode.

### `azgo.search`

`MCTS` owns synchronous, single-threaded PUCT traversal. It expands a root
before simulations, masks illegal actions before normalizing priors, selects
equal-scoring actions by smallest action index, uses exact terminal outcomes,
and alternates value perspective during backup. A `SearchResult` reports the
selected action, per-action visit counts and normalized visit policy, root
value, and completed simulation count.

Optional seeded Dirichlet noise is mixed once into legal root priors only.
`advance(action)` preserves an explored child subtree (or constructs a fresh
legal child), while `reset(state)` replaces the root. Nodes retain full
`GameState` history and are not merged through a transposition table: positions
with the same stones can have different superko legality due to prior boards.
The search layer does not own self-play temperature sampling, resignation,
virtual loss, inference queues, checkpoint loading, or time-based limits.

### `azgo.self_play`

`SelfPlayRunner` turns synchronous MCTS results into complete AlphaZero training
games. It derives independent search-noise and action-sampling random streams
from the configured self-play seed, search seed, and deterministic game index.
When enabled, root noise is used at every move. Early moves sample powered visit
counts at the configured positive temperature; later moves select the largest
visit count with the search layer's smallest-action tie rule. The explored child
subtree is retained after every action.

Each immutable `TrainingSample` owns contiguous, read-only canonical features
and visit policy plus its terminal current-player value, color, move number,
selected action, and game index. Values are assigned only after normal two-pass
termination. Hitting the configured safety move limit raises
`SelfPlayLimitError`, and the incomplete game produces no labeled result.

### `azgo.replay`

`ReplayBuffer` accepts validated complete games for one fixed board size and
history length. Its positive capacity counts positions; insertion evicts the
oldest positions first. Sampling is seeded and without replacement. Optional
augmentation chooses one D4 symmetry per sampled item and applies it to both
features and policy, leaving canonical stored arrays unchanged. Returned batch
arrays are contiguous and read-only.

Replay snapshots use compressed NPZ without object arrays or pickle. The fields
are `version`, `board_size`, `history_length`, `capacity`, `next_game_index`,
`features`, `policies`, `values`, `to_play`, `move_numbers`,
`selected_actions`, and `game_indices`. Loading validates version and metadata
as well as all shapes, dtypes, finite values, normalized policies, and bounded
targets. Saving writes a temporary snapshot beside the destination and
atomically replaces the target, so readers never observe a partially written
archive.

### `azgo.cli`

The Typer command line is a thin adapter. It loads validated settings, creates
engine objects, runs the requested operation, and reports a useful error on
invalid input. Business rules remain in `azgo.game`, and YAML parsing and
validation remain in `azgo.config`.

The current commands validate configuration, benchmark legal engine play,
analyze a move with uniform-evaluator MCTS, and generate self-play replay data.
The benchmark uses an explicit random seed and configured workload so runs can
be reproduced. `search-move` can reconstruct a state from repeatable row-major
actions and optionally enable seeded root noise; it emits a machine-readable
JSON search report. `generate-self-play` appends a complete configured batch to
a compatible snapshot or starts over with `--overwrite`. It generates all games
before mutating replay state and reports game outcomes and replay counts as
JSON. Checkpoint loading remains deferred to a later milestone.

## State and action data flow

`GameState` is a frozen value object. A board is represented by immutable
bytes, and chronological board history is an immutable tuple containing the
initial board and the board after every legal action. Pass entries are retained
even though their board bytes duplicate the preceding entry. Applying an
action computes a child state and leaves all parent-owned data unchanged.

Stone actions use the row-major range `0..N*N-1`, and pass is `N*N`. This same
layout is used by coordinate helpers and legal-action masks, preventing an
adapter-specific action convention from leaking into encoding, symmetry,
policy, evaluator, or search code.

For a placement, the engine computes captures and suicide legality, derives the
resulting board hash, and uses that hash to narrow positional-superko
candidates. It then compares exact board bytes before declaring a repetition.
This two-stage check preserves correctness even under a Zobrist collision.

Scoring depends on the board arrangement rather than capture counts. It returns
named Black and White area totals after adding komi to White. Terminal
`outcome(perspective)` converts those totals to `+1`, `0`, or `-1` for an
explicitly requested color. See [Go rules](game_rules.md) for the normative
behavior.

Search nodes store the complete immutable state, including chronological board
history. This makes tree reuse safe without assuming that a board arrangement
alone determines legal moves under positional superko.

Self-play captures each canonical feature tensor and normalized root visit
policy before applying its selected action. After two passes terminate the
game, the final score labels every captured position from that position's
player-to-move perspective. Replay preserves these canonical records in FIFO
order and transforms aligned feature/policy pairs only when augmentation is
requested during sampling.

## Dependency and reproducibility rules

- `azgo.game` must never import PyTorch or depend on a neural-network type.
- Encoding and symmetry code may consume the public game API but must not add
  PyTorch dependencies to `azgo.game`.
- `azgo.network` consumes tensors and validated dimensions; it does not own Go
  rule legality or mutate game states.
- `azgo.search` depends only on the evaluator contract and public immutable
  game API; neither the engine nor network imports search.
- `azgo.self_play` orchestrates search, encoding, and immutable game state; it
  does not alter rule semantics or construct learner tensors.
- `azgo.replay` stores validated self-play records and may use symmetry during
  sampling; neither the game engine nor search depends on replay.
- Mutable process-wide singletons are not used for rules, configuration, or
  random-number generation.
- Every stochastic operation accepts an explicit seed. Zobrist tables,
  benchmark action selection, optional root noise, self-play move selection,
  and replay sampling are therefore reproducible.
- YAML values are resolved and validated before they affect runtime behavior.
- Public functions and methods are typed, and tests enforce immutable parent
  states and collision-safe rule behavior.
- Correctness checks precede optimization. Hash-based candidate filtering is
  permitted because exact comparisons remain authoritative.

## Deferred AlphaZero subsystems

Concurrent inference queues, learning, checkpoints, arenas, SGF export,
observability, and distributed workers are outside the Phase 1-5 milestone.
There are no placeholder implementations or configuration sections for them.

Their eventual dependency direction is constrained by the current boundary:
they may consume the public Go engine, but the Go engine must not depend on
them. Documentation for those subsystems will be added when their behavior is
implemented and testable.
