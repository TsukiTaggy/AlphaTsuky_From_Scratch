# Architecture

`alphazero-go` is being built in correctness-gated milestones. Phase 1 provides
the package, validated configuration, command-line entry points, and quality
tooling. Phase 2 provides an immutable Go rules engine and its tests. This
document describes only those implemented boundaries.

## Current package boundaries

```text
YAML configuration
       |
       v
  azgo.config  <--------  azgo.cli
                            |
                            v
                        azgo.game
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

The Phase 1-2 schema covers only implemented behavior: board size, komi,
ruleset, Zobrist seed, benchmark seed, and benchmark workload. Neural-network,
search, self-play, replay, learner, and distributed settings are intentionally
absent until their owning subsystems exist.

### `azgo.cli`

The Typer command line is a thin adapter. It loads validated settings, creates
engine objects, runs the requested operation, and reports a useful error on
invalid input. Business rules remain in `azgo.game`, and YAML parsing and
validation remain in `azgo.config`.

The current commands validate configuration and benchmark legal engine play.
The benchmark uses an explicit random seed and configured workload so runs can
be reproduced. It exercises the engine; it is not a playing-strength or
training benchmark.

## State and action data flow

`GameState` is a frozen value object. A board is represented by immutable
bytes, and chronological board history is an immutable tuple containing the
initial board and the board after every legal action. Pass entries are retained
even though their board bytes duplicate the preceding entry. Applying an
action computes a child state and leaves all parent-owned data unchanged.

Stone actions use the row-major range `0..N*N-1`, and pass is `N*N`. This same
layout is used by coordinate helpers and legal-action masks, preventing an
adapter-specific action convention from leaking into future policy code.

For a placement, the engine computes captures and suicide legality, derives the
resulting board hash, and uses that hash to narrow positional-superko
candidates. It then compares exact board bytes before declaring a repetition.
This two-stage check preserves correctness even under a Zobrist collision.

Scoring depends on the board arrangement rather than capture counts. It returns
named Black and White area totals after adding komi to White. Terminal
`outcome(perspective)` converts those totals to `+1`, `0`, or `-1` for an
explicitly requested color. See [Go rules](game_rules.md) for the normative
behavior.

## Dependency and reproducibility rules

- `azgo.game` must never import PyTorch or depend on a neural-network type.
- Mutable process-wide singletons are not used for rules, configuration, or
  random-number generation.
- Every stochastic operation accepts an explicit seed. Zobrist tables and
  benchmark action selection are therefore reproducible.
- YAML values are resolved and validated before they affect runtime behavior.
- Public functions and methods are typed, and tests enforce immutable parent
  states and collision-safe rule behavior.
- Correctness checks precede optimization. Hash-based candidate filtering is
  permitted because exact comparisons remain authoritative.

## Deferred AlphaZero subsystems

State encoding, board symmetries, neural networks, MCTS, inference batching,
self-play, replay storage, learning, checkpoints, arenas, SGF export,
observability, and distributed workers are outside the Phase 1-2 milestone.
There are no placeholder implementations or configuration sections for them.

Their eventual dependency direction is constrained by the current boundary:
they may consume the public Go engine, but the Go engine must not depend on
them. Documentation for those subsystems will be added when their behavior is
implemented and testable.
