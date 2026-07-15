# alphazero-go

`alphazero-go` is a correctness-first, research-oriented implementation of an
AlphaZero-style Go system. The current Phase 1-10 milestone contains the Python
project foundation, validated configuration, a PyTorch-independent Go rules
engine, deterministic state encoding and board symmetries, and a CPU-first
policy-value network with deterministic PUCT search, self-play generation,
bounded replay storage, CPU training, resumable checkpoints, paired arena
evaluation, deterministic concurrent inference for parallel self-play, and
crash-safe orchestration of complete AlphaZero training runs, plus deterministic
SGF FF[4] game records. It is usable on
5x5, 9x9, 13x13, and 19x19 boards.

The longer-term project is intended to learn only from self-play and the rules
of Go. Distributed execution is not part of this milestone and is not
represented by placeholder modules or configuration.

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
root value, visit counts and policy, simulation count, applied moves, evaluator
kind, and checkpoint step. It uses the deterministic `UniformEvaluator` by
default. Pass a compatible trusted checkpoint to use the trained network:

```console
uv run azgo search-move -c configs/engine/go5.yaml --checkpoint checkpoints/go5.pt
```

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
pass `--checkpoint checkpoints/go5.pt` to generate games with a compatible
trained model.

Checked-in profiles retain one game and one worker. After copying a profile to
`configs/engine/go5-parallel.yaml` and setting `self_play.games` above one, it
can opt into parallel generation through `self_play.workers` or override it for
one command:

```console
uv run azgo generate-self-play -c configs/engine/go5-parallel.yaml --output data/go5.npz --checkpoint checkpoints/go5.pt --workers 4
```

Parallel games share one serialized model worker. Requests form deterministic
barriers ordered by worker ID rather than timing windows, so replay ordering and
batch formation remain reproducible. The JSON report includes request, position,
model-batch, mean-batch-size, and maximum-batch-size metrics.

Add `--sgf-output records/self-play.sgf` to atomically write the complete
generated games as a deterministic UTF-8 SGF collection. SGF records are
inspection artifacts; replay snapshots remain the training source because they
also contain normalized MCTS visit policies.

Train the configured CPU network from a replay snapshot and create a checkpoint:

```console
uv run azgo train-network -c configs/engine/go5.yaml --replay data/go5.npz --checkpoint checkpoints/go5.pt
```

An existing destination is never replaced implicitly. Use `--overwrite` to
start fresh or `--resume` to restore its model, SGD optimizer, global step, and
random state, then run `learner.steps` additional updates. Search, self-play,
and training checkpoints are serialized PyTorch artifacts: load checkpoints
only from trusted sources.

Evaluate a candidate checkpoint against the current incumbent through paired,
color-swapped arena games:

```console
uv run azgo evaluate-arena -c configs/engine/go5.yaml --candidate checkpoints/candidate.pt --incumbent checkpoints/best.pt
```

Each pair starts from the same seeded, model-independent legal opening. Arena
play uses deterministic maximum-visit moves without root noise or temperature
sampling. A win contributes one candidate point and a draw contributes one
half; the candidate is promotion-eligible when its score meets the configured
inclusive threshold. The JSON report includes checkpoint hashes and steps,
aggregate results, and compact per-game evidence.

Add `--sgf-output records/arena.sgf` to preserve both complete color-swapped
games in each pair, including their shared openings and checkpoint identities.

Promotion is never implicit. Supplying an explicit destination atomically
copies the evaluated candidate only when it passes the gate:

```console
uv run azgo evaluate-arena -c configs/engine/go5.yaml --candidate checkpoints/candidate.pt --incumbent checkpoints/best.pt --promote-to checkpoints/best.pt
```

If evaluation fails or the candidate misses the threshold, the destination is
left unchanged.

Validate and inspect a supported linear SGF collection against the configured
engine rules:

```console
uv run azgo inspect-sgf -c configs/engine/go5.yaml --input records/self-play.sgf
```

The command replays every move through the Go engine and reports names, move
counts, scores, results, and winners as JSON. Setup positions, handicap,
variations, resignation, unfinished games, illegal moves, and incompatible
rules are rejected.

Run the complete configured self-play, learning, and arena cycle from a
deterministically seeded step-zero network:

```console
uv run azgo run-training-cycle -c configs/engine/go5.yaml --run-dir runs/go5
```

The managed run executes `training_run.cycles` cycles. It keeps immutable
bootstrap and candidate checkpoints, alternates between two replay snapshots,
and commits a versioned manifest after self-play, training, and arena stages.
Only an arena-eligible candidate becomes the next incumbent; a rejected
candidate is retained as evidence but never becomes the parent of later
training. Every new cycle also journals `self-play.sgf` and `arena.sgf` with
their hashes and game counts. Existing Phase 9 manifests migrate without
fabricating records for already committed stages.

After interruption, resume with the same configuration. Omitting `--workers`
reuses the worker count recorded during initialization:

```console
uv run azgo run-training-cycle -c configs/engine/go5.yaml --run-dir runs/go5 --resume
```

Resume validates configuration, paths, hashes, checkpoint compatibility,
replay metadata, and committed SGF records before doing work. A completed run resumes as a validated
no-op. Fresh mode refuses an existing directory, and concurrent writers are
rejected through an operating-system lock.

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
1-10 subsystems:

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
- `self_play.workers`: a positive worker count no greater than `self_play.games`
- `inference.max_batch_size`: a positive deterministic model-batch position cap
- `replay.capacity`: a positive FIFO capacity counted in positions
- `learner.seed`: an unsigned 64-bit seed for initialization and replay sampling
- `learner.batch_size`: a positive number of positions per optimizer update
- `learner.steps`: a positive number of additional updates per training command
- `learner.learning_rate`: a finite positive SGD learning rate
- `learner.momentum`: a finite SGD momentum in the half-open range `[0, 1)`
- `learner.weight_decay`: a finite nonnegative SGD weight decay
- `learner.value_loss_weight`: a finite positive value-loss multiplier
- `learner.gradient_clip_norm`: a finite positive global gradient-norm limit
- `learner.checkpoint_interval`: a positive periodic checkpoint interval
- `learner.augment`: a strict boolean enabling seeded D4 replay augmentation
- `arena.seed`: an unsigned 64-bit seed for deterministic paired openings
- `arena.games`: a positive even number of color-balanced evaluation games
- `arena.opening_moves`: a nonnegative number of seeded non-pass opening moves
- `arena.max_moves`: a safety bound greater than the opening length
- `arena.promotion_threshold`: a strict finite candidate score in `(0.5, 1]`
- `training_run.cycles`: the immutable positive total cycle target for a managed run

Replay capacity must be at least `learner.batch_size`. During a managed run,
every cycle generates at least one configured self-play batch; the first cycle
continues generating batches until a learner batch is available.

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
`tanh` current-player values shaped `[B]`. Loss construction, optimization,
batched inference services, and checkpoint persistence live outside the model.

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
Search itself remains synchronous and time-independent; parallel self-play
coordinates multiple independent searches through the inference boundary.

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

`ParallelSelfPlayRunner` assigns contiguous game indices to fixed worker IDs by
stride. One worker uses the original direct execution path; multiple workers
share deterministic inference clients and return games sorted by game index.
Any client, game, or evaluator failure aborts the complete batch before replay
storage changes.

## Concurrent inference

`azgo.inference.DeterministicInferenceCoordinator` owns a single model worker
and synchronous evaluator client per active self-play worker. It waits for one
request from every active client, orders requests by client ID, and splits the
flattened positions into fixed chunks no larger than `inference.max_batch_size`.
Completed clients leave later barriers. There is no latency window, so operating
system thread timing cannot change batch composition.

The coordinator validates every request and model result before returning
detached slices to clients. Evaluation or lifecycle failure aborts the service,
releases every waiter, and prevents partial game results. Immutable aggregate
metrics expose requests, positions, model batches, mean batch size, and maximum
observed batch size.

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

## Learning and checkpoints

`azgo.learner.Learner` performs deterministic CPU SGD updates from replay
batches. Policy loss is soft-target cross entropy against MCTS visit policies;
value loss is mean squared error against terminal current-player outcomes. The
configured value weight combines them, then the configured global norm clips
gradients before the optimizer step. Replay sampling is derived from the
learner seed and global step, so resumed runs continue the same sample stream.

`azgo.checkpoint` saves the model, optimizer, global step, full configuration,
compatibility metadata, and PyTorch RNG state through atomic replacement.
Inference loading restores only model weights; training resume also restores
optimizer and PyTorch RNG state before continuing at the stored step. Loading
uses PyTorch's restricted `weights_only=True` mode and validates exact fields,
configuration compatibility, tensor shapes, scalar metadata, and finite
numerical state before mutation. Still load checkpoint files only from sources
you trust.
Format-version-1 Phase 6 through Phase 8 checkpoints remain loadable when they
lack newer operational arena, inference, worker, or training-run fields; those
values are supplied only while validating metadata and never change model
compatibility.

## Arena evaluation and promotion

`azgo.arena.ArenaRunner` compares candidate and incumbent evaluators without
changing either checkpoint. It derives one legal non-pass opening per pair from
the arena seed, then plays that position twice with checkpoint colors swapped.
Both evaluator-specific MCTS trees advance through every action, preserving
safe subtree reuse, while move selection remains noise-free and deterministic.

Arena runs are all-or-nothing: an invalid opening, evaluator failure, or move
limit aborts the evaluation rather than producing partial promotion evidence.
Candidate score is `(wins + 0.5 * draws) / games`, and threshold equality
passes. The CLI identifies both evaluated files by SHA-256 and rechecks the
candidate while copying, so a checkpoint changed during evaluation cannot be
promoted. Explicit promotion uses same-directory temporary storage, syncing,
and atomic replacement.

## Managed training runs

`azgo.training_run.TrainingRunRunner` composes the typed operations in
`azgo.operations` without importing the CLI. Initialization builds a seeded
step-zero network and manifest in a sibling staging directory, then atomically
renames it into place. Artifact paths stored in the manifest are relative, so a
complete run directory can be relocated.

Each stage writes its output atomically before the manifest advances. Self-play
reads the manifest-selected replay and writes the inactive replay slot;
training writes a cycle-specific candidate from the current accepted incumbent;
arena commits either that candidate or the unchanged incumbent. If the process
stops after an output write but before its manifest commit, resume regenerates
only that uncommitted stage from immutable inputs. A committed artifact that is
missing, corrupt, incompatible, or outside the run directory fails closed.

The stable run layout is:

```text
run/
  manifest.json
  bootstrap.pt
  replays/replay-0.npz
  replays/replay-1.npz       # appears after the next cycle
  cycles/000001/candidate.pt
  cycles/000001/self-play.sgf
  cycles/000001/arena.sgf
  cycles/000002/candidate.pt
```

See [Go rules](docs/game_rules.md) for the normative rule contract and
[Architecture](docs/architecture.md) for package boundaries and data flow.

## Development and verification

Run every Phase 1-10 quality gate from the repository root:

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
NPZ round trips, atomic saving, and the self-play CLI workflow. Phase 6 coverage
adds learner loss and gradient validation, deterministic resume behavior,
checkpoint compatibility and corruption handling, periodic/final checkpoint
saving, and uniform versus checkpoint-backed CLI evaluation.
Phase 7 coverage adds deterministic paired openings, color balance, arena
scoring and threshold decisions, compact reports, checkpoint identity checks,
and atomic explicit promotion with failure-safe destination preservation.
Phase 8 coverage adds deterministic barrier ordering, fixed batch chunking,
client lifecycle and abort propagation, parallel-game equivalence, stable
replay ordering, inference metrics, and legacy checkpoint metadata migration.
Phase 9 coverage adds deterministic bootstrap, accepted and rejected candidate
lineage, alternating replay artifacts, strict manifests and path containment,
exclusive writer locking, stage-level failure recovery, corruption detection,
configuration and worker resume checks, and completed-run no-op validation.
Phase 10 coverage adds canonical SGF collections, engine-validated imports,
complete self-play and arena records, atomic SGF output, CLI inspection,
manifest artifact hashing, Phase 9 manifest migration, and SGF-aware recovery.

The engine benchmark is intended for reproducible regression measurements.
Profile evidence should precede internal optimization, and optimizations must
preserve the public immutable-state and exact-legality contracts.
