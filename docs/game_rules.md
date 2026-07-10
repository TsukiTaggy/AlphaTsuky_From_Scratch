# Go rules

This document is the normative rules contract for the Phase 1-2 engine. The
engine implements a single baseline ruleset: Tromp-Taylor-style area scoring,
illegal suicide, and positional superko. It supports square boards of size 5,
9, 13, and 19.

## Board, colors, and actions

The board contains `N * N` intersections. Black moves first, and the player to
move alternates after every legal action, including a pass. Intersections use
zero-based `(row, column)` coordinates, with `(0, 0)` at the upper-left of the
engine's array representation.

Actions are row-major integers:

```text
stone action = row * N + column       # 0 through N*N - 1
pass action  = N * N                  # the final policy entry
```

There is no resignation action. A legal-action mask therefore has length
`N * N + 1`, and its final element represents pass.

## Groups, liberties, and capture

Stones are connected only along the four orthogonal directions. A group is a
maximal orthogonally connected set of stones of one color. Its liberties are
the distinct empty intersections orthogonally adjacent to any stone in the
group.

A stone placement is evaluated in this order:

1. The action must identify an intersection on the board, and that intersection
   must be empty.
2. The new stone is placed on a temporary board.
3. Every adjacent opponent group with no liberties is removed.
4. The new stone's group must have at least one liberty after those captures.
5. The resulting stone arrangement must satisfy positional superko.

Step 4 makes suicide illegal. A move that initially fills its last apparent
liberty is nevertheless legal when it captures opponent stones and thereby
gains a liberty.

## Positional superko

For every stone placement, the resulting arrangement of black, white, and
empty intersections must not equal any earlier board arrangement in the same
game. The comparison ignores whose turn it is; this is positional, not
situational, superko. Simple ko is covered by the same rule.

Pass is explicitly exempt from repetition checks. Passing is legal in every
non-terminal state even though it leaves the board unchanged. Board history
still records the position after a pass so chronological history remains
available to later state encoders.

The engine uses deterministic Zobrist hashes generated from the configured
seed to find possible repetitions efficiently. A matching hash is only a
candidate match: the engine also compares the exact immutable board bytes.
Consequently, a Zobrist collision cannot incorrectly make a move legal or
illegal. The hash describes the stone arrangement and is reproducible for a
given board size and Zobrist seed.

## Passing and game termination

A stone placement resets the consecutive-pass count. A pass increments it.
Two consecutive passes end the game. Once terminal, the state accepts no
further stone placement or pass, and its legal-action mask contains no legal
actions.

Maximum game length is deliberately not a rule-engine concept. A future
self-play system may enforce a separately configured safety limit, but such a
limit does not change legality or the score in this engine.

## Area scoring and komi

The terminal score uses Tromp-Taylor-style area scoring:

- Black receives one point for every black stone on the board.
- White receives one point for every white stone on the board.
- Each maximal orthogonally connected empty region is awarded to a color only
  when every stone bordering that region has that color.
- An empty region bordered by both colors, or by neither color, is neutral and
  scores for neither player.
- White receives the configured komi.

Captured stones are not counted separately; their effect is already reflected
by the final board area. In formulas:

```text
black_score = black_stones + black_owned_empty_points
white_score = white_stones + white_owned_empty_points + komi
```

Komi must be finite and may be integral or fractional. Equal totals are a draw.

## Outcome perspective

Outcome is defined only for a terminal state. From a requested player's
perspective it is:

```text
+1  requested player wins
 0  draw
-1  requested player loses
```

Changing the requested perspective negates a decisive outcome and leaves a
draw at zero. This perspective contract is intended to remain stable when
neural evaluation and search are added in later milestones.

## Immutability and history

Applying a legal action returns a new state. It never mutates the parent state
or any board/history storage reachable from it. Boards are stored in immutable
byte-backed form, and chronological board history includes the initial board
and every subsequent legal action, including duplicate positions produced by
passes. Failed legality checks and failed applications likewise leave the
source state unchanged.

The Phase 1-2 engine does not implement handicap placement, territory scoring,
dead-stone adjudication, resignation, or selectable alternate ko and suicide
rules.
