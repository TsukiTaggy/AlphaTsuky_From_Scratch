"""Tests for deterministic, engine-validated SGF FF[4] game records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from azgo.game import GameState, Rules
from azgo.sgf import (
    SgfError,
    SgfGameRecord,
    SgfSyntaxError,
    load_sgf_collection,
    parse_sgf_collection,
    save_sgf_collection,
    serialize_sgf_collection,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(
    rules: Rules,
    actions: tuple[int, ...] | None = None,
    *,
    name: str = "game",
) -> SgfGameRecord:
    played = (rules.pass_action, rules.pass_action) if actions is None else actions
    state = GameState.new(rules, zobrist_seed=0)
    for action in played:
        state = state.apply(action)
    return SgfGameRecord(
        rules,
        played,
        state.score(),
        game_name=name,
        black_player="Black ] \\ player",
        white_player="白",
    )


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_canonical_round_trip_on_every_supported_board(board_size: int) -> None:
    rules = Rules(board_size=board_size, komi=7.5)
    actions = (0, board_size * board_size - 1, rules.pass_action, rules.pass_action)
    record = _record(rules, actions)

    text = serialize_sgf_collection((record,))
    loaded = parse_sgf_collection(text, expected_rules=rules, zobrist_seed=123)

    assert loaded == (record,)
    assert text.startswith("(;FF[4]GM[1]CA[UTF-8]AP[alphazero-go:")
    assert f"SZ[{board_size}]KM[7.5]RU[Tromp-Taylor]" in text
    assert ";B[aa]" in text
    corner = chr(ord("a") + board_size - 1) * 2
    assert f";W[{corner}]" in text
    assert text.endswith(";B[];W[])\n")
    assert "PB[Black \\] \\\\ player]" in text
    assert "PW[白]" in text


def test_collection_preserves_game_order_and_ignores_annotations() -> None:
    rules = Rules(board_size=5, komi=7.5)
    first = _record(rules, name="first")
    second = _record(rules, (0, 1, 25, 25), name="second")
    text = serialize_sgf_collection((first, second))
    annotated = text.replace("GN[first]", "GN[first]C[root note]").replace(
        ";B[]", ";B[]C[move note]", 1
    )

    loaded = parse_sgf_collection(annotated, expected_rules=rules, zobrist_seed=0)

    assert [record.game_name for record in loaded] == ["first", "second"]
    assert [record.actions for record in loaded] == [first.actions, second.actions]


def test_draw_result_is_canonical_zero() -> None:
    rules = Rules(board_size=5, komi=0.0)
    text = serialize_sgf_collection((_record(rules),))

    assert "RE[0]" in text
    assert parse_sgf_collection(text, expected_rules=rules, zobrist_seed=0)[0].winner is None


def test_small_real_values_use_sgf_decimal_syntax_without_exponents() -> None:
    rules = Rules(board_size=5, komi=1e-20)
    text = serialize_sgf_collection((_record(rules),))

    assert "e-" not in text.casefold()
    assert parse_sgf_collection(text, expected_rules=rules, zobrist_seed=0)[0].rules == rules


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "at least one game tree"),
        ("(;FF[4]", "expected '\\)'"),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+7.5]"
            ";B[](;W[])(;W[aa]))",
            "variations",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+7.5]"
            "AB[aa];B[];W[])",
            "setup and handicap",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+R]"
            ";B[];W[])",
            "real number",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+7.5]"
            ";W[];B[])",
            "must be played by B",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+7.5]"
            ";B[aa];W[aa];B[];W[])",
            "illegal move",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[W+7.5]"
            ";B[])",
            "terminate with two consecutive passes",
        ),
        (
            "(;FF[4]GM[1]CA[UTF-8]SZ[5]KM[7.5]RU[Tromp-Taylor]RE[B+1.0]"
            ";B[];W[])",
            "does not match",
        ),
    ],
)
def test_parser_rejects_unsupported_or_invalid_games(text: str, match: str) -> None:
    error = SgfSyntaxError if "expected" in match or "variations" in match else SgfError
    with pytest.raises(error, match=match):
        parse_sgf_collection(
            text,
            expected_rules=Rules(board_size=5, komi=7.5),
            zobrist_seed=0,
        )


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (("SZ[5]", "SZ[9]"), "does not match board size"),
        (("KM[7.5]", "KM[6.5]"), "does not match komi"),
        (("RU[Tromp-Taylor]", "RU[Japanese]"), "unsupported SGF ruleset"),
        (("CA[UTF-8]", "CA[Latin-1]"), "must be 'UTF-8'"),
        (("KM[7.5]", "KM[7.5e0]"), "real number"),
    ],
)
def test_parser_requires_compatible_root_metadata(
    replacement: tuple[str, str],
    match: str,
) -> None:
    rules = Rules(board_size=5, komi=7.5)
    text = serialize_sgf_collection((_record(rules),)).replace(*replacement)

    with pytest.raises(SgfError, match=match):
        parse_sgf_collection(text, expected_rules=rules, zobrist_seed=0)


def test_atomic_save_round_trip_and_failure_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = Rules(board_size=5, komi=7.5)
    record = _record(rules)
    destination = tmp_path / "nested" / "games.sgf"
    save_sgf_collection(destination, (record,))
    original = destination.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        del source, target
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("azgo.sgf.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        save_sgf_collection(destination, (_record(rules, name="other"),))

    assert destination.read_bytes() == original
    assert load_sgf_collection(
        destination,
        expected_rules=rules,
        zobrist_seed=0,
    ) == (record,)
    assert not tuple(destination.parent.glob("*.tmp"))


def test_record_is_frozen_and_rejects_nonterminal_or_wrong_score() -> None:
    rules = Rules(board_size=5, komi=7.5)
    record = _record(rules)
    with pytest.raises(FrozenInstanceError):
        record.game_name = "changed"  # type: ignore[misc]
    with pytest.raises(SgfError, match="terminate"):
        SgfGameRecord(rules, (0,), record.final_score)
    with pytest.raises(SgfError, match="exactly match"):
        SgfGameRecord(rules, (25, 25), _record(Rules(5, 0.0)).final_score)
