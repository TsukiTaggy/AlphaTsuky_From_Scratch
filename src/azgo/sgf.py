"""Deterministic SGF FF[4] records for complete Tromp--Taylor games."""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import NoReturn

from azgo._version import __version__
from azgo.game import Color, GameState, Rules, Ruleset, Score, action_to_coord, coord_to_action

_REAL_PATTERN = re.compile(r"[+-]?\d+(?:\.\d*)?\Z")


class SgfError(ValueError):
    """Raised when an SGF record or operation violates the supported contract."""


class SgfSyntaxError(SgfError):
    """Raised when SGF text does not satisfy the FF[4] collection grammar."""


@dataclass(frozen=True, slots=True)
class SgfGameRecord:
    """One immutable, normally terminated linear Go game."""

    rules: Rules
    actions: tuple[int, ...]
    final_score: Score
    game_name: str = ""
    black_player: str = ""
    white_player: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rules, Rules):
            raise SgfError("rules must be Rules")
        try:
            actions = tuple(self.actions)
        except TypeError as exc:
            raise SgfError("actions must be iterable") from exc
        for index, action in enumerate(actions):
            if isinstance(action, bool) or not isinstance(action, int):
                raise SgfError(f"actions[{index}] must be an integer")
            if not 0 <= action < self.rules.action_size:
                raise SgfError(
                    f"actions[{index}] must be in [0, {self.rules.action_size - 1}]"
                )
        for name in ("game_name", "black_player", "white_player"):
            if type(getattr(self, name)) is not str:
                raise SgfError(f"{name} must be a string")

        state = _replay(self.rules, actions, zobrist_seed=0)
        score = state.score()
        if not isinstance(self.final_score, Score) or self.final_score != score:
            raise SgfError("final_score must exactly match the replayed terminal position")
        object.__setattr__(self, "actions", actions)

    @property
    def winner(self) -> Color | None:
        """Return the winner implied by the validated final score."""

        return self.final_score.winner

    @property
    def move_count(self) -> int:
        """Return the number of moves, including passes."""

        return len(self.actions)


def serialize_sgf_collection(records: tuple[SgfGameRecord, ...]) -> str:
    """Serialize a non-empty record collection to canonical UTF-8 SGF FF[4]."""

    try:
        normalized = tuple(records)
    except TypeError as exc:
        raise SgfError("records must be iterable") from exc
    if not normalized:
        raise SgfError("records must contain at least one game")
    if any(not isinstance(record, SgfGameRecord) for record in normalized):
        raise SgfError("records must contain only SgfGameRecord objects")
    return "".join(_serialize_game(record) for record in normalized) + "\n"


def parse_sgf_collection(
    text: str,
    *,
    expected_rules: Rules,
    zobrist_seed: int,
) -> tuple[SgfGameRecord, ...]:
    """Parse and engine-validate a supported linear SGF FF[4] collection."""

    if type(text) is not str:
        raise SgfError("SGF text must be a string")
    if not isinstance(expected_rules, Rules):
        raise SgfError("expected_rules must be Rules")
    if isinstance(zobrist_seed, bool) or not isinstance(zobrist_seed, int):
        raise SgfError("zobrist_seed must be an unsigned 64-bit integer")
    if not 0 <= zobrist_seed <= (1 << 64) - 1:
        raise SgfError("zobrist_seed must be an unsigned 64-bit integer")

    trees = _Parser(text).parse_collection()
    return tuple(
        _record_from_nodes(
            nodes,
            expected_rules=expected_rules,
            zobrist_seed=zobrist_seed,
            game_index=index,
        )
        for index, nodes in enumerate(trees)
    )


def save_sgf_collection(path: str | Path, records: tuple[SgfGameRecord, ...]) -> Path:
    """Atomically save a canonical SGF collection and return its resolved path."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_sgf_collection(records)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)  # noqa: PTH105
        temporary = None
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()
    return target


def load_sgf_collection(
    path: str | Path,
    *,
    expected_rules: Rules,
    zobrist_seed: int,
) -> tuple[SgfGameRecord, ...]:
    """Load a UTF-8 SGF collection and validate it against the expected rules."""

    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise SgfError("SGF file must be valid UTF-8") from exc
    return parse_sgf_collection(
        text,
        expected_rules=expected_rules,
        zobrist_seed=zobrist_seed,
    )


type _Properties = dict[str, tuple[str, ...]]


class _Parser:
    """Small FF[4] parser that rejects variation trees before semantic import."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._index = 0

    def parse_collection(self) -> tuple[tuple[_Properties, ...], ...]:
        trees: list[tuple[_Properties, ...]] = []
        self._skip_whitespace()
        while self._index < len(self._text):
            trees.append(self._parse_tree())
            self._skip_whitespace()
        if not trees:
            self._fail("collection must contain at least one game tree")
        return tuple(trees)

    def _parse_tree(self) -> tuple[_Properties, ...]:
        self._consume("(")
        self._skip_whitespace()
        nodes: list[_Properties] = []
        while self._peek() == ";":
            nodes.append(self._parse_node())
            self._skip_whitespace()
        if not nodes:
            self._fail("game tree must contain at least one node")
        if self._peek() == "(":
            self._fail("variations are not supported")
        self._consume(")")
        return tuple(nodes)

    def _parse_node(self) -> _Properties:
        self._consume(";")
        properties: dict[str, tuple[str, ...]] = {}
        self._skip_whitespace()
        while (character := self._peek()) is not None and character.isalpha():
            identifier = self._parse_identifier()
            if identifier in properties:
                self._fail(f"duplicate property {identifier}")
            self._skip_whitespace()
            values: list[str] = []
            while self._peek() == "[":
                values.append(self._parse_value())
                self._skip_whitespace()
            if not values:
                self._fail(f"property {identifier} must contain a value")
            properties[identifier] = tuple(values)
        return properties

    def _parse_identifier(self) -> str:
        start = self._index
        while (character := self._peek()) is not None and character.isalpha():
            if not "A" <= character <= "Z":
                self._fail("property identifiers must contain only uppercase ASCII letters")
            self._index += 1
        return self._text[start : self._index]

    def _parse_value(self) -> str:
        self._consume("[")
        value: list[str] = []
        while True:
            character = self._peek()
            if character is None:
                self._fail("unterminated property value")
            if character == "]":
                self._index += 1
                return "".join(value)
            if character != "\\":
                value.append(character)
                self._index += 1
                continue
            self._index += 1
            escaped = self._peek()
            if escaped is None:
                self._fail("property value ends with an incomplete escape")
            if escaped == "\r":
                self._index += 1
                if self._peek() == "\n":
                    self._index += 1
                continue
            if escaped == "\n":
                self._index += 1
                continue
            value.append(escaped)
            self._index += 1

    def _skip_whitespace(self) -> None:
        while (character := self._peek()) is not None and character.isspace():
            self._index += 1

    def _consume(self, expected: str) -> None:
        if self._peek() != expected:
            self._fail(f"expected {expected!r}")
        self._index += 1

    def _peek(self) -> str | None:
        if self._index >= len(self._text):
            return None
        return self._text[self._index]

    def _fail(self, message: str) -> NoReturn:
        raise SgfSyntaxError(f"{message} at character {self._index}")


def _record_from_nodes(
    nodes: tuple[_Properties, ...],
    *,
    expected_rules: Rules,
    zobrist_seed: int,
    game_index: int,
) -> SgfGameRecord:
    root = nodes[0]
    _reject_unsupported_properties(nodes)
    _require_scalar(root, "FF", expected="4")
    _require_scalar(root, "GM", expected="1")
    _require_scalar(root, "CA", expected="UTF-8", casefold=True)
    board_size = _parse_integer(_require_scalar(root, "SZ"), "SZ")
    if board_size != expected_rules.board_size:
        raise SgfError(
            f"game {game_index} SZ[{board_size}] does not match board size "
            f"{expected_rules.board_size}"
        )
    komi = _parse_real(_require_scalar(root, "KM"), "KM")
    if komi != expected_rules.komi:
        raise SgfError(
            f"game {game_index} KM[{komi}] does not match komi {expected_rules.komi}"
        )
    ruleset = _require_scalar(root, "RU")
    if _normalize_ruleset(ruleset) is not expected_rules.ruleset:
        raise SgfError(f"game {game_index} uses unsupported rules RU[{ruleset}]")
    declared_result = _require_scalar(root, "RE")
    if "B" in root or "W" in root:
        raise SgfError(f"game {game_index} root node must not contain a move")

    actions: list[int] = []
    for move_index, node in enumerate(nodes[1:]):
        move_properties = [name for name in ("B", "W") if name in node]
        if len(move_properties) != 1:
            raise SgfError(
                f"game {game_index} move node {move_index + 1} must contain exactly one move"
            )
        expected_color = "B" if move_index % 2 == 0 else "W"
        color = move_properties[0]
        if color != expected_color:
            raise SgfError(
                f"game {game_index} move {move_index} must be played by {expected_color}"
            )
        value = _require_scalar(node, color)
        actions.append(_sgf_value_to_action(value, expected_rules.board_size))

    try:
        state = _replay(expected_rules, tuple(actions), zobrist_seed=zobrist_seed)
    except SgfError as exc:
        raise SgfError(f"game {game_index} is invalid: {exc}") from exc
    score = state.score()
    _validate_declared_result(declared_result, score, game_index)
    return SgfGameRecord(
        rules=expected_rules,
        actions=tuple(actions),
        final_score=score,
        game_name=_optional_scalar(root, "GN"),
        black_player=_optional_scalar(root, "PB"),
        white_player=_optional_scalar(root, "PW"),
    )


def _reject_unsupported_properties(nodes: tuple[_Properties, ...]) -> None:
    unsupported = {"AB", "AW", "AE", "HA"}
    for node_index, node in enumerate(nodes):
        found = sorted(unsupported.intersection(node))
        if found:
            raise SgfError(
                f"setup and handicap properties are not supported: "
                f"node {node_index} contains {', '.join(found)}"
            )


def _serialize_game(record: SgfGameRecord) -> str:
    properties = [
        "FF[4]",
        "GM[1]",
        "CA[UTF-8]",
        f"AP[{_escape_simple_text(f'alphazero-go:{__version__}')}]",
    ]
    properties.extend(
        (
            f"SZ[{record.rules.board_size}]",
            f"KM[{_format_real(record.rules.komi)}]",
            "RU[Tromp-Taylor]",
            f"RE[{_result_text(record.final_score)}]",
        )
    )
    if record.game_name:
        properties.append(f"GN[{_escape_simple_text(record.game_name)}]")
    if record.black_player:
        properties.append(f"PB[{_escape_simple_text(record.black_player)}]")
    if record.white_player:
        properties.append(f"PW[{_escape_simple_text(record.white_player)}]")

    nodes = ["(;", "".join(properties)]
    for index, action in enumerate(record.actions):
        color = "B" if index % 2 == 0 else "W"
        nodes.extend((f";{color}[", _action_to_sgf_value(action, record.rules.board_size), "]"))
    nodes.append(")")
    return "".join(nodes)


def _replay(rules: Rules, actions: tuple[int, ...], *, zobrist_seed: int) -> GameState:
    state = GameState.new(rules, zobrist_seed=zobrist_seed)
    for move_index, action in enumerate(actions):
        try:
            state = state.apply(action)
        except ValueError as exc:
            raise SgfError(f"illegal move {move_index}: {exc}") from exc
    if not state.is_terminal:
        raise SgfError("game must terminate with two consecutive passes")
    return state


def _action_to_sgf_value(action: int, board_size: int) -> str:
    coordinate = action_to_coord(action, board_size)
    if coordinate is None:
        return ""
    row, column = coordinate
    return chr(ord("a") + column) + chr(ord("a") + row)


def _sgf_value_to_action(value: str, board_size: int) -> int:
    if value == "":
        return board_size * board_size
    if len(value) != 2 or any(not "a" <= character <= "z" for character in value):
        raise SgfError(f"invalid SGF move coordinate {value!r}")
    column = ord(value[0]) - ord("a")
    row = ord(value[1]) - ord("a")
    try:
        return coord_to_action(row, column, board_size)
    except ValueError as exc:
        raise SgfError(f"SGF move coordinate {value!r} is outside the board") from exc


def _require_scalar(
    properties: _Properties,
    name: str,
    *,
    expected: str | None = None,
    casefold: bool = False,
) -> str:
    values = properties.get(name)
    if values is None:
        raise SgfError(f"required SGF property {name} is missing")
    if len(values) != 1:
        raise SgfError(f"SGF property {name} must contain exactly one value")
    value = values[0]
    if expected is not None:
        actual_comparison = value.casefold() if casefold else value
        expected_comparison = expected.casefold() if casefold else expected
        if actual_comparison != expected_comparison:
            raise SgfError(f"SGF property {name} must be {expected!r}")
    return value


def _optional_scalar(properties: _Properties, name: str) -> str:
    values = properties.get(name)
    if values is None:
        return ""
    if len(values) != 1:
        raise SgfError(f"SGF property {name} must contain exactly one value")
    return values[0]


def _parse_integer(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SgfError(f"SGF property {name} must be an integer") from exc
    if str(parsed) != value:
        raise SgfError(f"SGF property {name} must use canonical integer syntax")
    return parsed


def _parse_real(value: str, name: str) -> float:
    if _REAL_PATTERN.fullmatch(value) is None:
        raise SgfError(f"SGF property {name} must be a real number")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SgfError(f"SGF property {name} must be a real number") from exc
    if not isfinite(parsed):
        raise SgfError(f"SGF property {name} must be finite")
    return parsed


def _normalize_ruleset(value: str) -> Ruleset:
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    if normalized != "tromptaylor":
        raise SgfError(f"unsupported SGF ruleset {value!r}")
    return Ruleset.TROMP_TAYLOR


def _validate_declared_result(value: str, score: Score, game_index: int) -> None:
    winner = score.winner
    if value in {"0", "Draw"}:
        if winner is not None:
            raise SgfError(f"game {game_index} declared a draw but the score has a winner")
        return
    if len(value) < 3 or value[0] not in {"B", "W"} or value[1] != "+":
        raise SgfError(
            f"game {game_index} result must be a draw or a numeric Black/White win"
        )
    margin = _parse_real(value[2:], "RE")
    if margin <= 0.0:
        raise SgfError(f"game {game_index} result margin must be positive")
    declared_winner = Color.BLACK if value[0] == "B" else Color.WHITE
    actual_margin = abs(score.black_margin)
    if winner is not declared_winner or margin != actual_margin:
        raise SgfError(f"game {game_index} declared result does not match its final score")


def _result_text(score: Score) -> str:
    winner = score.winner
    if winner is None:
        return "0"
    prefix = "B" if winner is Color.BLACK else "W"
    return f"{prefix}+{_format_real(abs(score.black_margin))}"


def _format_real(value: float) -> str:
    if not isfinite(value):
        raise SgfError("SGF real values must be finite")
    text = format(float(value), ".17g")
    if "e" in text.casefold():
        text = format(Decimal(text), "f")
    return "0" if text in {"-0", "+0"} else text


def _escape_simple_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = " ".join(normalized.splitlines())
    return normalized.replace("\\", "\\\\").replace("]", "\\]")


__all__ = [
    "SgfError",
    "SgfGameRecord",
    "SgfSyntaxError",
    "load_sgf_collection",
    "parse_sgf_collection",
    "save_sgf_collection",
    "serialize_sgf_collection",
]
