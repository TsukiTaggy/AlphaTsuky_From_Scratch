"""Deterministic Zobrist table tests."""

import pytest

from azgo.game import Color, ZobristTable


def test_equal_seeds_create_equal_tables_and_hashes() -> None:
    first = ZobristTable.create(5, seed=20260710)
    second = ZobristTable.create(5, seed=20260710)
    board = bytes([Color.BLACK, Color.WHITE, Color.EMPTY] + [Color.EMPTY] * 22)

    assert first == second
    assert first.hash_board(board) == second.hash_board(board)


def test_hash_is_xor_of_stone_keys() -> None:
    table = ZobristTable.create(5, seed=7)
    board = bytearray(25)
    board[3] = Color.BLACK
    board[17] = Color.WHITE

    assert table.hash_board(board) == table.black_keys[3] ^ table.white_keys[17]


def test_custom_zero_keys_make_collisions_reproducible() -> None:
    zeroes = [0] * 25
    table = ZobristTable(
        5,
        seed=0,
        black_keys=zeroes,
        white_keys=zeroes,
        side_to_play_key=0,
    )

    assert table.hash_board(bytes(25)) == 0
    assert table.hash_board(bytes([Color.BLACK]) + bytes(24)) == 0


@pytest.mark.parametrize("seed", [-1, 2**64, True])
def test_seed_must_be_an_unsigned_64_bit_integer(seed: int) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        ZobristTable.create(5, seed=seed)


@pytest.mark.parametrize("bad_value", [-1, 3])
def test_hash_rejects_invalid_intersection_values(bad_value: int) -> None:
    table = ZobristTable.create(5, seed=0)
    board = [0] * 25
    board[4] = bad_value

    with pytest.raises(ValueError, match="invalid intersection value"):
        table.hash_board(board)
