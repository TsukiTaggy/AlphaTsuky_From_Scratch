"""Bounded replay-buffer and portable snapshot tests."""

from pathlib import Path

import numpy as np
import pytest

from azgo.game import Color, Score
from azgo.replay import ReplayBuffer, ReplayError
from azgo.self_play import SelfPlayGame, TrainingSample
from azgo.symmetry import Symmetry

UINT64_MAX = (1 << 64) - 1


def make_sample(
    *,
    action: int = 0,
    move_number: int = 0,
    game_index: int = 0,
    to_play: Color = Color.BLACK,
    board_size: int = 5,
    history_length: int = 1,
    value: float = 1.0,
) -> TrainingSample:
    features = np.zeros(
        (2 * history_length + 1, board_size, board_size),
        dtype=np.float32,
    )
    if action < board_size * board_size:
        row, column = divmod(action, board_size)
        features[0, row, column] = 1.0
    policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
    policy[action] = 1.0
    return TrainingSample(
        features=features,
        policy=policy,
        value=value,
        to_play=to_play,
        move_number=move_number,
        selected_action=action,
        game_index=game_index,
    )


def make_game(game_index: int = 4) -> SelfPlayGame:
    samples = (
        make_sample(action=3, move_number=0, game_index=game_index, value=-1.0),
        make_sample(
            action=25,
            move_number=1,
            game_index=game_index,
            to_play=Color.WHITE,
            value=1.0,
        ),
    )
    final_score = Score(
        black_stones=0,
        white_stones=0,
        black_territory=0,
        white_territory=0,
        neutral_points=25,
        komi=7.5,
    )
    return SelfPlayGame(
        samples=samples,
        actions=(3, 25),
        final_score=final_score,
        winner=Color.WHITE,
        game_index=game_index,
    )


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
def test_buffer_accepts_supported_board_sizes(board_size: int) -> None:
    buffer = ReplayBuffer(board_size, history_length=2, capacity=3)
    assert buffer.board_size == board_size
    assert buffer.history_length == 2
    assert buffer.capacity == 3
    assert buffer.next_game_index == 0
    assert len(buffer) == 0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("board_size", True),
        ("board_size", 7),
        ("history_length", True),
        ("history_length", 0),
        ("history_length", 1.5),
        ("capacity", True),
        ("capacity", 0),
        ("capacity", -1),
        ("next_game_index", True),
        ("next_game_index", -1),
        ("next_game_index", UINT64_MAX + 1),
    ],
)
def test_constructor_rejects_invalid_metadata(name: str, value: object) -> None:
    arguments: dict[str, object] = {
        "board_size": 5,
        "history_length": 1,
        "capacity": 3,
        "next_game_index": 0,
    }
    arguments[name] = value
    with pytest.raises(ReplayError, match=name):
        ReplayBuffer(**arguments)  # type: ignore[arg-type]


def test_fifo_eviction_and_next_game_index() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=2)
    for game_index in range(3):
        buffer.append(make_sample(game_index=game_index, action=game_index))

    batch = buffer.sample(2, seed=8)
    assert len(buffer) == 2
    assert set(batch.game_indices.tolist()) == {1, 2}
    assert buffer.next_game_index == 3


def test_extend_is_transactional_for_mismatched_sample() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=4)
    valid = make_sample(game_index=2)
    mismatched = make_sample(board_size=9, game_index=3)

    with pytest.raises(ReplayError, match="features"):
        buffer.extend((valid, mismatched))

    assert len(buffer) == 0
    assert buffer.next_game_index == 0


def test_add_game_preserves_all_positions_and_metadata() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=10)
    buffer.add_game(make_game())
    batch = buffer.sample(2, seed=2)

    assert len(buffer) == 2
    assert buffer.next_game_index == 5
    assert set(batch.move_numbers.tolist()) == {0, 1}
    assert set(batch.selected_actions.tolist()) == {3, 25}
    assert set(batch.game_indices.tolist()) == {4}


def test_add_game_is_transactional_for_mismatched_position_shape() -> None:
    first = make_sample(action=3, move_number=0, game_index=4, value=-1.0)
    mismatched = make_sample(
        action=25,
        move_number=1,
        game_index=4,
        to_play=Color.WHITE,
        board_size=9,
        value=1.0,
    )
    final_score = Score(0, 0, 0, 0, 25, 7.5)
    game = SelfPlayGame(
        samples=(first, mismatched),
        actions=(3, 25),
        final_score=final_score,
        winner=Color.WHITE,
        game_index=4,
    )
    buffer = ReplayBuffer(5, history_length=1, capacity=10)

    with pytest.raises(ReplayError, match="features"):
        buffer.add_game(game)

    assert len(buffer) == 0
    assert buffer.next_game_index == 0


def test_uint64_game_index_edge_is_transactional() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=3, next_game_index=UINT64_MAX)
    buffer.append(make_sample(game_index=2))
    assert buffer.next_game_index == UINT64_MAX

    with pytest.raises(ReplayError, match="next_game_index"):
        buffer.append(make_sample(game_index=UINT64_MAX))

    assert len(buffer) == 1
    assert buffer.next_game_index == UINT64_MAX


def test_sampling_is_seeded_and_without_replacement() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=8)
    buffer.extend(make_sample(action=index, game_index=index) for index in range(6))

    first = buffer.sample(5, seed=4321)
    second = buffer.sample(5, seed=4321)

    np.testing.assert_array_equal(first.game_indices, second.game_indices)
    np.testing.assert_array_equal(first.features, second.features)
    assert len(set(first.game_indices.tolist())) == 5


@pytest.mark.parametrize(
    ("batch_size", "seed", "augment", "message"),
    [
        (0, 0, False, "batch_size"),
        (2, 0, False, "exceeds"),
        (1, -1, False, "seed"),
        (1, UINT64_MAX + 1, False, "seed"),
        (1, 0, 1, "augment"),
    ],
)
def test_sampling_rejects_invalid_requests(
    batch_size: int,
    seed: int,
    augment: object,
    message: str,
) -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=2)
    buffer.append(make_sample())
    with pytest.raises(ReplayError, match=message):
        buffer.sample(batch_size, seed, augment=augment)  # type: ignore[arg-type]


def test_augmentation_transforms_features_policy_and_selected_action_together() -> None:
    seed = 51
    action = 1
    original = make_sample(action=action)
    buffer = ReplayBuffer(5, history_length=1, capacity=1)
    buffer.append(original)

    batch = buffer.sample(1, seed=seed, augment=True)
    repeated = buffer.sample(1, seed=seed, augment=True)
    rng = np.random.default_rng(seed)
    rng.choice(1, size=1, replace=False)
    symmetry = tuple(Symmetry)[int(rng.integers(0, len(Symmetry), size=1)[0])]
    transformed_action = symmetry.transform_action(action, 5)

    np.testing.assert_array_equal(batch.features, repeated.features)
    np.testing.assert_array_equal(batch.policies, repeated.policies)
    assert int(batch.selected_actions[0]) == transformed_action
    assert batch.features[0, 0].reshape(-1)[transformed_action] == 1.0
    assert batch.policies[0, transformed_action] == 1.0


def test_sampled_arrays_are_detached_contiguous_and_read_only() -> None:
    buffer = ReplayBuffer(5, history_length=1, capacity=2)
    buffer.append(make_sample())
    batch = buffer.sample(1, seed=0)

    for array in (
        batch.features,
        batch.policies,
        batch.values,
        batch.to_play,
        batch.move_numbers,
        batch.selected_actions,
        batch.game_indices,
    ):
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0

    assert batch.features.dtype == np.float32
    assert batch.policies.dtype == np.float32
    assert batch.values.dtype == np.float32
    assert batch.to_play.dtype == np.uint8
    assert batch.move_numbers.dtype == np.int64
    assert batch.selected_actions.dtype == np.int64
    assert batch.game_indices.dtype == np.uint64


def test_empty_snapshot_round_trip_and_parent_creation(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "empty.npz"
    ReplayBuffer(9, history_length=3, capacity=12, next_game_index=7).save(path)
    loaded = ReplayBuffer.load(path)

    assert path.is_file()
    assert loaded.board_size == 9
    assert loaded.history_length == 3
    assert loaded.capacity == 12
    assert loaded.next_game_index == 7
    assert len(loaded) == 0


def test_full_snapshot_round_trip_and_append_continuity(tmp_path: Path) -> None:
    path = tmp_path / "replay.npz"
    original = ReplayBuffer(5, history_length=1, capacity=3)
    original.extend(make_sample(action=index, game_index=index) for index in range(3))
    original.save(path)

    loaded = ReplayBuffer.load(path)
    expected = original.sample(3, seed=123)
    actual = loaded.sample(3, seed=123)
    for expected_array, actual_array in zip(
        (
            expected.features,
            expected.policies,
            expected.values,
            expected.to_play,
            expected.move_numbers,
            expected.selected_actions,
            expected.game_indices,
        ),
        (
            actual.features,
            actual.policies,
            actual.values,
            actual.to_play,
            actual.move_numbers,
            actual.selected_actions,
            actual.game_indices,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, expected_array)

    loaded.append(make_sample(action=4, game_index=loaded.next_game_index))
    assert loaded.next_game_index == 4
    assert set(loaded.sample(3, seed=9).game_indices.tolist()) == {1, 2, 3}


def snapshot_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as snapshot:
        return {name: np.array(snapshot[name], copy=True) for name in snapshot.files}


def write_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)  # type: ignore[arg-type]


def valid_snapshot(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    path = tmp_path / "snapshot.npz"
    buffer = ReplayBuffer(5, history_length=1, capacity=3)
    buffer.append(make_sample())
    buffer.save(path)
    return path, snapshot_payload(path)


def test_load_rejects_unsupported_version(tmp_path: Path) -> None:
    path, payload = valid_snapshot(tmp_path)
    payload["version"] = np.asarray(99, dtype=np.uint32)
    write_payload(path, payload)
    with pytest.raises(ReplayError, match="unsupported"):
        ReplayBuffer.load(path)


@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_load_rejects_non_exact_fields(tmp_path: Path, mode: str) -> None:
    path, payload = valid_snapshot(tmp_path)
    if mode == "missing":
        del payload["values"]
    else:
        payload["unexpected"] = np.asarray(1)
    write_payload(path, payload)
    with pytest.raises(ReplayError, match="fields"):
        ReplayBuffer.load(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("capacity", np.asarray(3, dtype=np.int64), "capacity"),
        ("features", np.zeros((1, 3, 4, 5), dtype=np.float32), "features"),
        ("policies", np.zeros((1, 26), dtype=np.float64), "policies"),
        ("values", np.zeros((1, 1), dtype=np.float32), "values"),
        ("to_play", np.ones(1, dtype=np.int8), "to_play"),
        ("move_numbers", np.zeros(1, dtype=np.int32), "move_numbers"),
        ("selected_actions", np.zeros(1, dtype=np.int32), "selected_actions"),
        ("game_indices", np.zeros(1, dtype=np.int64), "game_indices"),
    ],
)
def test_load_rejects_wrong_dtypes_and_shapes(
    tmp_path: Path,
    field: str,
    replacement: np.ndarray,
    message: str,
) -> None:
    path, payload = valid_snapshot(tmp_path)
    payload[field] = replacement
    write_payload(path, payload)
    with pytest.raises(ReplayError, match=message):
        ReplayBuffer.load(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("features", np.nan, "finite"),
        ("policies", np.nan, "finite"),
        ("policies", -1.0, "non-negative"),
        ("values", 2.0, "values"),
        ("to_play", 0, "to_play"),
        ("move_numbers", -1, "move_numbers"),
        ("selected_actions", 26, "selected_actions"),
    ],
)
def test_load_rejects_invalid_array_values(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    path, payload = valid_snapshot(tmp_path)
    payload[field].flat[0] = value
    write_payload(path, payload)
    with pytest.raises(ReplayError, match=message):
        ReplayBuffer.load(path)


def test_load_rejects_policy_that_is_not_normalized(tmp_path: Path) -> None:
    path, payload = valid_snapshot(tmp_path)
    payload["policies"].fill(0.0)
    write_payload(path, payload)
    with pytest.raises(ReplayError, match="sum to one"):
        ReplayBuffer.load(path)


def test_load_rejects_next_index_behind_stored_games(tmp_path: Path) -> None:
    path, payload = valid_snapshot(tmp_path)
    payload["game_indices"][0] = 2
    payload["next_game_index"] = np.asarray(2, dtype=np.uint64)
    write_payload(path, payload)
    with pytest.raises(ReplayError, match="next_game_index"):
        ReplayBuffer.load(path)


def test_load_wraps_missing_and_corrupt_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npz"
    with pytest.raises(ReplayError, match="could not load"):
        ReplayBuffer.load(missing)

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not a zip file")
    with pytest.raises(ReplayError, match="could not load"):
        ReplayBuffer.load(corrupt)


@pytest.mark.parametrize("failure_point", ["serialize", "replace"])
def test_atomic_save_preserves_existing_target_and_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    path = tmp_path / "snapshot.npz"
    original = ReplayBuffer(5, history_length=1, capacity=3)
    original.append(make_sample(game_index=0))
    original.save(path)
    original_bytes = path.read_bytes()

    replacement = ReplayBuffer(5, history_length=1, capacity=3)
    replacement.append(make_sample(game_index=1))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected save failure")

    if failure_point == "serialize":
        monkeypatch.setattr("azgo.replay.np.savez_compressed", fail)
    else:
        monkeypatch.setattr("azgo.replay.os.replace", fail)

    with pytest.raises(ReplayError, match="could not save"):
        replacement.save(path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".snapshot.npz.*.tmp")) == []
