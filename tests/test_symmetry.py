"""Square-board action, feature, and policy symmetry tests."""

import numpy as np
import pytest

from azgo.symmetry import Symmetry


def test_symmetry_has_exactly_eight_operations() -> None:
    assert len(Symmetry) == 8


@pytest.mark.parametrize(
    ("symmetry", "expected_action"),
    [
        (Symmetry.IDENTITY, 1),
        (Symmetry.ROTATE_90, 15),
        (Symmetry.ROTATE_180, 23),
        (Symmetry.ROTATE_270, 9),
        (Symmetry.FLIP_HORIZONTAL, 21),
        (Symmetry.FLIP_VERTICAL, 3),
        (Symmetry.FLIP_MAIN_DIAGONAL, 5),
        (Symmetry.FLIP_ANTI_DIAGONAL, 19),
    ],
)
def test_action_mapping_is_unambiguous(symmetry: Symmetry, expected_action: int) -> None:
    # Action 1 is coordinate (0, 1) on a 5x5 board.
    assert symmetry.transform_action(1, 5) == expected_action


@pytest.mark.parametrize("board_size", [5, 9, 13, 19])
@pytest.mark.parametrize("symmetry", list(Symmetry))
def test_actions_and_pass_round_trip(board_size: int, symmetry: Symmetry) -> None:
    for action in range(board_size * board_size + 1):
        transformed = symmetry.transform_action(action, board_size)
        assert symmetry.inverse.transform_action(transformed, board_size) == action

    pass_action = board_size * board_size
    assert symmetry.transform_action(pass_action, board_size) == pass_action


@pytest.mark.parametrize("symmetry", list(Symmetry))
def test_features_round_trip_with_leading_dimensions(symmetry: Symmetry) -> None:
    storage = np.arange(2 * 3 * 5 * 10, dtype=np.float32).reshape(2, 3, 5, 10)
    features = storage[..., ::2]

    transformed = symmetry.transform_features(features)
    restored = symmetry.inverse.transform_features(transformed)

    assert not features.flags.c_contiguous
    assert transformed.flags.c_contiguous
    assert transformed.shape == features.shape
    assert transformed.dtype == features.dtype
    np.testing.assert_array_equal(restored, features)


@pytest.mark.parametrize("symmetry", list(Symmetry))
def test_policy_matches_action_and_feature_transforms(symmetry: Symmetry) -> None:
    board_size = 5
    action = 1
    pass_action = board_size * board_size
    feature = np.zeros((1, board_size, board_size), dtype=np.int16)
    feature[0, 0, 1] = 7
    policy = np.zeros((2, pass_action + 1), dtype=np.float64)
    policy[0, action] = 0.75
    policy[0, pass_action] = 0.25
    policy[1] = np.arange(pass_action + 1)

    transformed_feature = symmetry.transform_features(feature)
    transformed_policy = symmetry.transform_policy(policy, board_size)
    transformed_action = symmetry.transform_action(action, board_size)

    assert transformed_feature.reshape(-1)[transformed_action] == 7
    assert transformed_policy.shape == policy.shape
    assert transformed_policy.dtype == policy.dtype
    assert transformed_policy[0, transformed_action] == 0.75
    np.testing.assert_array_equal(transformed_policy[..., -1], policy[..., -1])
    np.testing.assert_allclose(transformed_policy.sum(axis=-1), policy.sum(axis=-1))
    np.testing.assert_array_equal(
        symmetry.inverse.transform_policy(transformed_policy, board_size),
        policy,
    )


@pytest.mark.parametrize("symmetry", list(Symmetry))
def test_single_board_feature_tensor_is_supported(symmetry: Symmetry) -> None:
    board = np.arange(25, dtype=np.uint8).reshape(5, 5)
    assert symmetry.transform_features(board).shape == (5, 5)


@pytest.mark.parametrize("board_size", [True, 0, -1, 5.5])
def test_symmetry_rejects_invalid_board_size(board_size: object) -> None:
    with pytest.raises(ValueError, match="board_size"):
        Symmetry.IDENTITY.transform_action(0, board_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("action", [True, -1, 26, 1.5])
def test_symmetry_rejects_invalid_action(action: object) -> None:
    with pytest.raises(ValueError, match="action"):
        Symmetry.IDENTITY.transform_action(action, 5)  # type: ignore[arg-type]


def test_symmetry_rejects_malformed_feature_and_policy_shapes() -> None:
    with pytest.raises(ValueError, match="two dimensions"):
        Symmetry.IDENTITY.transform_features(np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="square"):
        Symmetry.IDENTITY.transform_features(np.zeros((5, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="final dimension 26"):
        Symmetry.IDENTITY.transform_policy(np.zeros(25, dtype=np.float32), 5)
