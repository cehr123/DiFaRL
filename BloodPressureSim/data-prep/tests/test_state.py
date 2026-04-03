# tests/test_state.py
import numpy as np
import pytest

from BloodPressureSim.State import State


def test_constructor_requires_idx_or_categs():
    with pytest.raises(AssertionError):
        State()


@pytest.mark.parametrize("state_idx", [0, 1, 4, 5, 24, 25, 62, 124])
def test_set_state_by_idx_round_trip(state_idx):
    """
    State(state_idx=i).get_state_idx() should return i (for valid range).
    """
    s = State(state_idx=state_idx)
    assert s.get_state_idx() == state_idx


@pytest.mark.parametrize(
    "state_categs, expected_idx",
    [
        ([0, 0, 0], 0),
        ([0, 0, 1], 1),
        ([0, 0, 4], 4),
        ([0, 1, 0], 5),
        ([1, 0, 0], 25),
        ([4, 4, 4], 124),
        ([2, 3, 4], (2 * 25) + (3 * 5) + 4),
    ],
)
def test_constructor_with_state_categs_sets_fields_and_idx(state_categs, expected_idx):
    s = State(state_categs=state_categs)
    assert s.vascular_resistence_state == state_categs[0]
    assert s.heart_contraction_state == state_categs[1]
    assert s.blood_volume_state == state_categs[2]
    assert s.get_state_idx() == expected_idx


@pytest.mark.parametrize("state_idx", [0, 7, 19, 63, 124])
def test_set_state_by_idx_decodes_components_correctly(state_idx):
    """
    Ensure decoding matches encoding:
      idx = vr*25 + hc*5 + bv
    """
    s = State(state_idx=state_idx)

    vr = state_idx // 25
    rem = state_idx % 25
    hc = rem // 5
    bv = rem % 5

    assert s.vascular_resistence_state == vr
    assert s.heart_contraction_state == hc
    assert s.blood_volume_state == bv


def test_equality_and_inequality():
    s1 = State(state_categs=[1, 2, 3])
    s2 = State(state_categs=[1, 2, 3])
    s3 = State(state_categs=[1, 2, 4])

    assert s1 == s2
    assert s1 != s3
    assert (s1 == s3) is False


def test_hash_matches_state_idx_and_set_behavior():
    """
    __hash__ returns get_state_idx, so identical states should dedupe in sets.
    """
    s1 = State(state_idx=42)
    s2 = State(state_idx=42)
    s3 = State(state_idx=43)

    assert hash(s1) == 42
    assert hash(s2) == 42
    assert hash(s3) == 43

    st = {s1, s2, s3}
    assert len(st) == 2
    assert s1 in st
    assert s3 in st


def test_copy_state_produces_equal_but_distinct_object():
    s1 = State(state_categs=[4, 0, 2])
    s2 = s1.copy_state()

    assert s2 == s1
    assert s2 is not s1

    # Mutating the copy should not affect the original
    s2.vascular_resistence_state = 0
    assert s2 != s1
    assert s1.vascular_resistence_state == 4


@pytest.mark.parametrize("state_categs", [[0, 0, 0], [4, 4, 4], [2, 1, 3]])
def test_get_state_vector_shape_dtype_and_values(state_categs):
    s = State(state_categs=state_categs)
    v = s.get_state_vector()

    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    assert np.issubdtype(v.dtype, np.integer)
    assert v.tolist() == state_categs
