import numpy as np
import pytest
from BloodPressureSim.Action import Action


def expected_bits(action_idx: int):
    """
    Decode action_idx (0..15) into the 4 binary flags:
    iv, steroid, epinephrine, phenylephrine
    using the same bit significance as get_action_idx().
    """
    assert 0 <= action_idx < 16
    iv = (action_idx >> 3) & 1
    steroid = (action_idx >> 2) & 1
    epinephrine = (action_idx >> 1) & 1
    phenylephrine = action_idx & 1
    return iv, steroid, epinephrine, phenylephrine


@pytest.mark.parametrize("action_idx", list(range(16)))
def test_fields_match_expected_bits(action_idx):
    """
    For each index 0..15, constructor should decode the correct flags.
    """
    a = Action(action_idx=action_idx)
    iv, steroid, epi, phenyl = expected_bits(action_idx)

    assert a.iv == iv
    assert a.steroid == steroid
    assert a.epinephrine == epi
    assert a.phenylephrine == phenyl


@pytest.mark.parametrize("action_idx", list(range(16)))
def test_get_action_idx_round_trip(action_idx):
    """
    Action(action_idx=i).get_action_idx() should return i.
    """
    a = Action(action_idx=action_idx)
    assert a.get_action_idx() == action_idx


def test_equality_and_inequality():
    a1 = Action(action_idx=0)
    a2 = Action(action_idx=0)
    a3 = Action(action_idx=1)

    assert a1 == a2
    assert not (a1 != a2)

    assert a1 != a3
    assert not (a1 == a3)

    # Type mismatch should not be equal
    assert a1 != object()


def test_hash_matches_action_idx_and_set_behavior():
    """
    __hash__ returns get_action_idx, so identical actions should dedupe in sets.
    """
    a1 = Action(action_idx=7)
    a2 = Action(action_idx=7)
    a3 = Action(action_idx=8)

    assert hash(a1) == 7
    assert hash(a2) == 7
    assert hash(a3) == 8

    s = {a1, a2, a3}
    assert len(s) == 2
    assert a1 in s
    assert a3 in s


@pytest.mark.parametrize("action_idx", list(range(16)))
def test_get_action_vec_shape_and_values(action_idx):
    a = Action(action_idx=action_idx)
    vec = a.get_action_vec()

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (4, 1)

    iv, steroid, epi, phenyl = expected_bits(action_idx)
    assert np.array_equal(vec, np.array([[iv], [steroid], [epi], [phenyl]]))


def test_constructor_requires_action_idx():
    with pytest.raises(AssertionError):
        Action(action_idx=None)


def test_constants_are_consistent():
    assert Action.NUM_ACTIONS_TOTAL == 16
    assert Action.ACTION_VEC_SIZE == 4
