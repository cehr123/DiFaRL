# tests/test_mdp.py
import numpy as np
import pytest

from BloodPressureSim.MDP import MDP
from BloodPressureSim.State import State
from BloodPressureSim.Action import Action


class DummyAction:
    """
    Minimal action object for exercising MDP.transition without depending on
    Action's internal bit encoding. MDP.transition only needs these attributes.
    """
    def __init__(self, iv=0, epinephrine=0, steroid=0, phenylephrine=0):
        self.iv = iv
        self.epinephrine = epinephrine
        self.steroid = steroid
        self.phenylephrine = phenylephrine


def _make_uniform_sequence(monkeypatch, values):
    """
    Monkeypatch np.random.uniform() to return a fixed sequence.
    """
    it = iter(values)

    def fake_uniform(*args, **kwargs):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("np.random.uniform() called more times than expected")

    monkeypatch.setattr(np.random, "uniform", fake_uniform)


def test_init_accepts_valid_policy_shape():
    policy = np.ones((State.NUM_STATES, Action.NUM_ACTIONS_TOTAL), dtype=float)
    policy /= policy.sum(axis=1, keepdims=True)  # each row sums to 1
    mdp = MDP(init_state_idx=0, policy_array=policy)
    assert mdp.policy_array.shape == (State.NUM_STATES, Action.NUM_ACTIONS_TOTAL)
    assert mdp.state.get_state_idx() == 0


def test_init_rejects_invalid_policy_shape_rows():
    bad_policy = np.ones((State.NUM_STATES - 1, Action.NUM_ACTIONS_TOTAL), dtype=float)
    with pytest.raises(AssertionError):
        MDP(init_state_idx=0, policy_array=bad_policy)


def test_init_rejects_invalid_policy_shape_cols():
    bad_policy = np.ones((State.NUM_STATES, Action.NUM_ACTIONS_TOTAL - 1), dtype=float)
    with pytest.raises(AssertionError):
        MDP(init_state_idx=0, policy_array=bad_policy)


def test_get_new_state_with_idx_sets_exact_state():
    mdp = MDP(init_state_idx=0, policy_array=None)
    s = mdp.get_new_state(state_idx=124)
    assert isinstance(s, State)
    assert s.get_state_idx() == 124


def test_generate_random_state_uses_randint(monkeypatch):
    mdp = MDP(init_state_idx=0, policy_array=None)

    def fake_randint(low, high=None, size=None, dtype=int):
        # MDP calls np.random.randint(0, State.NUM_STATES)
        assert low == 0
        assert high == State.NUM_STATES
        return 17

    monkeypatch.setattr(np.random, "randint", fake_randint)

    s = mdp.generate_random_state()
    assert s.get_state_idx() == 17


@pytest.mark.parametrize(
    "categs, expected_reward",
    [
        ([0, 0, 0], (-4) + (-3) + (-4)),  # bv=-4 hc=-3 vr=-4
        ([2, 2, 2], (0) + (0) + (0)),
        ([4, 4, 4], (-1) + (-1) + (-4)),
        ([1, 3, 4], (-1) + (0) + (-2)),  # bv=4 -> -1, hc=3 -> 0, vr=1 -> -2
    ],
)
def test_calculate_reward_matches_tables(categs, expected_reward):
    mdp = MDP(init_state_idx=0, policy_array=None)
    mdp.state = State(state_categs=categs)
    assert mdp.calculate_reward() == expected_reward


def test_select_actions_requires_policy():
    mdp = MDP(init_state_idx=0, policy_array=None)
    with pytest.raises(AssertionError):
        mdp.select_actions()


def test_select_actions_uses_policy_row_and_choice(monkeypatch):
    # Policy: always pick action 5 at state 0
    policy = np.zeros((State.NUM_STATES, Action.NUM_ACTIONS_TOTAL), dtype=float)
    policy[0, 5] = 1.0
    mdp = MDP(init_state_idx=0, policy_array=policy)

    def fake_choice(arr, p=None):
        # ensure choice is called with correct action space
        assert np.array_equal(arr, np.arange(Action.NUM_ACTIONS_TOTAL))
        assert p is not None
        assert np.isclose(p.sum(), 1.0)
        # Should be deterministic here
        return 5

    monkeypatch.setattr(np.random, "choice", fake_choice)

    a = mdp.select_actions()
    assert isinstance(a, Action)
    assert int(a.get_action_idx()) == 5


def test_select_actions_normalizes_if_row_not_sum_to_one(monkeypatch):
    # Deliberately not normalized
    policy = np.zeros((State.NUM_STATES, Action.NUM_ACTIONS_TOTAL), dtype=float)
    policy[0, 2] = 2.0
    policy[0, 3] = 2.0  # sum = 4

    mdp = MDP(init_state_idx=0, policy_array=policy)

    def fake_choice(arr, p=None):
        assert np.isclose(np.sum(p), 1.0)          # must be normalized by MDP
        assert np.isclose(p[2], 0.5) and np.isclose(p[3], 0.5)
        return 3

    monkeypatch.setattr(np.random, "choice", fake_choice)

    a = mdp.select_actions()
    assert int(a.get_action_idx()) == 3


def test_transition_updates_state_with_controlled_randomness(monkeypatch):
    """
    Drive the stochastic branches deterministically via a fixed np.random.uniform sequence.
    We verify state updates are clamped to [0,4] and reward is computed from the new state.

    Uniform calls order in transition():
      1) <=0.7 for BV increase block
      2) <=0.2 for extra BV increase (only if 1 true)
      3) <=0.1 for BV decrease when iv==0 (only if iv==0)
      4) <=0.8 for HC decrease block
      5) <=0.1 for HC increase when epinephrine==0 (only if epinephrine==0)
      6) <=0.9 for VR update block
      7) <=0.1 for VR decrease when phenylephrine==0 (only if phenylephrine==0)
    """
    mdp = MDP(init_state_idx=0, policy_array=None)
    mdp.state = State(state_categs=[2, 2, 2])  # vr=2 hc=2 bv=2

    action = DummyAction(iv=1, epinephrine=1, steroid=1, phenylephrine=1)

    # Trigger all three main update blocks; skip all "if X==0 ..." blocks (since all are 1)
    # 1: yes BV block, 2: yes extra BV, 4: yes HC block, 6: yes VR block
    _make_uniform_sequence(monkeypatch, values=[0.0, 0.0, 0.0, 0.0])

    # Expected new state:
    # BV: 2 + 1 (first) => 3; plus 1 (second) => 4 (clamped)
    # HC: 2 - 1 => 1
    # VR: 2 + (1*1) => 3
    r = mdp.transition(action)

    assert mdp.state.blood_volume_state == 4
    assert mdp.state.heart_contraction_state == 1
    assert mdp.state.vascular_resistence_state == 3

    # Reward tables (from calculate_reward):
    # bv=4 -> -1, hc=1 -> -1, vr=3 -> -2  => total -4
    assert r == -4


def test_transition_iv_zero_can_decrease_bv(monkeypatch):
    mdp = MDP(init_state_idx=0, policy_array=None)
    mdp.state = State(state_categs=[2, 2, 1])  # bv=1

    action = DummyAction(iv=0, epinephrine=1, steroid=1, phenylephrine=1)

    # 1) BV increase block: do NOT trigger (uniform > 0.7)
    # 3) iv==0 decrease: trigger (uniform <= 0.1)
    # 4) HC block: skip (uniform > 0.8)
    # 6) VR block: skip (uniform > 0.9)
    _make_uniform_sequence(monkeypatch, values=[0.99, 0.0, 0.99, 0.99])

    r = mdp.transition(action)
    assert mdp.state.blood_volume_state == 0  # 1 - 1 clamped
    # HC and VR unchanged
    assert mdp.state.heart_contraction_state == 2
    assert mdp.state.vascular_resistence_state == 2
    # reward: bv=0 -> -4, hc=2 -> 0, vr=2 -> 0 => -4
    assert r == -4
