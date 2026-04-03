# tests/test_data_generator.py
import numpy as np
import pytest

from BloodPressureSim.DataGenerator import DataGenerator
from BloodPressureSim.State import State
from BloodPressureSim.Action import Action


def _make_deterministic_policy(action_idx: int) -> np.ndarray:
    """
    Build a policy array of shape (State.NUM_STATES, Action.NUM_ACTIONS_TOTAL)
    that deterministically selects `action_idx` for every state.
    """
    policy = np.zeros((State.NUM_STATES, Action.NUM_ACTIONS_TOTAL), dtype=float)
    policy[:, action_idx] = 1.0
    return policy


def test_simulate_requires_policy():
    dg = DataGenerator()
    with pytest.raises(AssertionError):
        dg.simulate(num_iters=1, max_num_steps=1, policy=None)


def test_select_actions_returns_policy_action_when_state_in_policy():
    dg = DataGenerator()
    s = State(state_idx=0)

    a = Action(action_idx=3)
    policy_dict = {s: a}  # relies on State.__hash__/__eq__

    out = dg.select_actions(s, policy_dict)
    assert isinstance(out, Action)
    assert int(out.get_action_idx()) == 3


def test_select_actions_returns_random_action_when_state_not_in_policy(monkeypatch):
    dg = DataGenerator()
    s = State(state_idx=0)

    def fake_randint(n):
        assert n == Action.NUM_ACTIONS_TOTAL
        return 12

    monkeypatch.setattr(np.random, "randint", fake_randint)

    out = dg.select_actions(s, policy={})
    assert isinstance(out, Action)
    assert int(out.get_action_idx()) == 12


def test_simulate_shapes_and_dtypes_basic():
    """
    Checks output shapes/dtypes and basic invariants without depending on stochastic dynamics.
    """
    dg = DataGenerator()
    policy = _make_deterministic_policy(action_idx=0)

    num_iters = 3
    max_steps = 5
    states, actions, rewards = dg.simulate(
        num_iters=num_iters,
        max_num_steps=max_steps,
        policy=policy,
        use_tqdm=False,
    )

    assert states.shape == (num_iters, max_steps + 1, 1)
    assert actions.shape == (num_iters, max_steps, 1)
    assert rewards.shape == (num_iters, max_steps, 1)

    assert np.issubdtype(states.dtype, np.integer)
    assert np.issubdtype(actions.dtype, np.integer)
    assert np.issubdtype(rewards.dtype, np.floating)

    # initial state indices are recorded (not -1)
    assert np.all(states[:, 0, 0] >= 0)

    # each step action should be in range
    assert np.all((actions[:, :, 0] >= 0) & (actions[:, :, 0] < Action.NUM_ACTIONS_TOTAL))

    # all recorded states should be within valid range
    assert np.all((states[:, :, 0] >= 0) & (states[:, :, 0] < State.NUM_STATES))


def test_simulate_deterministic_policy_produces_constant_action_idx():
    """
    With a deterministic policy (prob=1 on one action), the recorded actions should all equal that index.
    """
    dg = DataGenerator()
    chosen = 7
    policy = _make_deterministic_policy(action_idx=chosen)

    states, actions, rewards = dg.simulate(
        num_iters=2,
        max_num_steps=10,
        policy=policy,
        use_tqdm=False,
    )

    assert np.all(actions[:, :, 0] == chosen)


def test_simulate_rewards_are_finite():
    dg = DataGenerator()
    policy = _make_deterministic_policy(action_idx=0)

    _, _, rewards = dg.simulate(
        num_iters=2,
        max_num_steps=20,
        policy=policy,
        use_tqdm=False,
    )

    assert np.all(np.isfinite(rewards))
