import numpy as np
from BloodPressureSim.State import State
from BloodPressureSim.Action import Action
from BloodPressureSim.MDP import MDP
import joblib
import os 

if not os.path.exists('../../../BloodPressureSim/data'):
    os.makedirs('../../../BloodPressureSim/data')

nS = State.NUM_STATES
nA = Action.NUM_ACTIONS_TOTAL 


# Reward Matrix (S)
reward_per_state = np.zeros((nS))
for s_idx in range(nS):
    temp_state = State(state_idx=s_idx)
    dummy_mdp = MDP(init_state_idx=s_idx)
    r = dummy_mdp.calculate_reward()
    reward_per_state[s_idx] = r

# Reward matrix (A, S, S')
# The reward is R(S') where S' is the state transitioned to.
reward_matrix_ASS = np.zeros((nA, nS, nS))
for s_prime in range(nS):
    reward_matrix_ASS[:, :, s_prime] = reward_per_state[s_prime]


## Transition Matrix (A, S, S')
transition_matrix = np.zeros((nA, nS, nS))

num_samps = 10000

# Loop through all possible actions (a)
for a_idx in range(nA):
    action = Action(action_idx=a_idx)
    for s_idx in range(nS):
        vec = np.zeros((nS))
        for _ in range(num_samps):
            dummy_mdp = MDP(init_state_idx=s_idx)
            dummy_mdp.transition(action)
            vec[dummy_mdp.state.get_state_idx()] = vec[dummy_mdp.state.get_state_idx()] + 1
        vec = vec / num_samps
        transition_matrix[a_idx, s_idx] = vec

# Assert all rows sum to 1
for a in range(nA):
    for s_row in range(nS):
        row_sum = transition_matrix[a, s_row, :].sum()
        assert np.isclose(row_sum, 1.0), \
            f"Transition matrix row {s_row} for action {a} does not sum to 1.0. Sum: {row_sum}"

print("Transition matrix shape:", transition_matrix.shape)

## Initial State Distribution (S,)
prior_initial_state = np.ones(nS)/nS

print("Prior initial state shape:", prior_initial_state.shape)
joblib.dump(prior_initial_state, '../../../BloodPressureSim/data/prior_initial_state.joblib')

# Reward matrix (S, A)
# The reward is R(S') where S' is the state transitioned to.
reward_matrix_SA = np.zeros((nS, nA))

# Calculate R(s, a)
for s_idx in range(nS):
    for a_idx in range(nA):
        # The expected reward for taking action a_idx from state s_idx
        # is the sum of (probability of transitioning to s_prime * reward of s_prime)
        expected_reward = np.sum(transition_matrix[a_idx, s_idx, :] * reward_per_state)
        reward_matrix_SA[s_idx, a_idx] = expected_reward

## Save MDP Parameters
MDP_parameters = {
    'transition_matrix': transition_matrix,
    'reward_per_state': reward_per_state,
    'reward_matrix_ASS': reward_matrix_ASS,
    'prior_initial_state': prior_initial_state,
    'reward_matrix_SA': reward_matrix_SA
}

joblib.dump(MDP_parameters, '../../../BloodPressureSim/data/MDP_parameters.joblib')

print("\nSuccessfully updated and saved MDP parameters!")