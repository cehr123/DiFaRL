import numpy as np
import random
from .MDP import MDP
from .State import State
from .Action import Action
from tqdm import tqdm # Using tqdm for progress bars

'''
Simulates data generation from an MDP
'''
class DataGenerator(object):

    def select_actions(self, state, policy):
        '''
        Select action for state from policy.
        If the state is not explicitly in the policy, a random action is returned.
        '''
        if state not in policy:
            # Assuming Action.NUM_ACTIONS_TOTAL is 16 (for 4 binary actions)
            return Action(action_idx=np.random.randint(Action.NUM_ACTIONS_TOTAL))
        return policy[state]

    def simulate(self, num_iters, max_num_steps,
                 policy=None, use_tqdm=False, tqdm_desc=''):
        '''
        Simulates data generation from the MDP.

        :param num_iters: Number of simulation iterations (trajectories).
        :param max_num_steps: Maximum number of steps in each simulation iteration.
        :param policy: The policy array (State.NUM_STATES x Action.NUM_ACTIONS_TOTAL)
                       to use for action selection.
        :param use_tqdm: Boolean to enable/disable the tqdm progress bar.
        :param tqdm_desc: Description for the tqdm progress bar.
        :return: Tuple of arrays: iter_states, iter_actions, iter_rewards, iter_lengths.
                 - iter_states: (num_iters, max_num_steps+1, 1) array of state indices.
                 - iter_actions: (num_iters, max_num_steps, 1) array of action indices.
                 - iter_rewards: (num_iters, max_num_steps, 1) array of rewards.
                 - iter_lengths: (num_iters, 1) array of trajectory lengths.
        '''
        assert policy is not None, "Please specify a policy array for the simulation."

        # Initialize arrays to store simulation data
        iter_states = np.ones((num_iters, max_num_steps + 1, 1), dtype=int) * (-1)
        iter_actions = np.ones((num_iters, max_num_steps, 1), dtype=int) * (-1)
        iter_rewards = np.zeros((num_iters, max_num_steps, 1))
        iter_lengths = np.zeros((num_iters, 1), dtype=int)

        # Initialize the MDP with the provided policy
        # The MDP now directly uses the policy array
        mdp = MDP(init_state_idx=None, policy_array=policy)

        for itr in tqdm(range(num_iters), disable=not(use_tqdm), desc=tqdm_desc, leave=False):
            # Reset MDP state for each iteration (trajectory)
            mdp.state = mdp.get_new_state()

            # Record the initial state of the trajectory
            iter_states[itr, 0, 0] = mdp.state.get_state_idx()

            for step in range(max_num_steps):
                # Select an action based on the MDP's internal policy selection logic
                step_action = mdp.select_actions()

                this_action_idx = step_action.get_action_idx().astype(int)
                
                # Take the action, new state is a property of the MDP
                step_reward = mdp.transition(step_action)
                
                # The 'to_state_idx' is the state *after* the transition
                this_to_state_idx = mdp.state.get_state_idx()

                # Record the action and the resulting state
                iter_actions[itr, step, 0] = this_action_idx
                iter_states[itr, step + 1, 0] = this_to_state_idx
                iter_rewards[itr, step, 0] = step_reward # Store the reward for this step

        return iter_states, iter_actions, iter_rewards