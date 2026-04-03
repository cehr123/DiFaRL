import numpy as np
import joblib 
from .State import State
from .Action import Action

class MDP(object):

    def __init__(self, init_state_idx=None, policy_array=None):
        '''
        Initialize the simulator.

        :param init_state_idx: Optional initial state index. If None, a random state is generated.
        :param policy_array: Optional array representing the policy (state x actions).
        '''
        # Policy array dimensions: (NUM_STATES x NUM_ACTIONS_TOTAL)
        if policy_array is not None:
            assert policy_array.shape[0] == State.NUM_STATES, \
                f"Policy array's first dimension ({policy_array.shape[0]}) must match State.NUM_STATES ({State.NUM_STATES})"
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL, \
                f"Policy array's second dimension ({policy_array.shape[1]}) must match Action.NUM_ACTIONS_TOTAL ({Action.NUM_ACTIONS_TOTAL})"

        self.state = self.get_new_state(init_state_idx)
        self.policy_array = policy_array

    def get_new_state(self, state_idx=None):
        '''
        Use to start MDP over.

        :param state_idx: Optional initial state index. If None, a random state is generated.
        :return: A new State object.
        '''
        if state_idx is not None:
            init_state = State(state_idx=state_idx)
        else:
            init_state = self.generate_random_state()
        return init_state

    def generate_random_state(self):
        '''
        Generates a random initial state for the MDP.
        '''
        return State(state_idx=np.random.randint(0, State.NUM_STATES))
    
    def calculate_reward(self):
        '''
        Calculates the reward based on the current state's vital signs.
        Rewards are given for normal ranges, and penalties for abnormal ranges,
        with varying severity for minor and major abnormalities.
        '''
        bv_reward = [-4, -2, 0, 0, -1]
        hc_reward = [-3, -1, 0, 0, -1]
        vr_reward = [-4, -2, 0, -2, -4]
        return bv_reward[self.state.blood_volume_state] + hc_reward[self.state.heart_contraction_state] + vr_reward[self.state.vascular_resistence_state]

    def transition(self, action):
        '''
        Applies the chosen action and simulates the transition to the next state,
        then calculates the reward.
        '''
        # --- Apply treatment effects and fluxuations ---
        if np.random.uniform() <= 0.7:
            self.state.blood_volume_state = int(max(min(self.state.blood_volume_state + action.iv, 4), 0))

            if np.random.uniform() <= 0.2:
                self.state.blood_volume_state = int(max(min(self.state.blood_volume_state + action.iv, 4), 0))
                
        if action.iv == 0 and np.random.uniform() <= 0.1:
            self.state.blood_volume_state = int(max(min(self.state.blood_volume_state -1, 4), 0))

        if np.random.uniform() <= 0.8:
            self.state.heart_contraction_state = int(max(min(self.state.heart_contraction_state - action.epinephrine, 4), 0))
        if action.epinephrine == 0 and np.random.uniform() <= 0.1:
            self.state.heart_contraction_state = int(max(min(self.state.heart_contraction_state + 1, 4), 0))

        if np.random.uniform() <= 0.9:
            self.state.vascular_resistence_state = int(max(min(self.state.vascular_resistence_state + action.steroid * action.phenylephrine, 4), 0))
        if action.phenylephrine == 0 and np.random.uniform() <= 0.1:
            self.state.vascular_resistence_state = int(max(min(self.state.vascular_resistence_state -1, 4), 0))

        return self.calculate_reward()

    def select_actions(self):
        '''
        Selects an action based on the current policy and state.
        '''
        assert self.policy_array is not None, "Policy array must be set to select actions."
        
        # Get the probability distribution over actions for the current state
        probs = self.policy_array[self.state.get_state_idx()]
        
        # Ensure probabilities sum to 1 (or handle potential floating point errors)
        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs)

        # Sample an action index based on the probabilities
        action_idx = np.random.choice(np.arange(Action.NUM_ACTIONS_TOTAL), p=probs)
        
        return Action(action_idx=action_idx)