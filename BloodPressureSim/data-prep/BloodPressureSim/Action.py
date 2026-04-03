# python3 -m pytest -q

import numpy as np

class Action(object):

    NUM_ACTIONS_TOTAL = 16  # for 4 actions (2^4)
    ACTION_VEC_SIZE = 4

    def __init__(self, action_idx=None):
        assert action_idx is not None, "must specify action index"

        mod_idx = action_idx

        # Calculate iv status
        term_base_iv = Action.NUM_ACTIONS_TOTAL / 2
        self.iv = np.floor(mod_idx / term_base_iv).astype(int)
        mod_idx %= term_base_iv

        # Calculate steroid status
        term_base_steroid = term_base_iv / 2
        self.steroid = np.floor(mod_idx / term_base_steroid).astype(int)
        mod_idx %= term_base_steroid

        # Calculate epinephrine status
        term_base_epinephrine = term_base_steroid / 2
        self.epinephrine = np.floor(mod_idx / term_base_epinephrine).astype(int)
        mod_idx %= term_base_epinephrine
        
        # Calculate phenylephrine status
        term_base_phenylephrine = term_base_epinephrine / 2
        self.phenylephrine = np.floor(mod_idx / term_base_phenylephrine).astype(int)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.iv == other.iv and \
            self.steroid == other.steroid and \
            self.epinephrine == other.epinephrine and \
            self.phenylephrine == other.phenylephrine

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_action_idx(self):
        assert self.iv in (0, 1)
        assert self.steroid in (0, 1)
        assert self.epinephrine in (0, 1)
        assert self.phenylephrine in (0, 1)
        return 8 * self.iv + 4 * self.steroid + 2 * self.epinephrine + self.phenylephrine

    def __hash__(self):
        return int(self.get_action_idx())

    def get_action_vec(self):
        return np.array([[self.iv], [self.steroid], [self.epinephrine], [self.phenylephrine]])