import numpy as np

class State(object):
    NUM_STATES = 5**3

    def __init__(self, state_idx=None, state_categs=None):
        assert state_idx is not None or state_categs is not None, \
            "Must specify either state_idx or state_categs"

        if state_idx is not None:
            self.set_state_by_idx(state_idx)

        elif state_categs is not None:
            self.vascular_resistence_state = state_categs[0]
            self.heart_contraction_state = state_categs[1]
            self.blood_volume_state = state_categs[2]

    def set_state_by_idx(self, state_idx):
        """
        Sets the state by a given integer index.
        The index is decoded to determine the categorical values for each state variable.
        """
        mod_idx = int(state_idx)

        self.blood_volume_state = mod_idx % 5
        mod_idx //= 5

        self.heart_contraction_state = mod_idx % 5
        mod_idx //= 5

        self.vascular_resistence_state = mod_idx % 5


    def get_state_idx(self):
        '''
        Returns the integer index of the current state.
        The order of significance for encoding is: VascularResistence, HeartContractions, BloodVolume.
        '''
        sum_idx = (self.vascular_resistence_state * (5 * 5)) + \
                  (self.heart_contraction_state * (5)) + \
                  self.blood_volume_state
        return int(sum_idx)

    def __eq__(self, other):
        '''
        Overrides equals: two states are equal if all their internal categorical states are the same.
        '''
        return isinstance(other, self.__class__) and \
            self.vascular_resistence_state == other.vascular_resistence_state and \
            self.heart_contraction_state == other.heart_contraction_state and \
            self.blood_volume_state == other.blood_volume_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.get_state_idx()

    def copy_state(self):
        """
        Creates and returns a new State object with the same categorical values as the current state.
        """
        return State(state_categs = [
            self.vascular_resistence_state,
            self.heart_contraction_state,
            self.blood_volume_state])

    def get_state_vector(self):
        """
        Returns a NumPy array representing the categorical values of the current state.
        """
        return np.array([self.vascular_resistence_state,
            self.heart_contraction_state,
            self.blood_volume_state]).astype(int)