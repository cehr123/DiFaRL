import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import pickle
import copy
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
import mdptoolbox.mdp as mdptools

# Ground truth MDP model
MDP_parameters = joblib.load('../../../BloodPressureSim/data/MDP_parameters.joblib')
P = MDP_parameters['transition_matrix'] # (A, S, S_next)
R = MDP_parameters['reward_matrix_SA'] # (S, A)
nS, nA = R.shape
gamma = 0.99

isd = joblib.load('../../../BloodPressureSim/data/prior_initial_state.joblib')

# Map back to the policy in discrete MDP
def convert_to_policy_table(pi):
    pol = np.zeros((nS, nA))
    pol[list(np.arange(nS)), pi] = 1
    return pol

def policy_eval_analytic(P, R, pi, gamma):
    """
    Given the MDP model (transition probability P (S,A,S) and reward function R (S,A)),
    Compute the value function of a policy using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_pi = np.sum(R * pi, axis=1)
    P_pi = np.sum(P * np.expand_dims(pi, 2), axis=1)
    V_pi = np.linalg.inv(np.eye(nS) - gamma * P_pi) @ R_pi
    return V_pi
    

# Policy Iteration
PI = mdptools.PolicyIteration(P, R, discount=gamma)
PI.run()
V_star_PI = np.array(PI.V)
π_star_PI = np.array(PI.policy)

# Value Iteration
VI = mdptools.ValueIteration(P, R, discount=gamma, epsilon=1e-10)
VI.run()
V_star_VI = np.array(VI.V)
π_star_VI = np.array(VI.policy)

# re-evalute the learned policy
pi_star_VI = convert_to_policy_table(π_star_VI)
V_π_star_PE = policy_eval_analytic(P.transpose((1,0,2)), R, pi_star_VI, gamma)

print('Max policy value', isd @ V_star_PI)
print('Value and policy iteration are equal', np.all(π_star_VI == π_star_PI))

joblib.dump(π_star_PI, '../../../BloodPressureSim/data/π_star_PI.joblib')
joblib.dump(V_star_PI, '../../../BloodPressureSim/data/V_star_PI.joblib')

joblib.dump(π_star_VI, '../../../BloodPressureSim/data/π_star_VI.joblib')
joblib.dump(V_star_VI, '../../../BloodPressureSim/data/V_star_VI.joblib')
joblib.dump(V_π_star_PE, '../../../BloodPressureSim/data/V_π_star_PE.joblib')

joblib.dump(pi_star_VI, '../../../BloodPressureSim/data/π_star.joblib')



