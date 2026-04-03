import numpy as np

def convert_to_policy_table(Q, nS, nA):
    """
    Map the predicted Q-values (S,A) to a greedy policy (S,A) wrt tabular MDP
    Handles the last two absorbing states
    """
    a_star = Q.argmax(axis=1)
    pol = np.zeros((nS, nA))
    pol[list(np.arange(nS)), a_star] = 1
    return pol


def policy_eval_analytic(P, R, π, γ):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, _ = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = np.linalg.inv(np.eye(nS) - γ * P_π) @ R_π
    return V_π