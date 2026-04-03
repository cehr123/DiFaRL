import numpy as np
import pickle
from tqdm import tqdm
import scipy.sparse

# Sepsis Simulator code
from BloodPressureSim.State import State
from BloodPressureSim.Action import Action
from BloodPressureSim.DataGenerator import DataGenerator
import BloodPressureSim.MDP as simulator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import pathlib
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir_data', type=str, choices=['eps_1', 'eps_0_5', 'eps_0_2', 'eps_0_1'], default = 'eps_1')
parser.add_argument('--NSIMSAMPS', type=int, default=1000)
parser.add_argument('--runs', type=int, default=10)
args = parser.parse_args()

epsilon_options = {'eps_1':1, 'eps_0_5':0.5, 'eps_0_2':0.2, 'eps_0_1':0.1}

# --- Configuration ---
NSIMSAMPS = args.NSIMSAMPS
epsilon = epsilon_options[args.dir_data]
runs = list(range(args.runs))

output_dir = f'../../../BloodPressureSim/datagen/{args.dir_data}/'
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

# Max length of each trajectory
NSTEPS = 20

# Number of actions from properties of the simulator
n_actions = Action.NUM_ACTIONS_TOTAL
n_states = State.NUM_STATES
nAs = np.array([2,2,2,2])

def convert_factored_action(a, nAj_all):
    subactions = []
    for j in range(len(nAj_all)):
        _A_j = nAj_all[j]
        a_j = a % _A_j
        subactions.append(a_j)
        a = a // _A_j
    return subactions

# Load the optimal policy
optPol = joblib.load('../../../BloodPressureSim/data/π_star.joblib')
behaviorPol =  np.zeros_like(optPol)
for l, row in enumerate(optPol):
    act = np.argmax(row)
    act_fac = convert_factored_action(act, nAs)
    fac_row = []
    for Ai in act_fac:
        polj = np.zeros((2))
        polj[Ai] = 1
        randPolj = np.ones((2))/2
        behaviorPolj =  epsilon * randPolj + (1 - epsilon) * polj
        fac_row.append(behaviorPolj)
    for action in range(n_actions):
        prob = 1
        action_fac = convert_factored_action(action, nAs)
        for j, aj in enumerate(action_fac):
            prob *= fac_row[j][aj]
        behaviorPol[l][action] = prob

def get_one_hot(state):
    s_oh = np.zeros((n_states))
    s_oh[state] = 1
    return s_oh


# --- Data Generation ---
# Generate batches of data with N episodes each
for it in runs:
    print('Iteration:', it, flush=True)
    np.random.seed(it)
    dgen = DataGenerator()
    states, actions, rewards = dgen.simulate(NSIMSAMPS, NSTEPS, policy=behaviorPol, use_tqdm=True)

    X = []
    X_next = []
    A = []
    R = []
    transitions = []

    for i in tqdm(range(NSIMSAMPS), desc="Generating transitions"):
        episode_length = len(actions[i])
        for t in range(episode_length - 1):
            obs = get_one_hot(states[i][t])
            next_obs = get_one_hot(states[i][t + 1])
            a = int(actions[i][t].item())
            a_next = int(actions[i][t + 1].item()) if t + 1 < episode_length - 1 else None
            r = float(rewards[i][t].item())
            done = (t == episode_length - 2)
            info = {'episode': i, 't': t}

            transitions.append((obs, next_obs, a, a_next, r, done, info))
            X.append(obs)
            X_next.append(next_obs)
            A.append(a)
            R.append(r)

    batch_path = os.path.join(output_dir, f"transitions_batch_{it}.pkl")
    with open(batch_path, "wb") as f:
        pickle.dump(transitions, f)

    X = np.array(X)
    X_next = np.array(X_next)
    A = np.array(A)
    R = np.array(R)
    joblib.dump({
        'X': scipy.sparse.csr_matrix(X.astype(int)), 'A': A, 'R': R, 'X_next': scipy.sparse.csr_matrix(X_next.astype(int))
    }, f'{output_dir}/{it}-feature-matrices.sparse.joblib')
