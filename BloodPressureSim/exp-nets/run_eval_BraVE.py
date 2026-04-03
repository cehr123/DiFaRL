import argparse
import torch
from torch import nn
from OPE_utils import *
import numpy as np
from tqdm import tqdm
import joblib
import itertools
import json
from run_BraVE import BraVE, Sepsis
from utils import convert_factored_action

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default = 'eps_0_5')
parser.add_argument("--Ns", nargs="+", type=int, default = [50, 100])
parser.add_argument("--runs", type=int, default = 1)
args = parser.parse_args()

import os
os.makedirs('./results/{}/'.format(args.dir), exist_ok=True)

def compute_children(env, idx):
    action_shape = (2,) * 4
    current_action = env.compute_action_from_index(idx)
    last_activated_sub_action = np.max(np.where(current_action == 1)[0], initial=-1)

    children = []
    for i in range(last_activated_sub_action + 1, len(current_action)):
        child_action = np.copy(current_action)
        child_action[i] = 1
        child_index = np.ravel_multi_index(tuple(child_action), action_shape)
        children.append(child_index)

    return children


def compute_action(env, network, obs, device, sa, sab, k):
    obs = torch.Tensor(obs).to(device)
    beams = [0]
    beam_values = [-float('inf')]
    beams_to_explore = [0]
    explored_beams = set()

    while beams_to_explore:
        action = beams_to_explore.pop(0)
        if action in explored_beams:
            continue
        explored_beams.add(action)

        children = compute_children(env, action)
        children = [c for c in children if c in sab]

        with torch.no_grad():
            action_tensor = torch.tensor(env.compute_action_from_index(action), device=device).view(1, -1)
            values = network(obs.unsqueeze(0), action_tensor).flatten()[:len(children) + 1]

        # Use Q value instead of BVE
        action_idx = beams.index(action)
        beam_values[action_idx] = values[0].item()

        top_action_values, top_action_indices = torch.topk(values, min(k, len(children) + 1))

        if 0 in top_action_indices:
            if action not in sa:
                masked_values = values.clone()
                masked_values[0] = float('-inf')
                top_action_values, top_action_indices = torch.topk(masked_values, min(k, len(children) + 1))

        for i, action_value in enumerate(top_action_values):
            new_action = children[top_action_indices[i] - 1] if top_action_indices[i] > 0 else action
            if new_action not in explored_beams and new_action not in beams_to_explore:
                if len(beams) == k:
                    if action_value.item() >= min(beam_values):
                        min_action_value_idx = beam_values.index(min(beam_values))
                        action_to_remove = beams[min_action_value_idx]

                        if action_to_remove not in explored_beams:
                            beams_to_explore_idx = beams_to_explore.index(action_to_remove)
                            beams_to_explore.pop(beams_to_explore_idx)

                        beams[min_action_value_idx] = new_action
                        beam_values[min_action_value_idx] = action_value.item()
                        beams_to_explore.append(new_action)

                else:
                    beams.append(new_action)
                    beam_values.append(action_value.item())
                    beams_to_explore.append(new_action)

    max_index = beam_values.index(max(beam_values))
    return np.array([beams[max_index]])


def policy_eval_analytic(P, R, π, γ):
    """
    Given the MDP model transition probability P (S,A,S) and reward function R (S,A),
    Compute the value function of a stochastic policy π (S,A) using matrix inversion
    
        V_π = (I - γ P_π)^-1 R_π
    """
    nS, nA = R.shape
    R_π = np.sum(R * π, axis=1)
    P_π = np.sum(P * np.expand_dims(π, 2), axis=1)
    V_π = np.linalg.inv(np.eye(nS) - γ * P_π) @ R_π
    return V_π


def evaluate(env, network, device, seen_actions, seen_action_branches, num_beams):
    X_ALL_states = np.eye(125, dtype=int)

    import os
    this_dir = os.path.dirname(__file__)
    MDP_parameters = joblib.load(os.path.join(this_dir, '../../../BloodPressureSim/data', 'MDP_parameters.joblib'))
    isd = joblib.load(os.path.join(this_dir, '../../../BloodPressureSim/data', 'prior_initial_state.joblib'))
    P_ = MDP_parameters['transition_matrix']
    R_ = MDP_parameters['reward_matrix_SA'] 
    gamma = 0.99

    π_pred = np.zeros((125, 16))

    for i, state in enumerate(X_ALL_states):
        actions = compute_action(env, network, state, device, seen_actions, seen_action_branches, num_beams)
        π_pred[i][actions[0]] = 1

    true_value = isd @ policy_eval_analytic(P_.transpose((1,0,2)), R_, π_pred, gamma)

    return true_value

print('Load FQI models')
runs = list(range(args.runs))
Ns = args.Ns
k_list = list(range(19))

keys_list = list(itertools.product(Ns, runs, k_list))

print('Ground-truth performance')
true_value_dict = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 129
env = Sepsis(nA = 4)
num_layers = 2

for N, run, k in tqdm(keys_list):
    save_dir = f'../../../BloodPressureSim/output/N={N},run{run}/{args.dir}/BraVE/'

    with open(f"{save_dir}seen_actions", "r") as f:
        data = json.load(f)

    seen_actions = set(data["seen_actions"])
    seen_action_branches = set(data["seen_action_branches"])

    network = BraVE(input_size, env, num_layers).to(device)
    network.load_state_dict(torch.load(f'{save_dir}iter={k}.net', map_location=torch.device('cpu')))

    true_value = evaluate(env, network, device, seen_actions, seen_action_branches, 5)
    
   
    print(N, run, k, true_value)
    true_value_dict[N, run, k] = true_value

    joblib.dump(true_value_dict, f'../../../BloodPressureSim/results/{args.dir}/eval_BraVE.joblib')

joblib.dump(true_value_dict, f'../../../BloodPressureSim/results/{args.dir}/eval_BraVE.joblib')