import argparse
import torch
import os
import numpy as np
import joblib
import os
import itertools
import yaml
from tqdm import tqdm
from pathlib import Path
from OPE_utils import convert_to_policy_table, policy_eval_analytic
from utils import convert_factored_action
from network_architectures import AttentionNetwork, DenseNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default = 'eps_0_5')
parser.add_argument('--model', type=str, choices=['attention', 'dense'], default='attention')
parser.add_argument('--group', type=str, choices=['baseline', 'factored', 'oracle', 'DiFaRL'], default='oracle')
parser.add_argument("--Ns", nargs="+", type=int, default = [50, 100])
parser.add_argument("--runs", type=int, default = 1)
parser.add_argument("--k", type=int, default = 30)
args = parser.parse_args()

os.makedirs('../../../BloodPressureSim/results/{}/'.format(args.dir), exist_ok=True)

nAs = np.array([2,2,2,2])
MDP_parameters = joblib.load('../../../BloodPressureSim/data/MDP_parameters.joblib')
P_ = MDP_parameters['transition_matrix'] # (A, S, S_next)
R_ = MDP_parameters['reward_matrix_SA'] # (S, A)
nS, nA = R_.shape
d = nS
γ = gamma = 0.99
isd = joblib.load('../../../BloodPressureSim/data/prior_initial_state.joblib')
X_ALL_states = np.eye(125, dtype=int)

runs = list(range(args.runs))
k_list = list(range(args.k))
Ns = args.Ns
keys_list = list(itertools.product(Ns, runs, k_list))

true_value_dict = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for N, run, k in tqdm(keys_list):
    save_dir = f'../../../BloodPressureSim/output/N={N},run{run}/{args.dir}/{args.model}-{args.group}'

    with open(f"{save_dir}/hyperparameters.yaml") as f:
            hyperparameters = yaml.safe_load(f)

    file_path = Path(f"{save_dir}/iter={k}.net")
    if file_path.exists():
        model_weights = torch.load(
            f"{save_dir}/iter={k}.net",
            map_location="cpu"
        )
    else:
        raise FileNotFoundError(f"The checkpoint file for N = {N}, run = {run}, k = {k} was not found.")

    if args.model == "attention":
        mask = open(f"{save_dir}/attention_mask.txt").read()
        mask = mask.replace("[", "").replace("]", "")
        mask = np.fromstring(mask, sep=' ', dtype=float)
        mask = mask.reshape(4, 4)
        mask = mask.astype(int)
        mask = torch.tensor(mask).bool()
        model = AttentionNetwork(
            mask,
            state_size=d,
            action_dims=nAs,
            d_model=hyperparameters['d_model'],
            num_heads=hyperparameters['num_heads'],
            num_layers=hyperparameters['num_layers'],
            hidden_dim=hyperparameters['hidden_dim']
        )
    
    elif args.model == "dense":
        groups = eval(open(f"{save_dir}/groups.txt").read())
        model = DenseNetwork(
            groups,
            state_size=d,
            action_dims=nAs,
            num_layers=hyperparameters['num_layers'],
            hidden_dim=hyperparameters['hidden_dim']
        )

    model.load_state_dict(model_weights)
    model.eval()

    Q_eval = []
    for A_id in range(nA):
        A_batch = np.full(len(X_ALL_states), A_id)
        A_batch = np.array(convert_factored_action(A_batch, nAs)).T
        q = model(torch.tensor(X_ALL_states,dtype=torch.float32,device=device),
                torch.tensor(A_batch,dtype=torch.int64,device=device))
        Q_eval.append(q.detach().cpu().numpy())
    Q_pred = np.array(Q_eval).T
    π_pred = convert_to_policy_table(Q_pred, nS, nA)
    true_value = isd @ policy_eval_analytic(P_.transpose((1,0,2)), R_, π_pred, gamma)
    true_value_dict[N, run, k] = true_value
    print(true_value)
    joblib.dump(true_value_dict, f'../../../BloodPressureSim/results/{args.dir}/eval_{args.model}_{args.group}.joblib')

joblib.dump(true_value_dict, f'../../../BloodPressureSim/results/{args.dir}/eval_{args.model}_{args.group}.joblib')

