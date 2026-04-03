import numpy as np
import argparse
import random as python_random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import copy
from cd_utils import get_mask, get_groups
from utils import load_data, convert_factored_action
from utils import myDataset, EarlyStopping, save_hparams
import os
from utils import  update_target_network, train, val_loss_fn, get_target
from network_architectures import AttentionNetwork, DenseNetwork

NSTEPS = 20
nS, nA = 125, 16
d = 125
nAs = np.array([2,2,2,2])
dA = len(nAs)
assert nA == np.prod(nAs)
gamma = 0.99
X_ALL_states = np.eye(125, dtype=int)


def main(args):
    # make the output directory
    save_dir = f'../../../BloodPressureSim/output/N={args.N},run{args.run}/{args.dir}/{args.model}-{args.group}/'
    os.makedirs(save_dir, exist_ok=True)

    # save the hyperparameters
    yaml_path = os.path.join(save_dir, "hyperparameters.yaml")
    save_hparams(args, yaml_path)

    num_epoch = args.max_iterations

    X, A, R, X_next = load_data(f'../../../BloodPressureSim/datagen/{args.dir}', args.run, args.N, NSTEPS-1)
    A_fac = np.array(convert_factored_action(A, nAs)).T

    val_size = int(X.shape[0]*0.1)
    train_size = X.shape[0]-val_size

    data = myDataset(X, A_fac, np.zeros_like(R))
    train_data, val_data = random_split(data,[train_size,val_size])
    train_dl = DataLoader(train_data,batch_size=64,shuffle=True)
    val_dl = DataLoader(val_data,batch_size=val_size)
    early_stopping = EarlyStopping(patience=30,min_delta=1e-5,restore_best_weights=True)
           
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)
    python_random.seed(0)
    torch.manual_seed(0)

    print("\033[1m\033[31mFQI iteration 0\033[0m")
            
    if args.model == 'attention':
        mask = get_mask(args.group, X, A_fac, X_next, dA)
        with open(f'{save_dir}/attention_mask.txt', 'w') as file:
            file.write(str(mask))
        mask = torch.from_numpy(mask).bool().to(device)
        model = AttentionNetwork(mask, state_size = d, action_dims=nAs, d_model = args.d_model, num_heads = args.num_heads, num_layers = args.num_layers, hidden_dim = args.hidden_dim).to(device)
    elif args.model == 'dense':
        groups = get_groups(args.group, X, A_fac, X_next, dA)
        with open(f'{save_dir}/groups.txt', 'w') as file:
            file.write(str(groups))
        model = DenseNetwork(groups, state_size = d, action_dims=nAs, num_layers = args.num_layers, hidden_dim = args.hidden_dim).to(device)

    target_model = copy.deepcopy(model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, eps=1e-7, weight_decay=args.weight_decay)

    train_data = myDataset(X, A_fac, np.zeros_like(R))
    train_dl = DataLoader(train_data, batch_size=64)
    early_stopping = EarlyStopping(patience=30,min_delta=1e-5,restore_best_weights=True)

    # Pretrain initial model
    for epoch in range(100):
        train(train_dl, model, criterion, optimizer, device)
        with torch.no_grad():
            vloss = val_loss_fn(model, val_dl, criterion,device)
            early_stopping(vloss,model)
            if early_stopping.early_stop:
                print("early stop epoch:", epoch)
                break

    target_model.load_state_dict(model.state_dict())
    for k in range(num_epoch):
        print(f"\033[1m\033[31mFQI iteration {k}\033[0m")

        y = get_target(model, target_model, X_next, R, nA, nAs, device)

        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, eps=1e-7, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=30, min_delta=1e-5, restore_best_weights=True)

        data = myDataset(X, A_fac, y)
        train_data, val_data = random_split(data,[train_size,val_size])
        train_dl = DataLoader(train_data,batch_size=64,shuffle=True)
        val_dl = DataLoader(val_data,batch_size=val_size)

        for epoch in range(100):
            train(train_dl, model, criterion, optimizer, device)
            with torch.no_grad():
                vloss = val_loss_fn(model, val_dl, criterion, device)
                early_stopping(vloss, model)
                if early_stopping.early_stop:
                    print("early stop epoch:", epoch); break

        # Polyak averaging
        if (k) % args.target_update_freq == 0:
            update_target_network(target_model, model, tau=args.tau)

        torch.save(model.state_dict(), f'{save_dir}/iter={k}.net')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['attention', 'dense'], default='attention')
    temp = parser.parse_known_args()[0]

    if temp.model == 'attention':
        parser.add_argument('--d_model', type=int, default=32)
        parser.add_argument('--num_heads', type=int, default=4)  

    parser.add_argument('--group', type=str, choices=['baseline', 'factored', 'oracle', 'DiFaRL'], default='oracle')
    parser.add_argument('--dir', type=str, default="eps_0_5")
    parser.add_argument('--max_iterations', type=int, default=30)
    parser.add_argument('--tau', type=float, default=0.2)
    parser.add_argument('--target_update_freq', type=int, default=1)

    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args()
    main(args)