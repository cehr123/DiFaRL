import numpy as np
import joblib
import torch
from torch.utils.data import Dataset
import copy
import os
import pickle
from torch import nn
import yaml

''' data loading utilities '''

def load_sparse_features(input_dir, fname):
    feat_dict = joblib.load(f'{input_dir}/{fname}')
    return  feat_dict['X'].toarray(), feat_dict['A'], feat_dict['X_next'].toarray(), feat_dict['R']

def load_data(input_dir, run, N, NSTEPS):
    X, A, X_next, R = load_sparse_features(input_dir, f'{run}-feature-matrices.sparse.joblib')
    return X[:N*NSTEPS], A[:N*NSTEPS],  R[:N*NSTEPS],  X_next[:N*NSTEPS]

def convert_factored_action(a, nAj_all):
    subactions = []
    for j in range(len(nAj_all)):
        _A_j = nAj_all[j]
        a_j = a % _A_j
        subactions.append(a_j)
        a = a // _A_j
    return subactions

def load_transitions(path, run, N, NSTEPS):
    path = f'../../../BloodPressureSim/datagen/{path}'
    transitions = pickle.load(open(os.path.join(path, f'transitions_batch_{run}.pkl'), 'rb'))
    return transitions[:N * (NSTEPS - 1)]


''' torch utilities '''

class myDataset(Dataset):
    def __init__(self, X, A, y): 
        self.X, self.A, self.y = X, A, y

    def __len__(self): 
        return len(self.X)
        
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.A[idx], dtype=torch.int64),
                torch.tensor(self.y[idx], dtype=torch.float32))

class EarlyStopping:
    def __init__(self, patience, min_delta, restore_best_weights):
        self.patience, self.min_delta, self.restore_best_weights = patience, min_delta, restore_best_weights
        self.best_loss, self.best_model_state, self.counter, self.early_stop = float('inf'), None, 0, False
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_model_state, self.counter = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_state:
                    model.load_state_dict(self.best_model_state)


def build_mlp(in_features, out_features, num_layers=2, hidden_dim=256):
    layers, dim = [], in_features
    if num_layers == 1: layers.append(nn.Linear(dim,out_features))
    else:
        layers += [nn.Linear(dim,hidden_dim), nn.ReLU()]
        for _ in range(num_layers-2):
            layers += [nn.Linear(hidden_dim,hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim,out_features))
    return nn.Sequential(*layers)

def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss=0
    for X, A, y in dataloader:
        X, A, y = X.to(device), A.to(device), y.to(device)
        pred = model(X,A)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(dataloader.dataset)

def val_loss_fn(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, A, y in dataloader:
            X, A, y = X.to(device), A.to(device), y.to(device)
            pred = model(X, A)
            total_loss += loss_fn(pred,y).item() * len(X)
    return total_loss / len(dataloader.dataset)


def save_hparams(args, yaml_path):
    params = {k: v for k, v in vars(args).items()
              if isinstance(v, (int, float, str, bool))}
    with open(yaml_path, "w") as f:
        yaml.safe_dump(params, f, sort_keys=False)

''' RL utilities '''

def update_target_network(target, online, tau=0.2):
    with torch.no_grad():
        for t_p, s_p in zip(target.parameters(), online.parameters()):
            t_p.data.mul_(1 - tau).add_(tau * s_p.data)


def get_target(model, target_model, X_next, R, nA, nAs, device, gamma = 0.99):
    with torch.no_grad():
        Q = []
        for A_id in range(nA):
            A_batch = np.full(len(X_next), A_id)
            A_batch = np.array(convert_factored_action(A_batch,nAs)).T
            q = model(torch.tensor(X_next,dtype=torch.float32,device=device),
                    torch.tensor(A_batch,dtype=torch.int64,device=device))
            Q.append(q.detach().cpu().numpy())
        Q = np.stack(Q,axis=1)
        best_actions = np.argmax(Q,axis=1)

        Q_target = []
        for i,a_id in enumerate(best_actions):
            a_fac = np.array(convert_factored_action([a_id],nAs)).T
            q = target_model(torch.tensor(X_next[i:i+1],dtype=torch.float32,device=device),
                            torch.tensor(a_fac,dtype=torch.int64,device=device))
            Q_target.append(q.item())
        Q_target = np.array(Q_target)

    y = (R + gamma * Q_target).squeeze()
    return y