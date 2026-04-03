from utils import build_mlp
from torch import nn
import torch

class AttentionNetwork(nn.Module):
    def __init__(self, mask, state_size, action_dims, d_model, num_heads, num_layers, hidden_dim):
        super().__init__()
        self.dA, self.d_model = len(action_dims), d_model
        self.sub_action_embeddings = nn.ModuleList([nn.Embedding(a_dim,d_model) for a_dim in action_dims])
        self.film_gamma, self.film_beta = nn.Linear(state_size,d_model*self.dA), nn.Linear(state_size,d_model*self.dA)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.register_buffer("attn_mask", ~mask)
        self.sub_action_mlps = nn.ModuleList([build_mlp(d_model,1,num_layers,hidden_dim) for _ in range(self.dA)])

    def forward(self, state_input, action_input):
        B = state_input.size(0)
        action_emb = torch.stack([emb(action_input[:,i]) for i,emb in enumerate(self.sub_action_embeddings)], dim=1)
        gamma = self.film_gamma(state_input).view(B,self.dA,self.d_model)
        beta  = self.film_beta(state_input).view(B,self.dA,self.d_model)
        action_emb = gamma * action_emb + beta
        attended,_ = self.self_attention(action_emb, action_emb, action_emb, attn_mask=self.attn_mask)
        sub_qs = [mlp(attended[:,i,:]) for i,mlp in enumerate(self.sub_action_mlps)]
        total_q = torch.stack(sub_qs,dim=1).sum(dim=1)
        return total_q.squeeze(-1)

class DenseNetwork(nn.Module):
    def __init__(self, groups, state_size, action_dims, num_layers, hidden_dim):
        super().__init__()
        self.dA = len(action_dims)
        self.groups = groups
        self.num_groups = max(groups) + 1
        mlps = []
        for group in range(self.num_groups):
            mlps.append(build_mlp(self.groups.count(group) + state_size, 1, num_layers = num_layers, hidden_dim = hidden_dim))
        self.sub_action_mlps = nn.ModuleList(mlps)
        
    def forward(self, state_input, action_input):
        q_vals = []
        for i in range(self.num_groups):
            indices = [j for j, x in enumerate(self.groups) if x == i]
            x_i = torch.cat((state_input, action_input[:, indices]), dim=1)
            q_i = self.sub_action_mlps[i](x_i)
            q_vals.append(q_i)
        q_vals = torch.cat(q_vals, dim=1)
        total_q = q_vals.sum(dim=1)
        return total_q