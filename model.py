import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CosineSimilarity
import math
from KAN import KANLinear

class GCKLayer(nn.Module):
    def __init__(self, gamma, layer):
        super(GCKLayer, self).__init__()
        self.gamma = gamma
        self.layer = layer

    def rbf_kernel_X(self, X, gamma):
        n = X.shape[0]
        Sij = torch.matmul(X, X.T)
        Si = torch.unsqueeze(torch.diag(Sij), 0).T @ torch.ones(1, n).to(X.device)
        Sj = torch.ones(n, 1).to(X.device) @ torch.unsqueeze(torch.diag(Sij), 0)
        D2 = Si + Sj - 2 * Sij
        return torch.exp(-D2 * gamma)

    def rbf_kernel_K(self, K_t, gamma):
        n = K_t.shape[0]
        s = torch.unsqueeze(torch.diag(K_t), 0)
        D2 = torch.ones(n, 1).to(K_t.device) @ s + s.T @ torch.ones(1, n).to(K_t.device) - 2 * K_t
        return torch.exp(-D2 * gamma)

    def forward(self, adj, inputs):
        if self.layer == 0:
            X_t = torch.matmul(adj, inputs)
            return self.rbf_kernel_X(X_t, self.gamma)
        else:
            K_t = torch.matmul(torch.matmul(adj, inputs), adj.t())
            return self.rbf_kernel_K(K_t, self.gamma)

class GCKM(nn.Module):
    def __init__(self, gamma_list):
        super(GCKM, self).__init__()
        self.model = nn.ModuleList()
        for i, gamma in enumerate(gamma_list):
            self.model.append(GCKLayer(gamma, i))

    def forward(self, adj, X):
        K = X
        for layer in self.model:
            K = layer(adj, K)
        return K

class GeneKAN(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim, gamma_list,
                 alpha, device, reduction, num_nodes, num_head1, num_head2,
                 decode_type='KAN'):
        super(GeneKAN, self).__init__()
        self.device = device
        self.alpha = alpha
        self.reduction = reduction
        self.num_nodes = num_nodes
        self.gamma = gamma_list
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.decode_type = decode_type

        self.gckm1 = nn.ModuleList([GCKM(gamma_list) for _ in range(num_head1)])
        self.gckm2 = nn.ModuleList([GCKM(gamma_list) for _ in range(num_head2)])

        self.attn_weights1 = nn.Parameter(torch.Tensor(num_head1))
        self.attn_weights2 = nn.Parameter(torch.Tensor(num_head2))
        nn.init.normal_(self.attn_weights1)
        nn.init.normal_(self.attn_weights2)

        self.hidden1_dim = hidden1_dim * num_head1 if reduction == 'concate' else hidden1_dim
        self.hidden2_dim = hidden2_dim * num_head2 if reduction == 'concate' else hidden2_dim

        self.linear_0 = nn.Linear(num_nodes * num_head2, self.hidden2_dim)
        self.tf_linear1 = nn.Linear(self.hidden2_dim, output_dim)
        self.target_linear1 = nn.Linear(self.hidden2_dim, output_dim)

        if decode_type == 'MLP':
            self.linear = nn.Linear(2 * output_dim, 1)
        elif decode_type == 'KAN':
            self.linear = KANLinear(2 * output_dim, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.tf_linear1.weight)
        nn.init.xavier_uniform_(self.target_linear1.weight)
        if self.decode_type == 'MLP':
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def encode(self, x, adj):
        gckm1_outputs = [gckm(adj, x) for gckm in self.gckm1]
        stacked_gckm1 = torch.stack(gckm1_outputs, dim=0)  # [h1, n, n]
        attn1 = F.softmax(self.attn_weights1, dim=0)
        K1 = torch.einsum('h,hij->ij', attn1, stacked_gckm1)
        K1 = F.elu(K1)

        gckm2_outputs = [gckm(adj, K1) for gckm in self.gckm2]
        stacked_gckm2 = torch.stack(gckm2_outputs, dim=0)  # [h2, n, n]
        attn2 = F.softmax(self.attn_weights2, dim=0)
        K2 = torch.einsum('h,hij->ij', attn2, stacked_gckm2)
        K2 = F.elu(K2)

        if self.reduction == 'mean':
            embed = torch.mean(K2, dim=1)
        elif self.reduction == 'concate':
            embed = K2.view(K2.size(0), -1)
        return embed

    def decode(self, tf_embed, target_embed):
        if self.decode_type == 'dot':
            return torch.sum(tf_embed * target_embed, dim=1, keepdim=True)
        elif self.decode_type == 'cosine':
            return F.cosine_similarity(tf_embed, target_embed, dim=1).unsqueeze(1)
        else:
            combined = torch.cat([tf_embed, target_embed], dim=1)
            return self.linear(combined)

    def forward(self, x, adj, train_sample):
        embed = self.encode(x, adj)
        embed = F.dropout(F.elu(self.linear_0(embed)), p=0.01)
        
        tf_embed = F.elu(self.tf_linear1(embed))
        target_embed = F.elu(self.target_linear1(embed))
        
        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]
        return self.decode(train_tf, train_target)

    def get_embedding(self):
        return self.tf_linear1.weight, self.target_linear1.weight

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(self.transformer_encoder(x))