import torch
import numpy as np
import math

class GraphConv(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W1 = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if self.bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stv = 1/ math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stv, stv)
        if self.bias:
            self.bias.data.uniform_(-stv, stv)
        
    def forward(self, x, adj):
        x = torch.mm(x, self.W1)
        x = torch.spmm(x, adj)
        if self.bias:
            x = x + self.bias
        return x