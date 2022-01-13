import torch
import math
class GAT(torch.nn.Module):
    def __init__(self, in_features, out_features, nheads=3, alpha=0.2, bias=True, type=None):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nheads = nheads
        self.bias = bias
        if nheads>1:
            self.weights = torch.Parameter(torch.FloatTensor(in_features, out_features) )
        if self.bias:
            self.bias = torch.Parameter(torch.FloatTensor(out_features))    
        self.alpha = torch.Parameter(torch.FloatTensor(out_features*2 , 1))
        self.leaky = torch.nn.LeakyReLU(alpha=alpha)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weights, gain=1.44)
        torch.nn.init.xavier_uniform_(self.alpha, gain=1.44)
    def forward(self, x, adj=None):
        Wh = torch.mm(x, self.weights)
        e = self.prepare_forward(Wh)
        nonzero_vec = 9e-5*torch.ones_like(e) 
        attention = torch.where(adj > 0, e, nonzero_vec)
        attention = torch.nn.functional.dropout(attention, 0.2)
        attention = torch.softmax(attention)
        x = torch.spmm(attention, Wh)
        if self.bias:
            x = x + self.bias
        return x
    def prepare_forward(self, x):
        w1 = self.matmul(x, self.alpha[:self.out_features, :])
        w2 = self.matmul(x, self.alpha[self.out_features:, :])
        w = w1 + w2.T
        w = self.leaky(w)
        return w