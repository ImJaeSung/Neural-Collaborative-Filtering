import torch
import torch.nn as nn
from GMF import GMF
from MLP import MLP

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors, n_layers, fusion, use_pretrain, GMF_pretrain = None, MLP_pretrain = None):
        super(NeuMF, self).__init__()
        self.GMF = GMF(n_users, n_items, n_factors, fusion, use_pretrain, GMF_pretrain)
        self.MLP = MLP(n_users, n_items, n_factors, n_layers, fusion, use_pretrain, MLP_pretrain)

        self.fusion = fusion
        self.use_pretrain = use_pretrain
        self.GMF_pretrain = GMF_pretrain
        self.MLP_pretrain = MLP_pretrain
        
        self.pred_layer = nn.Linear(2*n_factors, 1)
        self.sigmoid = nn.Sigmoid()
        
        if use_pretrain:
            temp_weight = torch.cat([self.GMF_pretrain.pred_layer.weight,
                                    self.MLP_pretrain.pred_layer.weight], dim= 1)
            temp_bias = self.GMF_pretrain.pred_layer.bias + self.MLP_pretrain.pred_layer.bias
            self.pred_layer.weight.data.copy_(0.5*temp_weight) # alpha = 0.5
            self.pred_layer.bias.data.copy_(0.5*temp_bias)
        else:     
            nn.init.normal_(self.pred_layer.weight, mean = 0.0, std = 0.01)

    def forward(self, users, items):
        input = torch.cat([self.MLP(users, items), self.GMF(users, items)], dim = -1)
        out_NeuMF = self.pred_layer(input)
        out_NeuMF = self.sigmoid(out_NeuMF)

        return out_NeuMF.view(-1)