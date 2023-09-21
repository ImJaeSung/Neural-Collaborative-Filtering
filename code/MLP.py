import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_users, n_items, n_factors, n_layers, fusion, use_pretrain, MLP_pretrain = None):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings = n_users, 
                                           embedding_dim = n_factors * (2**(n_layers-2)))
        self.item_embedding = nn.Embedding(num_embeddings = n_items, 
                                           embedding_dim = n_factors * (2**(n_layers-2)))
        
        self.fusion = fusion
        self.use_pretrain = use_pretrain
        self.MLP_pretrain = MLP_pretrain

        MLP_modules = []
        for i in range(1, n_layers):
            input_size = n_factors * (2**(n_layers-i))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.pred_layer = nn.Linear(n_factors, 1)
        self.sigmoid = nn.Sigmoid()

        if use_pretrain:
            self.user_embedding.weight.data.copy_(
               self.MLP_pretrain.user_embedding.weight)
            self.item_embedding.weight.data.copy_(
                self.MLP_pretrain.item_embedding.weight)
        else:
            if not fusion:
                nn.init.normal_(self.pred_layer.weight, mean = 0.0, std = 0.01)
            else:
                nn.init.normal_(self.user_embedding.weight, mean = 0.0, std = 0.01)
                nn.init.normal_(self.item_embedding.weight, mean = 0.0, std = 0.01)

    def forward(self, users, items):
        user_embedded = self.user_embedding(users)
        item_embedded = self.item_embedding(items)

        input = torch.cat([user_embedded, item_embedded], dim = -1)
        out_MLP = self.MLP_layers(input)

        if self.fusion == False:
            pred = self.pred_layer(out_MLP)
            sigmoid_MLP = self.sigmoid(pred)
            return sigmoid_MLP.view(-1)
        else:
            return out_MLP