import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, n_users, n_items, n_factors, fusion, use_pretrain, GMF_pretrain = None):
        super(GMF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_embeddings = n_users, embedding_dim = n_factors)
        self.item_embedding = nn.Embedding(num_embeddings = n_items, embedding_dim = n_factors)
        # input을 one-hot encoding 하면 memory issue로 embedding층에서 nn.embedding 사용
        # 비슷한 효과라고 추측
        
        self.fusion = fusion
        self.use_pretrain = use_pretrain
        self.GMF_pretrain = GMF_pretrain

        pred_size = n_factors
        self.pred_layer = nn.Linear(pred_size, 1)
        self.sigmoid = nn.Sigmoid()
        
        if use_pretrain:
            self.user_embedding.weight.data.copy_(
                self.GMF_pretrain.user_embedding.weight)
            self.item_embedding.weight.data.copy_(
                self.GMF_pretrain.item_embedding.weight)
        else:
            if not fusion:
                nn.init.normal_(self.pred_layer.weight, mean = 0.0, std = 0.01)
            
            nn.init.normal_(self.user_embedding.weight, mean = 0, std = 0.01)
            nn.init.normal_(self.item_embedding.weight, mean = 0, std = 0.01)
    
    def forward(self, users, items):
        user_embedded = self.user_embedding(users)
        item_embedded = self.item_embedding(items)

        # hadamard product
        out_GMF = torch.mul(user_embedded, item_embedded)
        
        if self.fusion == False:
            pred = self.pred_layer(out_GMF)
            sigmoid_GMF = self.sigmoid(pred)
            return sigmoid_GMF.view(-1)
        else:
            return out_GMF
