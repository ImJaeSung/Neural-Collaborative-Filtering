from data_utils import read_ratings, data_preprocessing
from data_utils import CustomDataset
from GMF import GMF
from MLP import MLP
from NeuMF import NeuMF
from metrics import metrics
import numpy as np
import torch
import torch.nn as nn
from torch.utils import DataLoader
import torch.optim as optim
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
ratings = read_ratings()
train_ratings, test_ratings = data_preprocessing()

train_data = CustomDataset(ratings, train_ratings, negative_num = 4)
test_data = CustomDataset(ratings, test_ratings, negative_num = 99)
n_users, n_items = train_data.get_num()

train_dataloader = DataLoader(dataset = train_data, batch_size = 256, shuffle = True)
test_dataloader = DataLoader(dataset = test_data, batch_size = 100, shuffle = False)

# model initialize
model = GMF(n_users = n_users, n_items = n_items, n_factors = 8, fusion = False, use_pretrain = False)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

n_epochs = 50
GMF_losses = []
GMF_HRs = []
GMF_NDCGs = []

for epoch in range(n_epochs):
    for user, item, label in train_dataloader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        preds = model(user, item)
        loss = loss_function(preds, label)
        GMF_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    HR, NDCG = metrics(model = model, test_loader = test_dataloader, 
                       top_k = 10, device = device)
    GMF_HRs.append(np.mean(HR))
    GMF_NDCGs.append(np.mean(NDCG))
    print("epoch: {}\tHR{:.3f}\tNDCG{:.3f}".format(epoch + 1, np.mean(HR), np.mean(NDCG)))

torch.save(model, '../pretrain/GMF.pth')

#%%
model = MLP(n_users = n_users, n_items = n_items, n_factors = 8, n_layers = 3, fusion = False, use_pretrain = False)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

n_epochs = 50
MLP_losses = []
MLP_HRs = []
MLP_NDCGs = []

for epoch in range(n_epochs):
    for user, item, label in train_dataloader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        preds = model(user, item)
        loss = loss_function(preds, label)
        MLP_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    HR, NDCG = metrics(model = model, test_loader = test_dataloader, 
                       top_k = 10, device = device)
    MLP_HRs.append(np.mean(HR))
    MLP_NDCGs.append(np.mean(NDCG))
    print("epoch: {}\tHR{:.3f}\tNDCG{:.3f}".format(epoch + 1, np.mean(HR), np.mean(NDCG)))

torch.save(model, '../pretrain/MLP.pth')

#%%
# pretrain ver.
GMF_pretrain = torch.load('../pretrain/GMF.pth')
MLP_pretrain = torch.load('../pretrain/MLP.pth')

for param in GMF_pretrain.parameters():
    param.requires_grad = False

for param in MLP_pretrain.parameters():
    param.requires_grad = False

model = NeuMF(n_users = n_users, n_items = n_items, n_factors = 8, n_layers = 3, fusion = True, 
              use_pretrain= True, GMF_pretrain = GMF_pretrain, MLP_pretrain = MLP_pretrain)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

n_epochs = 50
pretrain_NeuMF_losses = []
pretrain_NeuMF_HRs = []
pretrain_NeuMF_NDCGs = []

for epoch in range(n_epochs):
    for user, item, label in train_dataloader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        preds = model(user, item)
        loss = loss_function(preds, label)
        pretrain_NeuMF_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    HR, NDCG = metrics(model = model, test_loader = test_dataloader, 
                       top_k = 10, device = device)
    pretrain_NeuMF_HRs.append(np.mean(HR))
    pretrain_NeuMF_NDCGs.append(np.mean(NDCG))
    print("epoch: {}\tHR{:.3f}\tNDCG{:.3f}".format(epoch + 1, np.mean(HR), np.mean(NDCG)))

#%% 
# no pretrain ver
model = NeuMF(n_users = n_users, n_items = n_items, n_factors = 8, n_layers = 3, fusion = True, use_pretrain = False)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

n_epochs = 50
NeuMF_losses = []
NeuMF_HRs = []
NeuMF_NDCGs = []

for epoch in range(n_epochs):
    for user, item, label in train_dataloader:
        user = user.to(device)
        item = item.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        preds = model(user, item)
        loss = loss_function(preds, label)
        NeuMF_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    HR, NDCG = metrics(model = model, test_loader = test_dataloader, 
                       top_k = 10, device = device)
    NeuMF_HRs.append(np.mean(HR))
    NeuMF_NDCGs.append(np.mean(NDCG))
    print("epoch: {}\tHR{:.3f}\tNDCG{:.3f}".format(epoch + 1, np.mean(HR), np.mean(NDCG)))

