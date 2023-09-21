import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def read_ratings():
    ratings = pd.read_csv('../dataset/ratings.dat', delimiter = '::', header = None, encoding = 'latin1',
                      names = ['userId', 'movieId', 'rating', 'timestamp'])
    ratings = ratings.drop(columns = ['timestamp'])

    return ratings

def data_preprocessing():
    train_ratings = read_ratings().copy()
    test_ratings = read_ratings().drop_duplicates(['userId'], keep = 'last')
    tmp = pd.concat([train_ratings, test_ratings], axis = 0)
    train_ratings = tmp.drop_duplicates(keep = False)

    # implict feedback
    train_ratings.loc[:,'rating'] = 1
    test_ratings.loc[:,'rating'] = 1

    return train_ratings, test_ratings

class CustomDataset(Dataset):
    def __init__(self, total_ratings, ratings, negative_num):
        super(CustomDataset, self).__init__()
        self.total_ratings = total_ratings
        self.ratings = ratings
        self.negative_num = negative_num
        self.n_users, self.n_items  = self.get_num()
        self.movieIds = self.get_movieIds()
        self.users, self.items, self.labels = self.negative_feedback()

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]
    
    def get_num(self):
        n_users = self.total_ratings['userId'].max() + 1
        n_items = self.total_ratings['movieId'].max() + 1

        return n_users, n_items
    
    def get_movieIds(self):
        movie_Ids = self.total_ratings['movieId'].unique()
        return movie_Ids
    
    def negative_feedback(self):
        users, items, labels = [], [], []
        user_item_set = set(zip(self.ratings['userId'], self.ratings['movieId']))
        total_user_item_set = set(zip(self.total_ratings['userId'], self.total_ratings['movieId']))
        
        for u, i in tqdm(user_item_set):
            # positive instance
            users.append(u)
            items.append(i)
            labels.append(1)
            tmp_check = []
            
            # negative feedback ratio
            negative_ratio = self.negative_num
            # negative instance
            for _ in range(negative_ratio):
                # random sampling
                negative_item = np.random.choice(self.movieIds)
                # checking interaction
                while (u, negative_item) in total_user_item_set or negative_item in tmp_check:
                    negative_item = np.random.choice(self.movieIds)
                users.append(u)
                items.append(negative_item)
                labels.append(0)
                tmp_check.append(negative_item)
        
        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)
