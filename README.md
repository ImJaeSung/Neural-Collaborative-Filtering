# Neural-Collaborative-Filtering
Repository for implementing Neural Collaborative Filtering (2017) 

![image](https://github.com/ImJaeSung/Neural-Collaborative-Filtering/assets/113405066/0daa12b2-0b27-4288-a22b-4ab47f0110ab)


Download dataset : http://grouplens.org/datasets/movielens/1m/ (논문 5p)

### code

|name|explanation|
|----|-----------|
|data_util.py|Data preprocessing code|
|GMF|General Matrix Factorization Layer class|
|MLP|Multi Layer Perceptron Layer class|
|NeuMF|Neural Matrix Factorization model class|
|matrics|Hit ratio and NDCG|
|main|Train the each model|

### pretrain

|name|explanation|
|----|------|
|GMF.pth|n_factors = 8|
|MLP.pth|n_factors = 8, n_layers = 3|

### Result
1. Performance of NeuMF with and without pre-training.

|-|With Pre-training|Without Pre-training|
|-------|-----|-----|
|Factors|HR@10|HR@10|
|8|0.6950|0.7230|

|-|With Pre-training|Without Pre-training|
|-------|-----|-----|
|Factors|HR@10|HR@10|
|8|0.0.4259|0.4776|

2. HR@10 of MLP with different layers.

|Factors|MLP-2|MLP-3|MLP-4|
|-------|-----|-----|------|
|8|0.6382|0.6742|0.6970|

3. NDCG@10 of MLP with different layers.

|Factors|MLP-2|MLP-3|MLP-4|
|-------|-----|-----|------|
|8|0.3778|0.4089|0.4296|
