import numpy as np
import torch

def hit_ratio(get_item, pred_item):
    if get_item in pred_item:
        return 1
    return 0

def ndcg(get_item, pred_item):
    if get_item in pred_item:
        idx = pred_item.index(get_item)
        return np.reciprocal(np.log2(idx + 2))
    return 0

def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []

    for user, item, _ in test_loader:
        user = user.to(device)
        item = item.to(device)

        preds = model(user, item)
        # top k choice, value and index
        _, idx = torch.topk(preds, top_k)

        recommends = torch.take(item, idx).cpu().numpy().tolist()

        get_item = item[0].item()
        HR.append(hit_ratio(get_item, recommends))
        NDCG.append(ndcg(get_item, recommends))

    return np.mean(HR), np.mean(NDCG)