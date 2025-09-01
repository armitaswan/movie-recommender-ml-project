import numpy as np

def precision_at_k(y_true_set, y_pred_list, k=10):
    hit = sum(1 for i in y_pred_list[:k] if i in y_true_set)
    return hit / float(k)

def recall_at_k(y_true_set, y_pred_list, k=10):
    if not y_true_set:
        return 0.0
    hit = sum(1 for i in y_pred_list[:k] if i in y_true_set)
    return hit / float(len(y_true_set))

def hit_rate_at_k(y_true_set, y_pred_list, k=10):
    return 1.0 if any(i in y_true_set for i in y_pred_list[:k]) else 0.0

def dcg_at_k(y_true_set, y_pred_list, k=10):
    dcg = 0.0
    for idx, item in enumerate(y_pred_list[:k], start=1):
        rel = 1.0 if item in y_true_set else 0.0
        dcg += rel / np.log2(idx + 1)
    return dcg

def idcg_at_k(n_relevant, k=10):
    return sum(1.0 / np.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))

def ndcg_at_k(y_true_set, y_pred_list, k=10):
    idcg = idcg_at_k(len(y_true_set), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(y_true_set, y_pred_list, k) / idcg
