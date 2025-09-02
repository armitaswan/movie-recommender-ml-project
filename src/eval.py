from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from .content_based import ContentIndexer
from .collaborative import CFRecommender
from .hybrid import HybridRecommender, HybridCfg

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = Path(__file__).resolve().parents[1] / "models"
ART = Path(__file__).resolve().parents[1] / "artifacts"

def leave_last_out(ratings: pd.DataFrame):
    '''Time-aware split: last interaction per user is test; others train.'''
    ratings = ratings.dropna(subset=['timestamp']).copy()
    ratings = ratings.sort_values(['userId','timestamp'])
    last = ratings.groupby('userId').tail(1)
    train = ratings.drop(last.index)
    return train, last

def precision_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k).intersection(ground_truth))
    return hits / k

def recall_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    denom = max(1, len(ground_truth))
    hits = len(set(rec_k).intersection(ground_truth))
    return hits / denom

def hit_rate_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    return 1.0 if any(i in rec_k for i in ground_truth) else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(rec_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 1)
    idcg = 1.0
    return dcg / idcg

def evaluate(k_list):
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    movies_joined = pd.read_parquet(DATA / "movies_joined.parquet")

    cb = ContentIndexer.load(str(MODELS))
    cf = CFRecommender.load(str(MODELS))
    cfg = joblib.load(MODELS / "hybrid_cfg.joblib")
    hy = HybridRecommender(cb, cf, HybridCfg(alpha=cfg["alpha"]))

    train, test = leave_last_out(ratings)
    user_hist = train.groupby('userId')

    results = []

    all_users = sorted(set(test['userId']).intersection(set(train['userId'])))
    for uid in tqdm(all_users, desc="Evaluating"):
        hist = user_hist.get_group(uid) if uid in user_hist.groups else pd.DataFrame(columns=ratings.columns)
        uv = cb.build_user(hist, movies_joined)
        seen = hist['movieId'].tolist()

        rec_cb = [i for i, s in cb.recommend(uv, k=max(k_list), exclude_ids=seen)]
        rec_cf = [i for i, s in cf.recommend(uid, k=max(k_list), exclude_seen=seen)]
        rec_hy = [i for i, s in hy.recommend(user_id=uid, user_profile_vec=uv, k=max(k_list), exclude_seen=seen)]

        gt = test[test['userId'] == uid]['movieId'].tolist()

        row = {'userId': uid}
        for k in k_list:
            row.update({
                f'P@{k}_CB': precision_at_k(rec_cb, gt, k),
                f'P@{k}_CF': precision_at_k(rec_cf, gt, k),
                f'P@{k}_HY': precision_at_k(rec_hy, gt, k),
                f'HR@{k}_CB': hit_rate_at_k(rec_cb, gt, k),
                f'HR@{k}_CF': hit_rate_at_k(rec_cf, gt, k),
                f'HR@{k}_HY': hit_rate_at_k(rec_hy, gt, k),
                f'NDCG@{k}_CB': ndcg_at_k(rec_cb, gt, k),
                f'NDCG@{k}_CF': ndcg_at_k(rec_cf, gt, k),
                f'NDCG@{k}_HY': ndcg_at_k(rec_hy, gt, k),
            })
        results.append(row)

    df = pd.DataFrame(results)
    ART.mkdir(parents=True, exist_ok=True)
    df.to_csv(ART / 'eval_results.csv', index=False)
    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k_list', nargs='+', type=int, default=[10, 20])
    args = p.parse_args()
    df = evaluate(args.k_list)
    print(df.describe())
