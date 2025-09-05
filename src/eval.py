from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from typing import List, Dict, Tuple
from itertools import combinations

from .content_based import ContentIndexer
from .collaborative import CFRecommender
try:
    from .neighborhood import KNNRecommender
    _HAS_KNN = True
except Exception:
    _HAS_KNN = False

from .hybrid import HybridRecommender, HybridCfg

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = Path(__file__).resolve().parents[1] / "models"
ART = Path(__file__).resolve().parents[1] / "artifacts"

def leave_last_out(ratings: pd.DataFrame):
    ratings = ratings.dropna(subset=['timestamp']).copy()
    ratings = ratings.sort_values(['userId','timestamp'])
    last = ratings.groupby('userId').tail(1)
    train = ratings.drop(last.index)
    return train, last

def precision_at_k(rec, gt, k):
    rec_k = rec[:k]; return len(set(rec_k) & set(gt)) / max(1, k)

def recall_at_k(rec, gt, k):
    rec_k = rec[:k]; return len(set(rec_k) & set(gt)) / max(1, len(gt))

def hit_rate_at_k(rec, gt, k):
    rec_k = rec[:k]; return 1.0 if any(i in rec_k for i in gt) else 0.0

def ndcg_at_k(rec, gt, k):
    rec_k = rec[:k]
    dcg = 0.0
    for r, it in enumerate(rec_k, start=1):
        if it in gt: dcg += 1.0 / np.log2(r+1)
    idcg = 1.0
    return dcg / idcg

def apk(rec, gt, k):
    if not gt: return 0.0
    rec_k = rec[:k]
    score, hits = 0.0, 0
    for i, it in enumerate(rec_k, start=1):
        if it in gt:
            hits += 1
            score += hits / i
    return score / min(len(gt), k)

def catalog_coverage(all_rec_lists: List[List[int]], catalog_size: int) -> float:
    uniq = set()
    for rec in all_rec_lists:
        uniq.update(rec)
    return len(uniq) / max(1, catalog_size)

def novelty(rec: List[int], item_pop: Dict[int,int], n_users: int) -> float:
    vals = []
    for i in rec:
        p = item_pop.get(i, 1)
        vals.append(-np.log2(p / max(1, n_users)))
    return float(np.mean(vals)) if vals else 0.0

def intra_list_diversity(rec: List[int], item_vecs: Dict[int, np.ndarray]) -> float:
    pairs = list(combinations(rec, 2))
    if not pairs: return 0.0
    sims = []
    for a,b in pairs:
        va, vb = item_vecs.get(a), item_vecs.get(b)
        if va is None or vb is None: continue
        na = np.linalg.norm(va) + 1e-12
        nb = np.linalg.norm(vb) + 1e-12
        sims.append(float((va @ vb) / (na*nb)))
    if not sims: return 0.0
    return 1.0 - float(np.mean(sims))

def evaluate(k_list: List[int], bootstrap: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    movies_joined = pd.read_parquet(DATA / "movies_joined.parquet")

    cb = ContentIndexer.load(str(MODELS))
    cf = CFRecommender.load(str(MODELS))
    cfg = joblib.load(MODELS / "hybrid_cfg.joblib")
    hy = HybridRecommender(cb, cf, HybridCfg(alpha=cfg["alpha"]))
    if _HAS_KNN:
        kn = KNNRecommender.load(str(MODELS))

    train, test = leave_last_out(ratings)
    user_hist = train.groupby('userId')

    # popularity & catalog stats
    item_pop = train.groupby("movieId")["userId"].nunique().to_dict()
    n_users  = train["userId"].nunique()
    catalog  = set(train["movieId"].unique().tolist())
    catalog_size = len(catalog)

    # cache CB item vectors (dense) for ILD
    item_vecs = {}
    V = cb.encode_items().toarray()
    for mid, row in zip(cb.movie_ids, V):
        item_vecs[mid] = row

    users = sorted(set(test['userId']).intersection(set(train['userId'])))
    rec_store = {m: [] for m in (["CB","CF","HY","KNN"] if _HAS_KNN else ["CB","CF","HY"])}

    rows = []
    for uid in tqdm(users, desc="Evaluating"):
        hist = user_hist.get_group(uid)
        uv = cb.build_user(hist, movies_joined)
        seen = hist['movieId'].tolist()
        gt = test[test['userId'] == uid]['movieId'].tolist()

        rec_cb = [i for i,_ in cb.recommend(uv, k=max(k_list), exclude_ids=seen)]
        rec_cf = [i for i,_ in cf.recommend(uid, k=max(k_list), exclude_seen=seen)]
        rec_hy = [i for i,_ in hy.recommend(user_id=uid, user_profile_vec=uv, k=max(k_list), exclude_seen=seen)]
        if _HAS_KNN:
            rec_kn = [i for i,_ in kn.recommend(uid, k=max(k_list), exclude_seen=seen)]

        if _HAS_KNN:
            rec_store["KNN"].append(rec_kn)
        rec_store["CB"].append(rec_cb)
        rec_store["CF"].append(rec_cf)
        rec_store["HY"].append(rec_hy)

        for k in k_list:
            row = {
                "userId": uid,
                f"P@{k}_CB": precision_at_k(rec_cb, gt, k),
                f"P@{k}_CF": precision_at_k(rec_cf, gt, k),
                f"P@{k}_HY": precision_at_k(rec_hy, gt, k),
                f"R@{k}_CB": recall_at_k(rec_cb, gt, k),
                f"R@{k}_CF": recall_at_k(rec_cf, gt, k),
                f"R@{k}_HY": recall_at_k(rec_hy, gt, k),
                f"HR@{k}_CB": hit_rate_at_k(rec_cb, gt, k),
                f"HR@{k}_CF": hit_rate_at_k(rec_cf, gt, k),
                f"HR@{k}_HY": hit_rate_at_k(rec_hy, gt, k),
                f"MAP@{k}_CB": apk(rec_cb, gt, k),
                f"MAP@{k}_CF": apk(rec_cf, gt, k),
                f"MAP@{k}_HY": apk(rec_hy, gt, k),
                f"NDCG@{k}_CB": ndcg_at_k(rec_cb, gt, k),
                f"NDCG@{k}_CF": ndcg_at_k(rec_cf, gt, k),
                f"NDCG@{k}_HY": ndcg_at_k(rec_hy, gt, k),
                f"COV@{k}_CB": catalog_coverage([rec_cb[:k]], catalog_size),
                f"COV@{k}_CF": catalog_coverage([rec_cf[:k]], catalog_size),
                f"COV@{k}_HY": catalog_coverage([rec_hy[:k]], catalog_size),
                f"NOV@{k}_CB": novelty(rec_cb[:k], item_pop, n_users),
                f"NOV@{k}_CF": novelty(rec_cf[:k], item_pop, n_users),
                f"NOV@{k}_HY": novelty(rec_hy[:k], item_pop, n_users),
                f"ILD@{k}_CB": intra_list_diversity(rec_cb[:k], item_vecs),
                f"ILD@{k}_CF": intra_list_diversity(rec_cf[:k], item_vecs),
                f"ILD@{k}_HY": intra_list_diversity(rec_hy[:k], item_vecs),
            }
            if _HAS_KNN:
                row.update({
                    f"P@{k}_KNN": precision_at_k(rec_kn, gt, k),
                    f"R@{k}_KNN": recall_at_k(rec_kn, gt, k),
                    f"HR@{k}_KNN": hit_rate_at_k(rec_kn, gt, k),
                    f"MAP@{k}_KNN": apk(rec_kn, gt, k),
                    f"NDCG@{k}_KNN": ndcg_at_k(rec_kn, gt, k),
                    f"COV@{k}_KNN": catalog_coverage([rec_kn[:k]], catalog_size),
                    f"NOV@{k}_KNN": novelty(rec_kn[:k], item_pop, n_users),
                    f"ILD@{k}_KNN": intra_list_diversity(rec_kn[:k], item_vecs),
                })
            rows.append(row)

    ART.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "eval_results.csv", index=False)

    # Bootstrap CIs (mean over users)
    metrics = sorted([c for c in df.columns if c != "userId"])
    out = []
    for m in metrics:
        vals = df[m].to_numpy()
        boot = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(vals), size=len(vals))
            boot.append(np.mean(vals[idx]))
        boot = np.array(boot)
        mean = float(np.mean(vals))
        lo, hi = np.percentile(boot, [2.5, 97.5]).tolist()
        out.append({"metric": m, "mean": mean, "ci95_low": float(lo), "ci95_high": float(hi)})
    pd.DataFrame(out).to_csv(ART / "eval_summary.csv", index=False)

    # Global catalog coverage across all users' lists
    for k in k_list:
        cov = {
            f"GlobalCOV@{k}_CB": catalog_coverage([r[:k] for r in rec_store["CB"]], catalog_size),
            f"GlobalCOV@{k}_CF": catalog_coverage([r[:k] for r in rec_store["CF"]], catalog_size),
            f"GlobalCOV@{k}_HY": catalog_coverage([r[:k] for r in rec_store["HY"]], catalog_size),
        }
        if _HAS_KNN:
            cov[f"GlobalCOV@{k}_KNN"] = catalog_coverage([r[:k] for r in rec_store["KNN"]], catalog_size)
        pd.Series(cov).to_csv(ART / f"eval_global_cov_k{k}.csv")

    return df

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--k_list', nargs='+', type=int, default=[10, 20])
    p.add_argument('--bootstrap', type=int, default=200)  # NEW
    args = p.parse_args()
    df = evaluate(args.k_list, bootstrap=args.bootstrap)
    # quick text summary
    print(df.describe())
