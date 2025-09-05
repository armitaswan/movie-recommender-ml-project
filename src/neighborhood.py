from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import joblib

@dataclass
class KNNConfig:
    k_neighbors: int = 50 
    min_ratings_item: int = 2

class KNNRecommender:
    def __init__(self, cfg: KNNConfig = KNNConfig()):
        self.cfg = cfg
        self.user_map = None
        self.item_map = None
        self.R = None            # user-item matrix (sparse)
        self.sim_top = None      # dict: item_index -> [(j, sim), ...]

    def fit(self, ratings: pd.DataFrame):
        item_counts = ratings.groupby("movieId")["rating"].count()
        keep_items = set(item_counts[item_counts >= self.cfg.min_ratings_item].index.tolist())
        r = ratings[ratings["movieId"].isin(keep_items)].copy()

        users = sorted(r["userId"].unique().tolist())
        items = sorted(r["movieId"].unique().tolist())
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {m:j for j,m in enumerate(items)}

        rows = [self.user_map[u] for u in r["userId"]]
        cols = [self.item_map[m] for m in r["movieId"]]
        vals = r["rating"].astype(float).to_numpy()
        self.R = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

        self.sim_top = {}
        block = 1024
        for start in range(0, self.R.shape[1], block):
            end = min(start+block, self.R.shape[1])
            sims = cosine_similarity(self.R.T[start:end], self.R.T)  # (b, n_items)
            for local_idx in range(end-start):
                j = start + local_idx
                row = sims[local_idx]
                row[j] = -1
                top_idx = np.argpartition(-row, self.cfg.k_neighbors)[:self.cfg.k_neighbors]
                top = sorted([(ti, float(row[ti])) for ti in top_idx], key=lambda x: -x[1])
                self.sim_top[j] = top

    def recommend(self, user_id: int, k: int = 10, exclude_seen: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if self.R is None or user_id not in self.user_map:
            return []
        seen = set(exclude_seen or [])
        u = self.user_map[user_id]
        user_vec = self.R[u].toarray().ravel()
        rated_idx = np.where(user_vec > 0)[0]

        scores = {}
        for idx in rated_idx:
            r_ui = user_vec[idx]
            for j, sim in self.sim_top.get(idx, []):
                if j in rated_idx: 
                    continue
                if list(self.item_map.keys())[j] in seen:
                    continue
                scores[j] = scores.get(j, 0.0) + sim * r_ui

        items = list(self.item_map.keys())
        ranked = sorted([(items[j], sc) for j, sc in scores.items()], key=lambda x: -x[1])
        return ranked[:k]

    def save(self, folder: str):
        joblib.dump({"cfg": self.cfg, "user_map": self.user_map, "item_map": self.item_map,
                     "sim_top": self.sim_top, "shape": (self.R.shape if self.R is not None else None)}, f"{folder}/knn.joblib")

    @staticmethod
    def load(folder: str) -> "KNNRecommender":
        d = joblib.load(f"{folder}/knn.joblib")
        obj = KNNRecommender(d["cfg"])
        obj.user_map = d["user_map"]
        obj.item_map = d["item_map"]
        obj.sim_top = d["sim_top"]
        return obj
