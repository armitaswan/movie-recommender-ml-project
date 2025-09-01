from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

@dataclass
class MatrixFactorization:
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.005
    reg_all: float = 0.02
    random_state: int = 42

    def fit(self, ratings: pd.DataFrame):
        reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
        data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
        trainset = data.build_full_trainset()
        self.algo = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs,
                        lr_all=self.lr_all, reg_all=self.reg_all,
                        random_state=self.random_state)
        self.algo.fit(trainset)
        return self

    def user_scores(self, user_id: int, item_ids: Iterable[int]):
        scores = []
        for iid in item_ids:
            try:
                pred = self.algo.predict(uid=int(user_id), iid=int(iid), verbose=False)
                scores.append(pred.est)
            except Exception:
                scores.append(np.nan)
        arr = np.array(scores, dtype=float)
        if np.isnan(arr).any():
            arr[np.isnan(arr)] = np.nanmean(arr)
        return arr

    def recommend(self, user_id: int, candidate_item_ids: Iterable[int], k: int = 10, exclude: Optional[Iterable[int]] = None):
        candidate_item_ids = list(candidate_item_ids)
        scores = self.user_scores(user_id, candidate_item_ids)
        order = np.argsort(-scores)
        recs, ex = [], set(exclude or [])
        for idx in order:
            item = candidate_item_ids[idx]
            if item in ex:
                continue
            recs.append((item, float(scores[idx])))
            if len(recs) >= k:
                break
        return recs
