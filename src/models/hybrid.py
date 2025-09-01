from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

@dataclass
class HybridRecommender:
    cb: any
    cf: any
    alpha: float = 0.6

    def recommend(self, user_id: int, user_profile_ids: Iterable[str], k: int = 10, exclude: Optional[Iterable[str]] = None):
        uvec = self.cb.build_user(user_profile_ids)
        if uvec is None:
            cb_scores = np.zeros(len(self.cb.movie_ids), dtype=float)
        else:
            cb_scores = (self.cb.item_matrix @ uvec.T).A.ravel()
        numeric_items = [int(x) if str(x).isdigit() else -1 for x in self.cb.movie_ids]
        cf_scores = self.cf.user_scores(user_id, numeric_items)
        scores = self.alpha * cf_scores + (1 - self.alpha) * cb_scores
        order = np.argsort(-scores)
        exclude_set = set(map(str, exclude or []))
        recs = []
        for idx in order:
            mid = self.cb.movie_ids[idx]
            if mid in exclude_set:
                continue
            recs.append((mid, float(scores[idx])))
            if len(recs) >= k:
                break
        return recs
