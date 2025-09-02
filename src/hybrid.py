from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

@dataclass
class HybridCfg:
    alpha: float = 0.5  # weight on CF; (1-alpha) on CB

class HybridRecommender:
    def __init__(self, cb, cf, cfg: HybridCfg = HybridCfg()):
        self.cb = cb
        self.cf = cf
        self.cfg = cfg

    def recommend(self, user_id: Optional[int] = None, user_profile_vec=None, k: int = 10, exclude_seen: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        exclude_seen = exclude_seen or []

        s_cf = []
        if user_id is not None and self.cf is not None:
            s_cf = self.cf.user_scores(user_id)
        s_cb = []
        if user_profile_vec is not None and self.cb is not None:
            sims = self.cb.encode_items() @ user_profile_vec
            sims = np.asarray(sims).ravel()
            s_cb = list(zip(self.cb.movie_ids, sims.tolist()))

        # Standardize scores before blending
        def _z(x):
            if not x:
                return x
            arr = np.array([s for _, s in x], dtype=float)
            mu, sd = arr.mean(), arr.std() + 1e-9
            return [(i, (s - mu) / sd) for i, s in x]

        s_cf = _z(s_cf)
        s_cb = _z(s_cb)

        from collections import defaultdict
        agg = defaultdict(float)
        for i, s in s_cf:
            agg[i] += self.cfg.alpha * s
        for i, s in s_cb:
            agg[i] += (1 - self.cfg.alpha) * s

        items = [(i, sc) for i, sc in agg.items() if i not in exclude_seen]
        items.sort(key=lambda x: -x[1])
        return items[:k]
