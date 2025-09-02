from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib

@dataclass
class CFConfig:
    n_factors: int = 64
    n_epochs: int = 20
    lr_all: float = 0.005
    reg_all: float = 0.02
    random_state: int = 42

class CFRecommender:
    def __init__(self, cfg: CFConfig = CFConfig()):
        self.cfg = cfg
        self.model = SVD(n_factors=cfg.n_factors, n_epochs=cfg.n_epochs, lr_all=cfg.lr_all, reg_all=cfg.reg_all, random_state=cfg.random_state)
        self.all_items = None

    def fit(self, ratings: pd.DataFrame):
        reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
        data = Dataset.load_from_df(ratings[["userId","movieId","rating"]], reader)
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        self.all_items = sorted(ratings["movieId"].unique().tolist())

    def user_scores(self, user_id: int, item_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if item_ids is None:
            item_ids = self.all_items
        preds = [(iid, self.model.predict(uid=user_id, iid=iid).est) for iid in item_ids]
        return preds

    def recommend(self, user_id: int, k: int = 10, exclude_seen: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if exclude_seen is None: exclude_seen = []
        preds = self.user_scores(user_id)
        preds = [(iid, s) for (iid, s) in preds if iid not in exclude_seen]
        preds.sort(key=lambda x: -x[1])
        return preds[:k]

    def save(self, folder: str):
        joblib.dump({"cfg": self.cfg, "model": self.model, "all_items": self.all_items}, f"{folder}/cf_svd.joblib")

    @staticmethod
    def load(folder: str):
        d = joblib.load(f"{folder}/cf_svd.joblib")
        obj = CFRecommender(d["cfg"])
        obj.model = d["model"]
        obj.all_items = d["all_items"]
        return obj
