from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix
import joblib

@dataclass
class ContentIndexerConfig:
    max_features_text: int = 50000
    svd_components: Optional[int] = 300
    top_k_cast: int = 5
    top_k_crew: int = 3
    min_df: int = 2
    seed: int = 42

class ContentIndexer:
    def __init__(self, config: ContentIndexerConfig = ContentIndexerConfig()):
        self.cfg = config
        self.text_vec = TfidfVectorizer(
            max_features=self.cfg.max_features_text,
            min_df=self.cfg.min_df,
            stop_words='english'
        )
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_keywords = MultiLabelBinarizer()
        self.mlb_cast = MultiLabelBinarizer()
        self.mlb_crew = MultiLabelBinarizer()
        self.svd = TruncatedSVD(n_components=self.cfg.svd_components, random_state=self.cfg.seed) if self.cfg.svd_components else None
        self.movie_ids: List[int] = []
        self._item_matrix = None
        self._id2row = {}  # movieId -> row index

    def _pack_text(self, df: pd.DataFrame) -> pd.Series:
        cast = df["cast"].apply(lambda lst: lst[: self.cfg.top_k_cast] if isinstance(lst, list) else [])
        crew = df["crew"].apply(lambda lst: lst[: self.cfg.top_k_crew] if isinstance(lst, list) else [])
        text = (
            df["title"].fillna("") + " " +
            df["tagline"].fillna("") + " " +
            df["overview"].fillna("") + " " +
            cast.apply(lambda l: " ".join(l)).fillna("") + " " +
            crew.apply(lambda l: " ".join(l)).fillna("")
        )
        return text

    def fit(self, movies_joined: pd.DataFrame):
        text = self._pack_text(movies_joined)
        X_text = self.text_vec.fit_transform(text)

        X_gen = self.mlb_genres.fit_transform(movies_joined["genres"].apply(lambda x: x if isinstance(x, list) else []))
        X_kw  = self.mlb_keywords.fit_transform(movies_joined["keywords"].apply(lambda x: x if isinstance(x, list) else []))
        X_ca  = self.mlb_cast.fit_transform(movies_joined["cast"].apply(lambda x: x[: self.cfg.top_k_cast] if isinstance(x, list) else []))
        X_cr  = self.mlb_crew.fit_transform(movies_joined["crew"].apply(lambda x: x[: self.cfg.top_k_crew] if isinstance(x, list) else []))

        X = hstack([X_text, csr_matrix(X_gen), csr_matrix(X_kw), csr_matrix(X_ca), csr_matrix(X_cr)]).tocsr()

        if self.svd is not None:
            from scipy import sparse
            X = self.svd.fit_transform(X)
            X = sparse.csr_matrix(X)

        X = normalize(X)
        self.movie_ids = movies_joined["movieId"].tolist()
        self._item_matrix = X
        self._id2row = {mid: i for i, mid in enumerate(self.movie_ids)}

    def encode_items(self) -> csr_matrix:
        return self._item_matrix

    def build_user(self, user_history: pd.DataFrame, movies_joined: pd.DataFrame) -> np.ndarray:
        d = self._item_matrix.shape[1]
        if user_history is None or user_history.empty or "movieId" not in user_history.columns:
            return np.zeros((d,), dtype=float)

        uh = user_history.copy()
        # deduplicate by movieId
        if "rating" in uh.columns:
            uh = uh.groupby("movieId", as_index=False)["rating"].mean()
        else:
            uh = uh.drop_duplicates(subset=["movieId"])
            uh["rating"] = 5.0

        # keep only items present in index
        uh = uh[uh["movieId"].isin(self._id2row.keys())]
        if uh.empty:
            return np.zeros((d,), dtype=float)

        idx = [self._id2row[mid] for mid in uh["movieId"].tolist()]
        V = self._item_matrix[idx].toarray()          # (n_i, d)
        w = uh["rating"].astype(float).to_numpy()     # (n_i,)

        # mean-center & shift positive
        w = w - w.mean()
        w = (w - w.min()) + 1e-6

        assert V.shape[0] == w.shape[0], f"User profile mismatch: rows={V.shape[0]} vs weights={w.shape[0]}"
        profile = w @ V                                # -> (d,)
        norm = np.linalg.norm(profile) + 1e-12
        return profile / norm

    def recommend(self, user_vector: np.ndarray, k: int = 10, exclude_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        if exclude_ids is None:
            exclude_ids = []
        sims = self._item_matrix @ user_vector
        sims = np.asarray(sims).ravel()
        order = np.argsort(-sims)
        out = []
        for j in order:
            mid = self.movie_ids[j]
            if mid in exclude_ids:
                continue
            out.append((mid, float(sims[j])))
            if len(out) >= k:
                break
        return out

    def save(self, folder: str):
        joblib.dump({
            "cfg": self.cfg,
            "text_vec": self.text_vec,
            "mlb_genres": self.mlb_genres,
            "mlb_keywords": self.mlb_keywords,
            "mlb_cast": self.mlb_cast,
            "mlb_crew": self.mlb_crew,
            "svd": self.svd,
            "movie_ids": self.movie_ids,
        }, f"{folder}/cb_index.joblib")
        from scipy import sparse
        sparse.save_npz(f"{folder}/cb_items.npz", self._item_matrix)

    @staticmethod
    def load(folder: str):
        d = joblib.load(f"{folder}/cb_index.joblib")
        obj = ContentIndexer(d["cfg"])
        obj.text_vec     = d["text_vec"]
        obj.mlb_genres   = d["mlb_genres"]
        obj.mlb_keywords = d["mlb_keywords"]
        obj.mlb_cast     = d["mlb_cast"]
        obj.mlb_crew     = d["mlb_crew"]
        obj.svd          = d["svd"]
        obj.movie_ids    = d["movie_ids"]
        from scipy import sparse
        obj._item_matrix = sparse.load_npz(f"{folder}/cb_items.npz")
        obj._id2row      = {mid: i for i, mid in enumerate(obj.movie_ids)}
        return obj
