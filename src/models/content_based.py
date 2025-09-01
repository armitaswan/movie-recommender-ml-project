from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from scipy import sparse
import re

def _prep_text(x: str) -> str:
    if not isinstance(x, str):
        return ''
    x = x.lower()
    x = re.sub(r'\s+', ' ', x).strip()
    return x

@dataclass
class ContentIndexer:
    max_features_text: int = 5000
    def fit(self, movies_df: pd.DataFrame):
        text = (movies_df['overview'].fillna('') + ' ' + movies_df['tagline'].fillna('')).map(_prep_text)
        self.tfidf = TfidfVectorizer(max_features=self.max_features_text, stop_words='english')
        X_text = self.tfidf.fit_transform(text)
        self.gen_mlb = MultiLabelBinarizer()
        X_gen = self.gen_mlb.fit_transform(movies_df['genres_list'].fillna([]))
        mats = [X_text, sparse.csr_matrix(X_gen)]
        if 'keywords_list' in movies_df.columns:
            self.kw_mlb = MultiLabelBinarizer()
            X_kw = self.kw_mlb.fit_transform(movies_df['keywords_list'].fillna([]))
            mats.append(sparse.csr_matrix(X_kw))
        else:
            self.kw_mlb = None
        self.item_matrix = normalize(sparse.hstack(mats).tocsr())
        self.movie_ids = movies_df['id'].astype(str).tolist()
        self.index_by_id = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.movies_df = movies_df[['id','title']].copy()
        return self

    def build_user(self, seen_item_ids: Iterable[str]):
        idxs = [self.index_by_id[str(i)] for i in seen_item_ids if str(i) in self.index_by_id]
        if not idxs:
            return None
        vecs = self.item_matrix[idxs]
        profile = vecs.mean(axis=0)
        return normalize(profile)

    def recommend(self, user_vector, k: int = 10, exclude: Optional[Iterable[str]] = None):
        scores = (self.item_matrix @ user_vector.T).A.ravel()
        order = np.argsort(-scores)
        titles = self.movies_df['title'].to_numpy()
        ids = np.array(self.movie_ids)
        ex = set(map(str, exclude or []))
        recs = []
        for idx in order:
            if ids[idx] in ex:
                continue
            recs.append((ids[idx], titles[idx], float(scores[idx])))
            if len(recs) >= k:
                break
        return recs
