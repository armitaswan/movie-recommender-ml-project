import pandas as pd
from src.config import PROC_DIR, MODELS_DIR, ALPHA, SEED
from src.models.content_based import ContentIndexer
from src.models.mf import MatrixFactorization
from src.models.hybrid import HybridRecommender
from src.utils import save_pickle

def main():
    movies = pd.read_parquet(PROC_DIR / 'movies.parquet')
    ratings = pd.read_parquet(PROC_DIR / 'ratings.parquet')
    cb = ContentIndexer(max_features_text=8000).fit(movies)
    cf = MatrixFactorization(n_factors=80, n_epochs=20, random_state=SEED).fit(ratings)
    hyb = HybridRecommender(cb=cb, cf=cf, alpha=ALPHA)
    save_pickle(cb, MODELS_DIR / 'content_indexer.pkl')
    save_pickle(cf, MODELS_DIR / 'mf_model.pkl')
    save_pickle(hyb, MODELS_DIR / 'hybrid.pkl')
    print('Saved content_indexer.pkl, mf_model.pkl, hybrid.pkl')

if __name__ == '__main__':
    main()
