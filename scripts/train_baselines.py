import pandas as pd
from src.config import PROC_DIR, MODELS_DIR, MIN_VOTES_PERCENTILE
from src.baselines.popularity import imdb_weighted_rating
from src.utils import save_pickle

def main():
    movies = pd.read_parquet(PROC_DIR / 'movies.parquet')
    ranked = imdb_weighted_rating(movies, m_threshold=MIN_VOTES_PERCENTILE)
    save_pickle(ranked[['id','title','wr_score']], MODELS_DIR / 'popularity_wr.pkl')
    print('Saved popularity baseline to models/popularity_wr.pkl')

if __name__ == '__main__':
    main()
