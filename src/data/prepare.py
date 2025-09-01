from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from ..config import RAW_DIR, PROC_DIR
from ..utils import save_json

def _parse_json_list(raw):
    if pd.isna(raw):
        return []
    try:
        obj = json.loads(raw)
        return [d.get('name') for d in obj if isinstance(d, dict) and 'name' in d]
    except Exception:
        return []

def load_and_clean_movies() -> pd.DataFrame:
    movies_path = RAW_DIR / 'movies_metadata.csv'
    keywords_path = RAW_DIR / 'keywords.csv'
    credits_path = RAW_DIR / 'credits.csv'
    links_path = RAW_DIR / 'links.csv'

    movies = pd.read_csv(movies_path, low_memory=False)
    before = len(movies)
    keep = ['id','imdb_id','title','original_title','overview','tagline','genres','original_language','release_date','runtime','popularity','vote_average','vote_count']
    movies = movies[keep].copy()
    movies['genres_list'] = movies['genres'].apply(_parse_json_list)
    for col in ['runtime','popularity','vote_average','vote_count']:
        movies[col] = pd.to_numeric(movies[col], errors='coerce')
    movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
    if keywords_path.exists():
        kw = pd.read_csv(keywords_path)
        kw['keywords_list'] = kw['keywords'].apply(_parse_json_list)
        kw = kw[['id','keywords_list']]
        movies = movies.merge(kw, on='id', how='left')
    if credits_path.exists():
        cr = pd.read_csv(credits_path)
        def _names(raw):
            if pd.isna(raw):
                return []
            try:
                arr = json.loads(raw)
                return [d.get('name') for d in arr[:5] if isinstance(d, dict) and 'name' in d]
            except Exception:
                return []
        cr = cr[['id','cast','crew']].copy()
        cr['cast_names'] = cr['cast'].apply(_names)
        cr['crew_names'] = cr['crew'].apply(_names)
        cr = cr[['id','cast_names','crew_names']]
        movies = movies.merge(cr, on='id', how='left')
    after = len(movies)
    dropped = before - after
    if links_path.exists():
        links = pd.read_csv(links_path)
        movies = movies.merge(links, left_on='id', right_on='tmdbId', how='left')
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / 'movies.parquet'
    movies.to_parquet(out_path, index=False)
    meta = {'rows_before': int(before),'rows_after': int(after),'rows_dropped': int(dropped),'columns': movies.columns.tolist()}
    save_json(meta, PROC_DIR / 'movies_stats.json')
    return movies

def load_and_clean_ratings() -> pd.DataFrame:
    ratings_path_small = RAW_DIR / 'ratings_small.csv'
    ratings_path = RAW_DIR / 'ratings.csv'
    path = ratings_path_small if ratings_path_small.exists() else ratings_path
    if not path.exists():
        raise FileNotFoundError('Place ratings_small.csv or ratings.csv in data/raw/')
    ratings = pd.read_csv(path)
    keep = ['userId','movieId','rating','timestamp']
    ratings = ratings[keep].copy()
    ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce').astype('Int64')
    ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce').astype('Int64')
    ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
    ratings = ratings.dropna(subset=['userId','movieId','rating']).reset_index(drop=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(PROC_DIR / 'ratings.parquet', index=False)
    return ratings

def main():
    movies = load_and_clean_movies()
    ratings = load_and_clean_ratings()
    print(f'Saved movies: {len(movies)}; ratings: {len(ratings)} rows to {PROC_DIR}')

if __name__ == '__main__':
    main()
