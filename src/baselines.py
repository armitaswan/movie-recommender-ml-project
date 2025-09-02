import pandas as pd
import numpy as np

def imdb_wr(df: pd.DataFrame, m_quantile: float = 0.80) -> pd.DataFrame:
    '''Compute IMDb-style weighted rating:
        WR = (v/(v+m))*R + (m/(v+m))*C
    where R is item mean rating, v is vote_count, C is global mean, and m is min votes threshold.
    Returns (table, globals_dict).'''
    stats = df.groupby('movieId')['rating'].agg(['mean','count']).rename(columns={'mean':'R','count':'v'})
    C = df['rating'].mean()
    m = stats['v'].quantile(m_quantile)
    stats['WR'] = (stats['v']/(stats['v']+m))*stats['R'] + (m/(stats['v']+m))*C
    stats = stats.sort_values('WR', ascending=False).reset_index()
    stats['rank'] = np.arange(1, len(stats)+1)
    return stats, dict(C=C, m=m)

def per_genre_popularity(ratings: pd.DataFrame, movies_joined: pd.DataFrame, m_quantile: float = 0.80) -> pd.DataFrame:
    df = ratings.merge(movies_joined[['movieId','genres','title']], on='movieId', how='left')
    df = df.explode('genres')
    by_g = df.groupby(['genres','movieId'])['rating'].agg(['mean','count']).rename(columns={'mean':'R','count':'v'}).reset_index()
    C_by_g = by_g.groupby('genres')['R'].transform('mean')
    m_by_g = by_g.groupby('genres')['v'].transform(lambda s: s.quantile(m_quantile))
    by_g['WR'] = (by_g['v']/(by_g['v']+m_by_g))*by_g['R'] + (m_by_g/(by_g['v']+m_by_g))*C_by_g
    top = by_g.sort_values(['genres','WR'], ascending=[True, False])
    return top
