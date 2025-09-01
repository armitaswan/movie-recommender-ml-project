import numpy as np
import pandas as pd

# IMDb-style weighted rating (WR)
# WR = (v/(v+m))*R + (m/(v+m))*C
def imdb_weighted_rating(df: pd.DataFrame, m_threshold: float = 0.80) -> pd.DataFrame:
    C = df['vote_average'].fillna(0).mean()
    m = np.nanpercentile(df['vote_count'].fillna(0), 100 * m_threshold)
    v = df['vote_count'].fillna(0).astype(float)
    R = df['vote_average'].fillna(0).astype(float)
    wr = (v / (v + m)) * R + (m / (v + m)) * C
    out = df.copy()
    out['wr_score'] = wr
    return out.sort_values('wr_score', ascending=False)
