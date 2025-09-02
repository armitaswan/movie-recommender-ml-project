import pandas as pd
import numpy as np
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data"
OUT = RAW

SEED = 42
np.random.seed(SEED)

def _safe_json_loads(x):
    if pd.isna(x):
        return []
    try:
        obj = json.loads(x) if isinstance(x, str) else x
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def prepare_movies():
    mm_path = RAW / "movies_metadata.csv"
    cr_path = RAW / "credits.csv"
    kw_path = RAW / "keywords.csv"

    mm = pd.read_csv(mm_path, low_memory=False)
    credits = pd.read_csv(cr_path)
    keywords = pd.read_csv(kw_path)

    # Filter non-numeric ids
    non_numeric = ~mm["id"].astype(str).str.fullmatch(r"\d+")
    dropped_mm = int(non_numeric.sum())
    mm = mm[~non_numeric].copy()
    mm["id"] = mm["id"].astype(int)

    # Parse JSON-like columns to lists of names
    for col in ["genres", "production_countries", "spoken_languages"]:
        mm[col] = mm[col].apply(_safe_json_loads).apply(lambda lst: [d.get("name") for d in lst if isinstance(d, dict)])

    credits["cast"] = credits["cast"].apply(_safe_json_loads).apply(lambda lst: [d.get("name") for d in lst if isinstance(d, dict)])
    credits["crew"] = credits["crew"].apply(_safe_json_loads).apply(lambda lst: [d.get("name") for d in lst if isinstance(d, dict)])
    keywords["keywords"] = keywords["keywords"].apply(_safe_json_loads).apply(lambda lst: [d.get("name") for d in lst if isinstance(d, dict)])

    movies = mm.merge(credits, on="id", how="left").merge(keywords, on="id", how="left")

    # Basic cleaning
    for col in ["title", "overview", "tagline", "original_language"]:
        movies[col] = movies.get(col, "").fillna("")
    for col in ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]:
        if col in movies:
            movies[col] = pd.to_numeric(movies[col], errors="coerce")

    keep = ["id","imdb_id","title","overview","tagline","genres","keywords","cast","crew",
            "original_language","runtime","budget","revenue","popularity","vote_average","vote_count"]
    movies = movies[keep]

    OUT.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(OUT / "movies.parquet", index=False)
    pd.Series({"dropped_non_numeric_id_rows": dropped_mm}).to_json(OUT / "prep_stats.json", indent=2)
    return movies

def prepare_links_and_ratings():
    links_path = RAW / "links.csv"
    ratings_path = RAW / "ratings.csv"
    if not links_path.exists():
        links_path = RAW / "links_small.csv"
    if not ratings_path.exists():
        ratings_path = RAW / "ratings_small.csv"

    links = pd.read_csv(links_path)
    ratings = pd.read_csv(ratings_path)

    for col in ["movieId","imdbId","tmdbId"]:
        if col in links:
            links[col] = pd.to_numeric(links[col], errors="coerce").astype("Int64")

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")

    links.to_parquet(OUT / "links.parquet", index=False)
    ratings.to_parquet(OUT / "ratings.parquet", index=False)
    return links, ratings

def join_by_tmdbid(movies, links):
    df = links[["movieId","tmdbId","imdbId"]].copy()
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
    movies = movies.copy()
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")

    j = df.merge(movies, left_on="tmdbId", right_on="id", how="inner")
    j = j.rename(columns={"movieId":"movieId"})
    j.to_parquet(OUT / "movies_joined.parquet", index=False)
    return j

if __name__ == "__main__":
    print("Preparing movies...")
    movies = prepare_movies()
    print("Preparing links & ratings...")
    links, ratings = prepare_links_and_ratings()
    print("Joining...")
    joined = join_by_tmdbid(movies, links)
    print("Done. Saved parquet files in ./data/")
