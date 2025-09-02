import argparse
from pathlib import Path
import pandas as pd
import joblib

from .content_based import ContentIndexer
from .collaborative import CFRecommender
from .baselines import imdb_wr

MODELS = Path(__file__).resolve().parents[1] / "models"
DATA = Path(__file__).resolve().parents[1] / "data"
ART = Path(__file__).resolve().parents[1] / "artifacts"

def main(alpha: float, k: int):
    movies = pd.read_parquet(DATA / "movies.parquet")
    links = pd.read_parquet(DATA / "links.parquet")
    ratings = pd.read_parquet(DATA / "ratings.parquet")
    joined = pd.read_parquet(DATA / "movies_joined.parquet")

    MODELS.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    wr, info = imdb_wr(ratings)
    wr.to_csv(ART / "baseline_wr.csv", index=False)
    joblib.dump(info, ART / "baseline_wr_info.joblib")

    cb = ContentIndexer()
    cb.fit(joined)
    cb.save(str(MODELS))

    cf = CFRecommender()
    cf.fit(ratings)
    cf.save(str(MODELS))

    joblib.dump({"alpha": alpha, "k": k}, MODELS / "hybrid_cfg.joblib")
    print("Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    main(alpha=args.alpha, k=args.k)
