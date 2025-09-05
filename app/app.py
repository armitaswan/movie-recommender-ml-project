import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.content_based import ContentIndexer
from src.collaborative import CFRecommender
from src.hybrid import HybridRecommender, HybridCfg

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DATA = ROOT / "data"

cb = ContentIndexer.load(str(MODELS))
cf = CFRecommender.load(str(MODELS))
cfg = joblib.load(MODELS / "hybrid_cfg.joblib")
hy = HybridRecommender(cb, cf, HybridCfg(alpha=cfg["alpha"]))

movies = pd.read_parquet(DATA / "movies_joined.parquet")[["movieId","title","genres","cast","poster_path"]]
ratings = pd.read_parquet(DATA / "ratings.parquet")

def _poster_url(row):
    p = row.get("poster_path")
    if isinstance(p, str) and p.strip():
        return f"https://image.tmdb.org/t/p/w185{p}"
    return None

def _explain(row):
    g = row["genres"][:3] if isinstance(row["genres"], list) else []
    c = row["cast"][:3] if isinstance(row["cast"], list) else []
    bits = []
    if g: bits.append("Genres: " + ", ".join(g))
    if c: bits.append("Cast: " + ", ".join(c))
    return " | ".join(bits) if bits else "Similar to your tastes"

def recommend(uid, fav_titles, k):
    seen = ratings[ratings["userId"] == uid]["movieId"].tolist() if uid is not None else []
    hist = ratings[ratings["userId"] == uid][["movieId","rating"]] if uid is not None else pd.DataFrame(columns=["movieId","rating"])
    uv = cb.build_user(hist, movies)

    if len(seen) == 0 and fav_titles:
        fav = movies[movies["title"].str.lower().isin([t.strip().lower() for t in fav_titles.split(",")])]
        pseudo = pd.DataFrame({"movieId": fav["movieId"], "rating": 5.0})
        uv = cb.build_user(pseudo, movies)

    items = hy.recommend(user_id=uid, user_profile_vec=uv, k=k, exclude_seen=seen)  # (mid, score, cf_z, cb_z)
    cards = []
    table = []
    for mid, score, cfz, cbz in items:
        row = movies[movies["movieId"] == mid].iloc[0].to_dict()
        url = _poster_url(row)
        title = row.get("title","(unknown)")
        why = _explain(row)
        cb_share = (cbz / (abs(cfz) + abs(cbz) + 1e-9))
        # کارت گالری
        if url:
            cards.append((url, f"{title}\nCB share: {cb_share:.2f}\n{why}"))
        else:
            cards.append((None, f"{title}\nCB share: {cb_share:.2f}\n{why}"))
        # جدول
        table.append([title, float(score), float(cfz), float(cbz), why, url or ""])
    df = pd.DataFrame(table, columns=["Title","Score","CF_z","CB_z","Why this","Poster URL"])
    return cards, df

with gr.Blocks(title="Movie Recommender (Hybrid)") as demo:
    gr.Markdown("# 🎬 Movie Recommender — Hybrid (CB + CF)")
    with gr.Row():
        uid = gr.Number(label="User ID (optional)", value=None)
        fav = gr.Textbox(label="Favorite titles (comma-separated)")
        k = gr.Slider(5, 30, value=10, step=1, label="How many?")
    btn = gr.Button("Recommend")
    gallery = gr.Gallery(label="Top-K with posters").style(grid=[5], height="auto")
    out = gr.Dataframe(headers=["Title","Score","CF_z","CB_z","Why this","Poster URL"], interactive=False)

    def _wrap(uid_val, fav_val, k_val):
        try:
            uid_i = int(uid_val) if uid_val not in (None, "") else None
        except Exception:
            uid_i = None
        return recommend(uid_i, fav_val, int(k_val))

    btn.click(_wrap, [uid, fav, k], [gallery, out])

if __name__ == "__main__":
    demo.launch()
