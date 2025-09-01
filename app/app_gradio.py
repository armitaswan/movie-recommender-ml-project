import gradio as gr
import pandas as pd
from src.utils import load_pickle
from src.config import MODELS_DIR, PROC_DIR, TOP_K

CONTENT = MODELS_DIR / 'content_indexer.pkl'
HYBRID = MODELS_DIR / 'hybrid.pkl'

movies = pd.read_parquet(PROC_DIR / 'movies.parquet')
movie_map = {str(r['id']): r['title'] for _, r in movies[['id','title']].iterrows()}

cb = load_pickle(CONTENT)
try:
    hyb = load_pickle(HYBRID)
except Exception:
    hyb = None

def recommend_by_titles(liked_titles, k=TOP_K):
    title2id = {v:k for k,v in movie_map.items()}
    seen_ids = [title2id.get(t) for t in liked_titles if title2id.get(t) is not None]
    if not seen_ids:
        return []
    if hyb is None:
        uvec = cb.build_user(seen_ids)
        recs = cb.recommend(uvec, k=k, exclude=seen_ids)
        return [(title, float(score)) for _, title, score in recs]
    recs = hyb.recommend(user_id=0, user_profile_ids=seen_ids, k=k, exclude=seen_ids)
    out = [(movie_map.get(str(mid), str(mid)), float(score)) for mid, score in recs]
    return out

with gr.Blocks() as demo:
    gr.Markdown('# 🎬 Movie Recommender Demo')
    liked = gr.CheckboxGroup(choices=sorted(list(movie_map.values()))[:200], label='Pick a few movies you like')
    k = gr.Slider(5, 20, value=10, step=1, label='Top-K')
    btn = gr.Button('Recommend')
    out = gr.Dataframe(headers=['Title','Score'], datatype=['str','number'])
    btn.click(fn=lambda titles, kk: recommend_by_titles(titles, int(kk)), inputs=[liked, k], outputs=out)

if __name__ == '__main__':
    demo.launch()
