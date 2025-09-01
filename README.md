# Movie Recommender – Course Capstone Starter

Scaffold for EDA → CB/CF → Hybrid → Evaluation → App deploy. Start on `ratings_small.csv` then scale.

## Layout
```
src/ data/ models/ app/ notebooks/ report/ scripts/
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train_baselines.py
python scripts/train_hybrid.py
python app/app_gradio.py
```
