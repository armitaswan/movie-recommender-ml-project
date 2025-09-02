# One-Day Movie Recommender — MVP

Implements the capstone spec end-to-end with a minimal but complete pipeline.

## Quickstart
```bash
conda create -y -n recsys-mvp python=3.10
conda activate recsys-mvp
pip install -r requirements.txt

# Put raw CSVs into ./data/ :
# movies_metadata.csv, credits.csv, keywords.csv, links.csv (or links_small.csv), ratings.csv (or ratings_small.csv)

python -m src.data_prep
python -m src.train --alpha 0.6 --k 10
python -m src.eval --k_list 10 20
python -m app.app
```
