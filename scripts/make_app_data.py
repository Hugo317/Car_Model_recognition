"""Collect the small result files the dashboard pages read into app_data/, so the hosted app works without
models/ (3.8 GB) and data/ (17 GB). Run from the repo root after the notebooks:  python scripts/make_app_data.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd

SRC_M, SRC_D, OUT = Path("models"), Path("data"), Path("app_data")
M, D = OUT / "models", OUT / "data"
for d in (M / "model_heads", M / "model_finetuned", M / "features", D / "resized_DVM"):
    d.mkdir(parents=True, exist_ok=True)

for f in ["test_predictions.csv", "small_cnn_history.csv", "efficientnet_b0_history.csv"]:
    shutil.copy(SRC_M / f, M / f)
heads = SRC_M / "model_heads"
for f in ["summary.csv", "pipeline_test.csv", *[p.name for p in heads.glob("*_classes.json")]]:
    shutil.copy(heads / f, M / "model_heads" / f)
# the page only reads these columns
pd.read_csv(heads / "test_predictions.csv", usecols=["brand", "model", "angle", "advert_id", "predicted_model", "correct"]).to_csv(
    M / "model_heads" / "test_predictions.csv", index=False)
fine = SRC_M / "model_finetuned"
for p in list(fine.glob("*_done.txt")) + [q for q in fine.glob("*_log.csv") if q.stat().st_size] + list(fine.glob("comparison.csv")):
    shutil.copy(p, M / "model_finetuned" / p.name)
shutil.copy(SRC_M / "features" / "features_info.json", M / "features" / "features_info.json")

# photo counts for the Data page (the full photo index is 155 MB), and the one car shown from 8 angles
photos = pd.read_csv(SRC_M / "features" / "photo_index.csv", usecols=["path", "brand", "model", "angle", "advert_id", "split"])
full = photos.groupby("advert_id")["angle"].nunique()
example = photos[photos["advert_id"] == full[full == 8].index[0]].sort_values("angle").groupby("angle").head(1)
summary = {
    "per_brand": photos.groupby("brand").agg(photos=("path", "size"), models=("model", "nunique")).sort_values("photos").reset_index().to_dict("records"),
    "per_angle": {int(k): int(v) for k, v in photos["angle"].value_counts().sort_index().items()},
    "per_split": {k: int(v) for k, v in photos["split"].value_counts().items()},
    "example": example[["path", "brand", "model", "angle", "advert_id"]].to_dict("records"),
}
(M / "features" / "photo_summary.json").write_text(json.dumps(summary))
for path in example["path"]:
    target = D / "resized_DVM" / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC_D / "resized_DVM" / path, target)
print("app_data:", round(sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6, 1), "MB")
