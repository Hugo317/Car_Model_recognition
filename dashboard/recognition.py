"""The two-step recognition used by the Identify page.

Step 1, brand: the front photo goes to the brand model (car_brand_predictor.ipynb).
Step 2, model: every photo goes to the car-model classifiers of the likely brands (car_model_predictor.ipynb,
or the fine-tuned ones from car_model_finetune.ipynb when they exist), averaged over the photos.

The models run on ONNX Runtime (no TensorFlow or PyTorch), so the app fits a small free host. The weights come from
`models_onnx/` when it exists, otherwise from the Hugging Face Hub; scripts/convert_to_onnx.py makes them.
"""
import io
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import detector
from common import MODELS_DIR

WEIGHTS_REPO = "Hugomnc/car-spotter-models"
LOCAL_WEIGHTS = Path(__file__).resolve().parent.parent / "models_onnx"
HEADS_DIR = MODELS_DIR / "model_heads"
FINETUNED_DIR = MODELS_DIR / "model_finetuned"

CROP_SIZE = 300           # the training photos are 300×300
MODEL_SIZE = 224          # EfficientNetB0 takes 224×224
MIN_BRAND_PROBABILITY = 0.05
MAX_CANDIDATE_BRANDS = 3


def weight_path(name):
    local = LOCAL_WEIGHTS / name
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(WEIGHTS_REPO, name)


def _session(name):
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False   # keep memory flat on a small host
    options.enable_mem_pattern = False
    return ort.InferenceSession(weight_path(name), options, providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner="Loading the car detector (first time only)...")
def load_yolo():
    return _session("yolo11m-seg.onnx")


@st.cache_resource(show_spinner="Loading the brand model (first time only)...")
def load_brand_model():
    """One pass gives the 1,280 features (what the car-model classifiers use) and the 33 brand scores."""
    return _session("brand.onnx")


@st.cache_resource(show_spinner="Loading the car-model classifiers (first time only)...")
def load_heads():
    with np.load(weight_path("heads.npz")) as z:
        keys = list(z.keys())
        return {b: [z[f"{b}|{i}"] for i in range(4)] for b in sorted({k.split("|")[0] for k in keys})}


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@st.cache_data
def is_finetuned(brand):
    """A fine-tuned classifier exists for this brand (finished in the notebook, and its ONNX weights can be fetched)."""
    if not (FINETUNED_DIR / f"{brand}_done.txt").exists():
        return False
    try:
        weight_path(f"finetuned/{brand}.onnx")
        return True
    except Exception:
        return False


@st.cache_resource(max_entries=6)
def load_finetuned(brand):
    return _session(f"finetuned/{brand}.onnx")


@st.cache_data
def brand_names():
    """Brands in label order: alphabetical, exactly as during training."""
    return sorted(p.name.removesuffix("_classes.json") for p in HEADS_DIR.glob("*_classes.json"))


def car_models_available():
    return len(brand_names()) > 0


@st.cache_data(max_entries=16, show_spinner=False)
def crop_car(photo_bytes):
    """Cut out the biggest car, white background, centred on a square: the style of the training photos.
    Returns (photo, crop, detection confidence); crop and confidence are None when no car is found."""
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
    pixels = np.array(photo)
    found = detector.detect(load_yolo(), pixels)
    photo.thumbnail((1000, 1000))   # only shown on the page; the full-size photo isn't kept in the cache
    if found is None:
        return photo, None, None
    mask, (x1, y1, x2, y2), confidence = found
    white = np.where(mask[..., None], pixels, 255).astype(np.uint8)
    # crop to the detection box: the mask can have stray pixels far from the car
    car = white[max(y1, 0):y2, max(x1, 0):x2]
    side = max(car.shape[:2])
    square = np.full((side, side, 3), 255, np.uint8)
    top, left = (side - car.shape[0]) // 2, (side - car.shape[1]) // 2
    square[top:top + car.shape[0], left:left + car.shape[1]] = car
    crop = Image.fromarray(square).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    return photo, crop, confidence


def _inputs(crops):
    """Crops → a float batch at 224×224 (plain bilinear, like tf.image.resize)."""
    return np.stack([cv2.resize(np.array(c, dtype="float32"), (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR) for c in crops])


def _run(session, x):
    return session.run(None, {session.get_inputs()[0].name: x})


def _brand_pass(x):
    """(features, brand scores) for a batch."""
    outputs = _run(load_brand_model(), x)
    features = next(o for o in outputs if o.shape[1] == 1280)
    scores = next(o for o in outputs if o.shape[1] == len(brand_names()))
    return features, scores


def predict_brand(front_crop):
    """[(brand, probability)], most likely first."""
    probabilities = _softmax(_brand_pass(_inputs([front_crop]))[1][0])
    names = brand_names()
    return [(names[i], float(probabilities[i])) for i in probabilities.argsort()[::-1]]


def predict_car_model(brand_ranking, crops):
    """Run the car-model classifiers of the likely brands on every photo.

    Returns (ranking, per_photo, finetuned):
    - ranking: [(brand, model, score)] with score = P(brand) × P(model | brand), best first
    - per_photo: {brand: [(model, probability)] one per photo}
    - finetuned: {brand: True if its fine-tuned classifier was used}
    """
    candidates = [(b, p) for b, p in brand_ranking[:MAX_CANDIDATE_BRANDS] if p >= MIN_BRAND_PROBABILITY] or brand_ranking[:1]
    x = _inputs(crops)
    features = None
    heads = load_heads()
    ranking, per_photo, finetuned = [], {}, {}
    for brand, p_brand in candidates:
        classes = json.loads((HEADS_DIR / f"{brand}_classes.json").read_text())
        finetuned[brand] = is_finetuned(brand)
        if finetuned[brand]:
            scores = _run(load_finetuned(brand), x)[0]
        else:
            if features is None:
                features = _brand_pass(x)[0]
            w1, b1, w2, b2 = heads[brand]
            scores = np.maximum(features @ w1 + b1, 0) @ w2 + b2
        photo_p = _softmax(scores)
        per_photo[brand] = [(classes[p.argmax()], float(p.max())) for p in photo_p]
        car_p = photo_p.mean(axis=0)
        ranking += [(brand, classes[i], p_brand * float(car_p[i])) for i in range(len(classes))]
    ranking.sort(key=lambda r: r[2], reverse=True)
    return ranking, per_photo, finetuned
