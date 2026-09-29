"""The two-step recognition used by the Identify page.

Step 1, brand: the front photo goes to the brand model (car_brand_predictor.ipynb).
Step 2, model: every photo goes to the car-model classifiers of the likely brands (car_model_predictor.ipynb,
or the fine-tuned ones from car_model_finetune.ipynb when they exist), averaged over the photos.
"""
import io
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from common import DATA_DIR, MODELS_DIR

YOLO_PATH = MODELS_DIR / "yolo11m-seg.pt"
BRAND_MODEL_PATH = MODELS_DIR / "efficientnet_b0.keras"
HEADS_DIR = MODELS_DIR / "model_heads"
FINETUNED_DIR = MODELS_DIR / "model_finetuned"

CAR_CLASSES = [2, 7]      # COCO: car, truck (SUVs and pickups are often detected as "truck")
CROP_SIZE = 300           # the training photos are 300×300
MODEL_SIZE = 224          # EfficientNetB0 takes 224×224
MIN_BRAND_PROBABILITY = 0.05
MAX_CANDIDATE_BRANDS = 3


@st.cache_resource(show_spinner="Loading the car detector (first time only)...")
def load_yolo():
    from ultralytics import YOLO
    return YOLO(str(YOLO_PATH))


@st.cache_resource(show_spinner="Loading the brand model (first time only)...")
def load_brand_model():
    import keras
    return keras.models.load_model(BRAND_MODEL_PATH)


@st.cache_resource
def load_feature_extractor():
    """The brand model's EfficientNet + average pooling: the features the car-model classifiers were trained on."""
    import keras
    from keras import layers
    base = load_brand_model().get_layer("efficientnetb0")
    inputs = keras.Input(shape=(MODEL_SIZE, MODEL_SIZE, 3))
    return keras.Model(inputs, layers.GlobalAveragePooling2D()(base(inputs, training=False)))


def is_finetuned(brand):
    return (FINETUNED_DIR / f"{brand}_done.txt").exists() and (FINETUNED_DIR / f"{brand}.keras").exists()


@st.cache_resource(max_entries=12)
def load_car_model_classifier(brand, finetuned):
    """(model, classes): the fine-tuned model (takes images) or the classifier on features."""
    import keras
    classes = json.loads((HEADS_DIR / f"{brand}_classes.json").read_text())
    path = FINETUNED_DIR / f"{brand}.keras" if finetuned else HEADS_DIR / f"{brand}.keras"
    return keras.models.load_model(path), classes


@st.cache_data
def brand_names():
    """Brands in label order: alphabetical, exactly as during training."""
    return sorted(pd.read_csv(DATA_DIR / "index.csv", usecols=["brand"])["brand"].unique())


def car_models_available():
    return all((HEADS_DIR / f"{b}.keras").exists() for b in brand_names())


@st.cache_data(max_entries=64, show_spinner=False)
def crop_car(photo_bytes):
    """Cut out the biggest car, white background, centred on a square: the style of the training photos.
    Returns (photo, crop, detection confidence); crop and confidence are None when no car is found."""
    photo = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
    pixels = np.array(photo)
    result = load_yolo()(pixels, classes=CAR_CLASSES, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return photo, None, None
    masks = result.masks.data.cpu().numpy() > 0.5
    biggest = masks.sum(axis=(1, 2)).argmax()
    white = np.where(masks[biggest][..., None], pixels, 255).astype(np.uint8)
    # crop to the detection box: the mask can have stray pixels far from the car
    x1, y1, x2, y2 = result.boxes.xyxy[biggest].cpu().numpy().round().astype(int)
    car = white[max(y1, 0):y2, max(x1, 0):x2]
    side = max(car.shape[:2])
    square = np.full((side, side, 3), 255, np.uint8)
    top, left = (side - car.shape[0]) // 2, (side - car.shape[1]) // 2
    square[top:top + car.shape[0], left:left + car.shape[1]] = car
    crop = Image.fromarray(square).resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)
    return photo, crop, float(result.boxes.conf[biggest])


def _inputs(crops):
    import tensorflow as tf
    return tf.image.resize(np.stack([np.array(c, dtype="float32") for c in crops]), (MODEL_SIZE, MODEL_SIZE))


def predict_brand(front_crop):
    """[(brand, probability)], most likely first."""
    import tensorflow as tf
    probabilities = tf.nn.softmax(load_brand_model()(_inputs([front_crop]), training=False)[0]).numpy()
    names = brand_names()
    return [(names[i], float(probabilities[i])) for i in probabilities.argsort()[::-1]]


def predict_car_model(brand_ranking, crops):
    """Run the car-model classifiers of the likely brands on every photo.

    Returns (ranking, per_photo, finetuned):
    - ranking: [(brand, model, score)] with score = P(brand) × P(model | brand), best first
    - per_photo: {brand: [(model, probability)] one per photo}
    - finetuned: {brand: True if its fine-tuned classifier was used}
    """
    import tensorflow as tf
    candidates = [(b, p) for b, p in brand_ranking[:MAX_CANDIDATE_BRANDS] if p >= MIN_BRAND_PROBABILITY] or brand_ranking[:1]
    x = _inputs(crops)
    features = None
    ranking, per_photo, finetuned = [], {}, {}
    for brand, p_brand in candidates:
        finetuned[brand] = is_finetuned(brand)
        classifier, classes = load_car_model_classifier(brand, finetuned[brand])
        if finetuned[brand]:
            scores = classifier(x, training=False)
        else:
            if features is None:
                features = load_feature_extractor()(x, training=False)
            scores = classifier(features, training=False)
        photo_p = tf.nn.softmax(scores).numpy()
        per_photo[brand] = [(classes[p.argmax()], float(p.max())) for p in photo_p]
        car_p = photo_p.mean(axis=0)
        ranking += [(brand, classes[i], p_brand * float(car_p[i])) for i in range(len(classes))]
    ranking.sort(key=lambda r: r[2], reverse=True)
    return ranking, per_photo, finetuned
