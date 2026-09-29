import hashlib

import plotly.graph_objects as go
import streamlit as st

import recognition as rec
from common import GRAPHITE, GREEN, LINE, RED, SUBTEXT, TEXT, chart, panel, pct

UNSURE_BELOW = 0.5   # top probability under this → "not sure"
PER_ROW = 6

st.title("Identify a car")
st.caption("Upload one or more photos of the same car, from any angle, and mark the one that shows the front. "
           "The front gives the make; every photo helps name the model.")

uploads = st.file_uploader("Photos of one car", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
if not uploads:
    st.info("Best results: one car per photo, clearly visible and not cut off, with at least one photo straight "
            "from the front. Photos from the sides and the back make the model surer.", icon=":material/photo_camera:")
    st.stop()

with st.spinner("Finding the car in each photo..."):
    processed = [rec.crop_car(u.getvalue()) for u in uploads]
names = [f"Photo {i + 1}" for i in range(len(uploads))]


def grid(items):
    """Show (caption, image) pairs, PER_ROW per row."""
    for start in range(0, len(items), PER_ROW):
        for column, (caption, image) in zip(st.columns(PER_ROW), items[start:start + PER_ROW]):
            column.image(image, caption=caption)


with panel("Your photos"):
    grid([(name + ("" if crop is not None else " · no car found"), photo)
          for name, (photo, crop, _) in zip(names, processed)])

front = st.segmented_control(
    "Which photo shows the front of the car?", options=range(len(uploads)), format_func=lambda i: names[i],
    default=0 if len(uploads) == 1 else None,
    key="front_" + hashlib.md5("".join(f"{u.name}{u.size}" for u in uploads).encode()).hexdigest(),
)
if front is None:
    st.info("Mark the front photo above to start: the make is read from the front of the car.",
            icon=":material/touch_app:")
    st.stop()

front_crop = processed[front][1]
if front_crop is None:
    st.error(f"No car found in {names[front]}. The front photo is needed for the make: try one where the car is "
             f"larger and fully in the frame.", icon=":material/error:")
    st.stop()

usable = [i for i, (_, crop, _) in enumerate(processed) if crop is not None]
with st.spinner("Naming the make and the model..."):
    brand_ranking = rec.predict_brand(front_crop)
    has_models = rec.car_models_available()
    if has_models:
        model_ranking, per_photo, finetuned = rec.predict_car_model(brand_ranking, [processed[i][1] for i in usable])

p_brands = dict(brand_ranking)

# ---------- result ----------
if has_models:
    brand, model, score = model_ranking[0]
    p_model = score / p_brands[brand]
    sure = p_brands[brand] >= UNSURE_BELOW and p_model >= UNSURE_BELOW
    cards = st.columns(4)
    cards[0].metric("Make", brand, delta=f"{pct(p_brands[brand], 0)} from the front", delta_arrow="off",
                    delta_color="normal" if p_brands[brand] >= UNSURE_BELOW else "inverse", border=True)
    cards[1].metric("Model", model, delta=f"{pct(p_model, 0)} given the make", delta_arrow="off",
                    delta_color="normal" if p_model >= UNSURE_BELOW else "inverse", border=True)
    cards[2].metric("Overall", pct(score, 0), border=True, help="P(make) × P(model | make).")
    cards[3].metric("Photos used", f"{len(usable)} of {len(uploads)}", border=True,
                    help="Photos where no car was found are skipped.")
    colour = GREEN if sure else RED
    verdict = (f"This looks like a <b style='color:{colour}'>{brand} {model}</b>."
               if sure else f"Not sure. Best guess: <b style='color:{colour}'>{brand} {model}</b>, but the "
                            f"{'make' if p_brands[brand] < UNSURE_BELOW else 'model'} is uncertain. "
                            f"More photos from other sides usually help.")
else:
    brand, p_brand = brand_ranking[0]
    sure = p_brand >= UNSURE_BELOW
    colour = GREEN if sure else RED
    st.columns(4)[0].metric("Make", brand, delta=pct(p_brand, 0), delta_arrow="off", border=True)
    verdict = (f"The make is <b style='color:{colour}'>{brand}</b>. Car models aren't available yet: "
               f"run <code>car_model_predictor.ipynb</code>.")
st.markdown(f"<div style='border-left:4px solid {colour}; padding:0.75rem 1rem; margin:0.5rem 0 1.5rem; "
            f"font-size:1.15rem; color:{TEXT}'>{verdict}</div>", unsafe_allow_html=True)

# ---------- what the models see ----------
with panel("What the models see", key="crops"):
    st.caption("Each car cut out of its background, like the training photos"
               + (f", with {brand}'s best model for that photo alone." if has_models else "."))
    items = []
    for i, name in enumerate(names):
        crop = processed[i][1]
        if crop is None:
            continue
        caption = name + (" · front" if i == front else "")
        if has_models and brand in per_photo:
            photo_model, p = per_photo[brand][usable.index(i)]
            caption += f" · {photo_model} {pct(p, 0)}"
        items.append((caption, crop))
    grid(items)


def bars(items, title, key):
    labels, values = [label for label, _ in items], [p for _, p in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=[GREEN] + [GRAPHITE] * (len(items) - 1),
        text=[pct(v) for v in values], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=250, bargap=0.35)
    with panel(title, key=key):
        chart(fig)


left, right = st.columns(2)
with left:
    if has_models:
        bars([(f"{b} {m}", s) for b, m, s in model_ranking[:5]], "Top 5 models", "top_models")
        used = [f"{b}: {'fine-tuned' if f else 'standard'} classifier" for b, f in finetuned.items()]
        st.caption("Makes checked: " + ", ".join(used) + ".")
with right:
    bars(brand_ranking[:5], f"Top 5 makes, from {names[front].lower()}", "top_brands")

st.caption(f"<span style='color:{SUBTEXT}'>The tool knows 33 makes and 421 models, mostly UK cars from 2000–2021; "
           f"anything else is reported as the closest car it knows. Only the biggest car in each photo is used.</span>",
           unsafe_allow_html=True)
