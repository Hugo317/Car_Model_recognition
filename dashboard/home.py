import json

import streamlit as st

from common import GREEN, MODELS_DIR, SUBTEXT, TEXT, load, pct

brand_test = load(MODELS_DIR / "test_predictions.csv", "car_brand_predictor.ipynb", usecols=["brand", "correct"])
pipeline = load(MODELS_DIR / "model_heads" / "pipeline_test.csv", "car_model_predictor.ipynb")
makes = brand_test["brand"].nunique()
models = sum(len(json.loads(p.read_text())) for p in (MODELS_DIR / "model_heads").glob("*_classes.json"))
photos = json.loads((MODELS_DIR / "features" / "features_info.json").read_text())["done"]

# ---------- hero ----------
st.markdown(
    f"<div style='max-width:50rem; padding:3rem 0 1.5rem'>"
    f"<div style='font-size:3.2rem; font-weight:700; line-height:1.1; color:{TEXT}'>"
    f"Don't know what you're driving?</div>"
    f"<p style='font-size:1.25rem; line-height:1.6; color:{SUBTEXT}; margin-top:1.5rem'>"
    f"There are <b style='color:{TEXT}'>{makes} makes</b> and <b style='color:{TEXT}'>{models} models</b> "
    f"in this garage, and plenty of them look alike from across the street. Is that an A4 or an A5? "
    f"A 3 Series or a 4 Series?</p>"
    f"<p style='font-size:1.25rem; line-height:1.6; color:{TEXT}'>"
    f"Show me a few photos and I'll <b style='color:{GREEN}'>figure out what you're driving</b>: "
    f"the make from the front, the model from every angle.</p></div>",
    unsafe_allow_html=True,
)
if st.button("Identify a car →", type="primary"):
    st.switch_page("identify.py")
st.caption("Trained on photos from UK car adverts: the DVM-CAR dataset (Huang et al., 2022).")

# ---------- numbers ----------
st.write("")
cards = st.columns(3)
cards[0].metric("Car photos it learned from", f"{photos:,}", border=True,
                help="Photos of the 33 makes from every angle, after removing broken photos and duplicates: "
                     "see the Data page.")
cards[1].metric("Make right", pct(brand_test["correct"].mean()), border=True,
                help="Share of test cars whose make is right, from one front photo the model never saw: "
                     "see the Brand model page.")
cards[2].metric("Make and model right", pct(pipeline["both correct"].mean()), border=True,
                help=f"The whole tool on {len(pipeline):,} test cars it never saw: make from the front photo, "
                     "model from all the car's photos. See the Car models page.")

# ---------- how it works ----------
st.subheader("How it works")
STEPS = [
    ("Upload your photos", "One or more photos of the same car, from any side. Mark the one that shows the front."),
    ("Get the make", f"The front photo goes to a model trained on 60,000 car fronts. It picks one of {makes} makes."),
    ("Get the model", f"Every photo then goes to that make's own classifier, which knows its models from every "
                      f"angle. More photos, surer answer."),
]
for column, (number, (title, text)) in zip(st.columns(3), enumerate(STEPS, start=1)):
    with column.container(border=True):
        st.markdown(f"<div style='font-size:2rem; font-weight:700; color:{GREEN}'>{number}</div>"
                    f"<div style='font-weight:700; margin:0.25rem 0'>{title}</div>"
                    f"<div style='color:{SUBTEXT}'>{text}</div>", unsafe_allow_html=True)

st.write("")
st.divider()
st.caption("**Note:** the tool only knows the makes and models it was trained on, mostly UK cars from 2000–2021. "
           "Anything else is reported as the closest car it knows.")
