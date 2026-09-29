import plotly.graph_objects as go
import streamlit as st

from common import BLUE, DATA_DIR, GREEN, LINE, MODELS_DIR, RED, TEXT, chart, load, panel

# DVM-CAR's predicted viewpoint, in degrees around the car
ANGLES = {0: "front (0°)", 45: "45°", 90: "side (90°)", 135: "135°", 180: "rear (180°)",
          225: "225°", 270: "side (270°)", 315: "315°"}

# counts printed by the notebooks while cleaning (section 1 of each)
FRONTS_FUNNEL = [  # car_brand_predictor.ipynb
    ("In DVM-CAR", 61_827),
    ("Under 2 makes", -125),
    ("Duplicates", -500),
    ("Rare makes", -1_076),
]
ALL_ANGLES_FUNNEL = [  # car_model_predictor.ipynb
    ("In DVM-CAR", 1_451_784),
    ("Filtered", -78_460),
    ("2 labels", -4_795),
    ("Duplicates", -16_710),
]


@st.cache_data
def photo_counts(modified):
    photos = load(MODELS_DIR / "features" / "photo_index.csv", "car_model_predictor.ipynb",
                  usecols=["path", "brand", "model", "angle", "advert_id", "split"])
    per_brand = photos.groupby("brand").agg(photos=("path", "size"), models=("model", "nunique")).sort_values("photos")
    per_angle = photos["angle"].value_counts().sort_index()
    per_split = photos["split"].value_counts()
    full_adverts = photos.groupby("advert_id")["angle"].nunique()
    example = (photos[photos["advert_id"] == full_adverts[full_adverts == 8].index[0]]
               .sort_values("angle").groupby("angle").head(1))  # one photo per angle
    return per_brand, per_angle, per_split, example


def funnel(steps, title):
    labels, values, measures = [s[0] for s in steps] + ["Kept"], [s[1] for s in steps] + [0], \
        ["absolute"] + ["relative"] * (len(steps) - 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures, text=[f"{v:+,}" if i else f"{v:,}" for i, v in enumerate(values[:-1])]
        + [f"{sum(values):,}"], textposition="outside", textfont=dict(color=TEXT),
        increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED), totals=dict(marker_color=GREEN),
        connector=dict(line=dict(color=LINE)),
    ))
    fig.update_yaxes(showgrid=True, gridcolor=LINE, range=[0, steps[0][1] * 1.15])
    fig.update_layout(height=360, showlegend=False)
    with panel(title):
        chart(fig)


st.title("Data")
st.caption("Every photo comes from DVM-CAR, a public dataset of 1.45 million photos from about 250,000 UK car "
           "adverts (2000–2021). Each photo's make, model, year and angle are in its file name and in "
           "`Image_table.csv`.")

path = MODELS_DIR / "features" / "photo_index.csv"
per_brand, per_angle, per_split, example = photo_counts(path.stat().st_mtime if path.exists() else 0)

st.subheader("Two sets of photos")
left, right = st.columns(2)
with left:
    st.markdown("**Fronts, for the make.** The dataset's 61,827 photos confirmed to show the front of the car. "
                "Abarth is merged into Fiat and DS into Citroën, which share their fronts.")
    funnel(FRONTS_FUNNEL, "Front photos → 60,126 kept, 33 makes")
    st.caption("Removed: the same photo filed under two makes, duplicate copies, and 28 makes with under "
               "200 photos.")
with right:
    st.markdown("**Every angle, for the model.** All photos of the same 33 makes, from 8 angles, for car models "
                "with at least 300 photos (421 models, 98% of the photos).")
    funnel(ALL_ANGLES_FUNNEL, "All photos → 1,351,819 kept, 421 models")
    st.caption("Filtered: photos that failed the dataset's quality check, makes outside the 33, and "
               "models with under 300 photos. Then photos with two different labels, and duplicate copies.")

with st.expander("How the photos were cleaned"):
    st.markdown(
        "- **Unreadable photos**: every photo was fully decoded; none were broken.\n"
        "- **Duplicates**: adverts often reuse the same photo. Exact copies were found with an MD5 fingerprint "
        "and kept once, so no photo can sit in both the training and the test set.\n"
        "- **Conflicting labels**: dealers reuse stock photos across adverts. One Kia Picanto photo appears 37 "
        "times, filed under Audi, Citroën, Fiat, Kia and Toyota. A photo filed under two labels is dropped.\n"
        "- **Quality check**: the dataset authors flagged 19,809 photos that don't clearly show a car; they're dropped.\n"
        "- **Split by advert**: all photos of one car go to the same set (80% training, 10% validation, 10% test), "
        "so the model is always tested on cars it has never seen."
    )

st.subheader("One car, eight angles")
st.caption(f"Every advert is photographed from up to 8 angles. Here: a {example['brand'].iloc[0]} "
           f"{example['model'].iloc[0]}, as the models see it (background removed, 300×300).")
for column, (_, row) in zip(st.columns(8), example.iterrows()):
    column.image(str(DATA_DIR / "resized_DVM" / row["path"]), caption=ANGLES[row["angle"]])

st.subheader("Photos per make and per angle")
left, right = st.columns([3, 2])
with left, panel("Photos per make, all angles"):
    fig = go.Figure(go.Bar(
        x=per_brand["photos"], y=per_brand.index, orientation="h", marker_color=GREEN,
        customdata=per_brand["models"],
        hovertemplate="<b>%{y}</b><br>%{x:,} photos · %{customdata} models<extra></extra>",
    ))
    fig.update_xaxes(showgrid=True, gridcolor=LINE, title="photos")
    fig.update_layout(height=22 * len(per_brand) + 60, bargap=0.25)
    chart(fig)
with right:
    with panel("Photos per angle"):
        fig = go.Figure(go.Bar(
            x=[ANGLES[a] for a in per_angle.index], y=per_angle.values,
            marker_color=[GREEN if a == 0 else BLUE for a in per_angle.index],
            hovertemplate="%{x}: %{y:,} photos<extra></extra>",
        ))
        fig.update_yaxes(showgrid=True, gridcolor=LINE)
        fig.update_layout(height=300)
        chart(fig)
    st.caption(f"Ford has {per_brand['photos'].max() / per_brand['photos'].min():.0f}× more photos than "
               f"{per_brand.index[0]}: class weights make every make and model count equally in training.")

st.subheader("Split by advert")
cards = st.columns(3)
for card, split, label, share in zip(cards, ["train", "val", "test"], ["Training photos", "Validation photos", "Test photos"],
                                     ["learns from them", "picks the best epoch", "scored once, at the end"]):
    card.metric(label, f"{per_split[split]:,}", delta=share, delta_color="off", delta_arrow="off", border=True)
