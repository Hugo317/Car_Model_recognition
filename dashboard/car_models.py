import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AMBER, BG, BLUE, GRAPHITE, GREEN, LINE, MODELS_DIR, PANEL, RED, TEXT, chart, load, panel, pct

NOTEBOOK = "car_model_predictor.ipynb"
FINETUNE_NOTEBOOK = "car_model_finetune.ipynb"
HEADS_DIR = MODELS_DIR / "model_heads"
FINETUNED_DIR = MODELS_DIR / "model_finetuned"
# DVM-CAR's predicted viewpoint, in degrees around the car
ANGLES = {0: "front (0°)", 45: "45°", 90: "side (90°)", 135: "135°", 180: "rear (180°)",
          225: "225°", 270: "side (270°)", 315: "315°"}

summary = load(HEADS_DIR / "summary.csv", NOTEBOOK)
test = load(HEADS_DIR / "test_predictions.csv", NOTEBOOK,
            usecols=["brand", "model", "angle", "advert_id", "predicted_model", "correct"])
pipeline = load(HEADS_DIR / "pipeline_test.csv", NOTEBOOK)

car_accuracy = np.average(summary["car accuracy"], weights=summary["test cars"])

st.title("Car models")
st.caption("Step 2 of the tool: once the make is known, which of its models is it? Each make has its own "
           "classifier, trained on photos from every angle. Scores are on test cars neither step ever saw.")

# ---------- headline ----------
cards = st.columns(4)
cards[0].metric("Make and model right", pct(pipeline["both correct"].mean()), border=True,
                help=f"The whole tool on {len(pipeline):,} test cars: make from the front photo, model from all "
                     f"of the car's photos ({pipeline['photos'].mean():.1f} on average).")
cards[1].metric("Error", pct(1 - pipeline["both correct"].mean()), border=True,
                help="Test cars where the make or the model is wrong.")
cards[2].metric("Model right, per car", pct(car_accuracy), border=True,
                help="Make given, all photos of a car averaged: the second step on its own.")
cards[3].metric("Model right, one photo", pct(test["correct"].mean()), border=True,
                help="A single photo from any angle, make given. Some angles say much less than others.")

with st.expander("How it works, and why it's cheap"):
    st.markdown(
        f"- **Features, once**: the brand model's EfficientNet turns each of the 1,351,819 photos into 1,280 "
        f"numbers describing it (about 90 minutes on the Mac's GPU, done once and saved).\n"
        f"- **One small classifier per make**: {len(summary)} small networks (one hidden layer) learn the models "
        f"of their make from those numbers: from a few seconds to a minute each, instead of days for 33 full networks.\n"
        f"- **Fine-tuning for the weakest makes**: makes under 90% per car get the EfficientNet itself re-trained "
        f"on their own photos (`{FINETUNE_NOTEBOOK}`).\n"
        f"- **In the tool**: the makes with at least 5% probability from the front photo (up to 3) run their "
        f"classifier on every photo; the photos are averaged, and each (make, model) scores "
        f"P(make) × P(model | make)."
    )

# ---------- per make ----------
st.subheader("Accuracy per make")
with panel("Model accuracy per make, test"):
    ordered = summary.sort_values("car accuracy")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ordered["photo accuracy"], y=ordered["brand"], orientation="h", name="one photo",
                         marker_color=GRAPHITE, hovertemplate="%{y}, one photo: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=ordered["car accuracy"], y=ordered["brand"], orientation="h", name="per car (all photos)",
                         marker_color=GREEN, customdata=ordered[["models", "test cars"]],
                         hovertemplate="%{y}, per car: %{x:.1%}<br>%{customdata[0]} models · "
                                       "%{customdata[1]:,} test cars<extra></extra>"))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=26 * len(ordered) + 80, barmode="group", bargap=0.25, bargroupgap=0.05,
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    chart(fig)
    st.caption("Makes with many look-alike models are hardest: Audi has 33 models, including A3, S3 and RS3.")

left, right = st.columns(2)
with left, panel("One photo: accuracy per angle"):
    per_angle = test.groupby("angle")["correct"].mean()
    fig = go.Figure(go.Bar(
        x=[ANGLES[a] for a in per_angle.index], y=per_angle.values,
        marker_color=[GREEN if a == per_angle.idxmax() else BLUE for a in per_angle.index],
        text=[pct(a, 0) for a in per_angle.values], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=300)
    chart(fig)
with right, panel("More photos, surer answer"):
    by_photos = pipeline.groupby(pipeline["photos"].clip(upper=6))["both correct"].agg(["mean", "size"])
    fig = go.Figure(go.Bar(
        x=[f"{n}+" if n == 6 else str(n) for n in by_photos.index], y=by_photos["mean"], marker_color=GREEN,
        text=[pct(a, 0) for a in by_photos["mean"]], textposition="outside", textfont=dict(color=TEXT),
        customdata=by_photos["size"], hovertemplate="%{x} photos: %{y:.1%} (%{customdata:,} cars)<extra></extra>",
    ))
    fig.update_xaxes(title="photos of the car")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=300)
    chart(fig)

# ---------- fine-tuning ----------
st.subheader("Fine-tuning the weakest makes")
comparison_path = FINETUNED_DIR / "comparison.csv"
logs = {p.name.removesuffix("_log.csv"): p for p in sorted(FINETUNED_DIR.glob("*_log.csv")) if p.stat().st_size > 0}
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path).sort_values("car accuracy before")
    with panel("Per car, test: before and after fine-tuning"):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comparison["car accuracy before"], y=comparison["brand"], orientation="h",
                             name="frozen features", marker_color=GRAPHITE,
                             hovertemplate="%{y}, before: %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Bar(x=comparison["car accuracy after"], y=comparison["brand"], orientation="h",
                             name="fine-tuned", marker_color=GREEN,
                             hovertemplate="%{y}, after: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=LINE)
        fig.update_layout(height=60 * len(comparison) + 80, barmode="group",
                          legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
        chart(fig)
elif logs:
    st.info(f"Fine-tuning is still running (`{FINETUNE_NOTEBOOK}`). The before/after test scores appear here "
            f"when it's done; meanwhile, validation accuracy per epoch:", icon=":material/hourglass_top:")
    with panel("Validation accuracy per photo, per epoch"):
        fig = go.Figure()
        for (brand, path), colour in zip(logs.items(), [GREEN, BLUE, AMBER, RED, GRAPHITE, TEXT]):
            log = pd.read_csv(path)
            fig.add_trace(go.Scatter(x=log["epoch"] + 1, y=log["val_accuracy"], name=brand, mode="lines+markers",
                                     line=dict(color=colour, width=2)))
        fig.update_xaxes(title="epoch", dtick=1)
        fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor=LINE)
        fig.update_layout(height=300, legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
        chart(fig)
else:
    st.caption(f"Not run yet: `{FINETUNE_NOTEBOOK}`.")

# ---------- confusion ----------
st.subheader("Which models get confused")
brands = summary.sort_values("car accuracy")["brand"].tolist()
brand = st.selectbox("Make", brands, index=0, help="Sorted from the least to the most accurate.")
part = test[test["brand"] == brand]
models = sorted(part["model"].unique())
left, right = st.columns([3, 2])
with left, panel(f"{brand}: true model vs predicted model, one photo", key="confusion"):
    matrix = pd.crosstab(part["model"], part["predicted_model"], normalize="index").reindex(
        index=models, columns=models, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=models, y=models, zmin=0, zmax=1, xgap=1, ygap=1,
        colorscale=[[0, PANEL], [0.15, "#123d24"], [1, GREEN]], colorbar=dict(tickformat=".0%", outlinewidth=0),
        hovertemplate="true %{y} → predicted %{x}: %{z:.0%} of its photos<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="true model", tickmode="array", tickvals=models, showgrid=False)
    fig.update_xaxes(title="predicted model", tickangle=-60, tickmode="array", tickvals=models)
    fig.update_layout(height=max(360, 22 * len(models) + 160))
    chart(fig)
with right, panel(f"{brand}: most common mix-ups", key="mixups"):
    wrong = part[~part["correct"]]
    top = wrong.groupby(["model", "predicted_model"]).size().sort_values(ascending=False).head(12)
    share = top / part.groupby("model").size().reindex(top.index.get_level_values(0)).values
    table = pd.DataFrame({"True model": top.index.get_level_values(0), "Predicted as": top.index.get_level_values(1),
                          "Photos": top.values, "Share of its photos": share.values})
    st.dataframe(table, hide_index=True,
                 column_config={"Share of its photos": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    row = summary.set_index("brand").loc[brand]
    st.caption(f"{brand}: {int(row['models'])} models, {pct(row['photo accuracy'])} per photo, "
               f"{pct(row['car accuracy'])} per car (test, frozen features).")
