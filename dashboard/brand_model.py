import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BG, BLUE, GRAPHITE, GREEN, LINE, MODELS_DIR, PANEL, RED, TEXT, chart, load, panel, pct

NOTEBOOK = "car_brand_predictor.ipynb"

test = load(MODELS_DIR / "test_predictions.csv", NOTEBOOK, usecols=["brand", "predicted_brand", "correct"])
small_cnn = load(MODELS_DIR / "small_cnn_history.csv", NOTEBOOK)
effnet = load(MODELS_DIR / "efficientnet_b0_history.csv", NOTEBOOK)

per_brand = test.groupby("brand")["correct"].mean().sort_values()
accuracy, balanced = test["correct"].mean(), per_brand.mean()

st.title("Brand model")
st.caption("Step 1 of the tool: which make is this, from one photo of the front? Two models were trained on the "
           "60,126 front photos; the better one was scored once on test photos it never saw.")

# ---------- test scores ----------
cards = st.columns(4)
cards[0].metric("Test accuracy", pct(accuracy), border=True,
                help=f"Share of the {len(test):,} test photos whose make is right.")
cards[1].metric("Error", pct(1 - accuracy), border=True,
                help=f"{(~test['correct']).sum()} wrong out of {len(test):,} test photos.")
cards[2].metric("Balanced accuracy", pct(balanced), border=True,
                help="Accuracy of each make, averaged: Suzuki counts as much as Audi, even with 44× fewer photos.")
cards[3].metric("Weakest make", f"{per_brand.index[0]}", delta=pct(per_brand.iloc[0]),
                delta_color="off", delta_arrow="off", border=True)

# ---------- models compared ----------
st.subheader("Two models compared")
st.caption("Both were chosen on the validation set; only the winner was scored on the test set.")
left, right = st.columns([2, 3])
with left, panel("Best validation accuracy"):
    models = pd.DataFrame({
        "model": ["Small CNN, from scratch", "EfficientNetB0, pretrained"],
        "accuracy": [small_cnn["val_accuracy"].max(), effnet["val_accuracy"].max()],
        "epochs": [len(small_cnn), len(effnet)],
    })
    fig = go.Figure(go.Bar(
        x=models["accuracy"], y=models["model"], orientation="h", marker_color=[GRAPHITE, GREEN],
        text=[pct(a) for a in models["accuracy"]], textposition="inside", insidetextanchor="end",
        textfont=dict(color=BG, size=14), customdata=models["epochs"],
        hovertemplate="%{y}: %{x:.1%} after %{customdata} epochs<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0.9, 1], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=200, bargap=0.35)
    chart(fig)
    st.caption("The axis starts at 90%: both are good, the pretrained one makes about half as many mistakes.")
with right, panel("Validation accuracy while training"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=small_cnn["val_accuracy"], name="Small CNN", line=dict(color=GRAPHITE, width=2)))
    fig.add_trace(go.Scatter(y=effnet["val_accuracy"], name="EfficientNetB0", line=dict(color=GREEN, width=3)))
    fig.add_vline(x=2.5, line=dict(color=BLUE, dash="dot"), annotation_text="EfficientNet: fine-tuning starts",
                  annotation_position="bottom right", annotation_font_color=BLUE)
    fig.update_yaxes(range=[0.7, 1.005], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_xaxes(title="epoch")
    fig.update_layout(height=280, margin=dict(t=40), legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
                      hovermode="x unified")
    chart(fig)

with st.expander("What the two models are"):
    st.markdown(
        "- **Small CNN**: 5 convolution blocks (1.6 million weights) trained from random weights on 128×128 photos. "
        "It learned everything from our 48,000 training photos.\n"
        "- **EfficientNetB0**: a network pretrained on ImageNet (1.3 million everyday photos), at 224×224. We replaced "
        "its last layer with our 33 makes, trained that layer for 3 epochs, then fine-tuned the network's top blocks "
        "(5 to 7) with a 10× lower learning rate.\n"
        "- **Both**: random shifts, zoom, rotation, contrast and brightness while training (never a mirror image, "
        "which would flip the badges); class weights so rare makes count as much as common ones; early stopping on "
        "the validation loss."
    )

# ---------- per make ----------
st.subheader("Accuracy per make")
with panel("Test accuracy per make"):
    fig = go.Figure(go.Bar(
        x=per_brand.values, y=per_brand.index, orientation="h",
        marker_color=[RED if a < 0.97 else GREEN for a in per_brand.values],
        text=[pct(a) for a in per_brand.values], textposition="outside", textfont=dict(color=TEXT),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_xaxes(range=[0.9, 1.02], tickformat=".0%", showgrid=True, gridcolor=LINE)
    fig.update_layout(height=20 * len(per_brand) + 60, bargap=0.25)
    chart(fig)
    st.caption("Red: under 97%. The axis starts at 90%.")

# ---------- confusion ----------
st.subheader("Which makes get confused")
mistakes = test[~test["correct"]]
left, right = st.columns([3, 2])
with left, panel("Mistakes: true make vs predicted make"):
    brands = sorted(test["brand"].unique())
    counts = pd.crosstab(mistakes["brand"], mistakes["predicted_brand"]).reindex(index=brands, columns=brands, fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=counts.replace(0, np.nan).values, x=brands, y=brands, colorscale=[[0, "#3a1d16"], [1, RED]],
        hovertemplate="true %{y} → predicted %{x}: %{z} photos<extra></extra>", showscale=False, xgap=1, ygap=1,
        hoverongaps=False,
    ))
    fig.update_yaxes(autorange="reversed", title="true make", tickmode="array", tickvals=brands, showgrid=False)
    fig.update_xaxes(title="predicted make", tickangle=-60, tickmode="array", tickvals=brands)
    fig.update_layout(height=640, plot_bgcolor=PANEL)
    chart(fig)
with right, panel("Most common mix-ups"):
    top = mistakes.groupby(["brand", "predicted_brand"]).size().sort_values(ascending=False).head(12).reset_index()
    top.columns = ["True make", "Predicted as", "Photos"]
    st.dataframe(top, hide_index=True)
    st.caption(f"Only {len(mistakes)} of {len(test):,} test photos are wrong, and no pair of makes is confused "
               f"more than {top['Photos'].max()} times.")
