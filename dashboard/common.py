"""Shared by every dashboard page: the palette, the Plotly chart style, panels, and loading the saved results.

The numbers come from the notebooks (car_brand_predictor, car_model_predictor, car_model_finetune), which save
their results to models/; the pages only read them.
"""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Locally the notebooks write to models/ and data/. The hosted app has neither (3.8 GB and 17 GB), so it reads the small
# copies in app_data/ (made by scripts/make_app_data.py).
APP_DATA = Path(__file__).resolve().parent.parent / "app_data"
MODELS_DIR = Path("models") if (Path("models") / "test_predictions.csv").exists() else APP_DATA / "models"
DATA_DIR = Path("data") if (Path("data") / "resized_DVM").exists() else APP_DATA / "data"

# palette: Graphite & Green (dark), same family as the Fair Car Price dashboard with green for gold
BG = "#0c0c0e"        # page
PANEL = "#161618"     # panels and cards
LINE = "#2a2a2e"      # borders, gridlines, faint marks
TEXT = "#f2efe8"      # warm white
SUBTEXT = "#a09c94"   # secondary text
GREEN = "#1db954"     # the accent: main series, buttons, logo
BLUE = "#5a8fe0"      # second series
RED = "#e05a3a"       # bad, mistakes
AMBER = "#e0a83a"     # fourth series
GRAPHITE = "#6b6b73"  # neutral series
GREY = "#4a4a50"      # reference marks
FONT = "Space Grotesk, sans-serif"

TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=TEXT, size=13),
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    colorway=[GREEN, BLUE, RED, AMBER],
    xaxis=dict(showgrid=False, zeroline=False, linecolor=GRAPHITE, tickcolor=GRAPHITE, automargin=True),
    yaxis=dict(gridcolor=LINE, zeroline=False, automargin=True),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=10, b=10),
    hoverlabel=dict(font_family=FONT),
))

STYLE = f"""<style>
[class*="st-key-panel_"] {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 1rem; padding: 1.25rem 1.5rem; }}
[data-testid="stMetric"] {{ background: {PANEL}; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"] {{ display: none; }}
[data-testid^="stBaseButton-primary"], [data-testid^="stBaseButton-primary"] p {{ color: {BG}; font-weight: 700; }}
</style>"""


def style():
    st.html(STYLE)


def panel(title, key=None):
    """A rounded panel box with the title inside: `with panel("Accuracy per brand"): ...`"""
    box = st.container(key="panel_" + re.sub(r"\W+", "_", (key or title).lower()))
    box.markdown(f"**{title}**")
    return box


def chart(fig):
    # theme=None stops Streamlit restyling the chart; the backgrounds are set on the figure itself
    fig.update_layout(template=TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL)
    st.plotly_chart(fig, theme=None)


@st.cache_data
def _read(path, modified, usecols):
    return pd.read_csv(path, usecols=usecols)


def load(path, notebook, usecols=None):
    """A saved CSV; stops the page with a hint when the notebook that writes it hasn't been run yet."""
    path = Path(path)
    if not path.exists():
        st.warning(f"`{path}` not found: run `{notebook}` first.")
        st.stop()
    return _read(path, path.stat().st_mtime, usecols)  # the timestamp re-reads the file after a re-run


def pct(x, digits=1):
    return f"{x:.{digits}%}"
