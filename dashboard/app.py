"""Car Spotter dashboard. Run from the project root: .venv/bin/streamlit run dashboard/app.py"""

from pathlib import Path

import streamlit as st

from common import style

HERE = Path(__file__).parent

st.set_page_config(page_title="Car Spotter", page_icon=":material/directions_car:", layout="wide")
st.logo(str(HERE / "logo.svg"), size="large")
style()

pages = [
    st.Page("home.py", title="Home", default=True),
    st.Page("data.py", title="Data"),
    st.Page("brand_model.py", title="Brand model"),
    st.Page("car_models.py", title="Car models"),
    st.Page("identify.py", title="Identify"),
]
st.navigation(pages, position="top").run()
