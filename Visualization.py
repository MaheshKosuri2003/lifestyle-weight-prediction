import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Visualizations",
    page_icon="📈",
    layout="wide"
)

df = pd.read_csv("lifestyle weight tracker.csv")

st.title("📈 Data Visualizations")

column = st.selectbox(
    "Select Column",
    df.select_dtypes(include="number").columns
)

fig = px.histogram(
    df,
    x=column,
    color_discrete_sequence=["#1565C0"]
)

st.plotly_chart(fig, use_container_width=True)

st.write("---")

fig2 = px.box(
    df,
    y=column,
    color_discrete_sequence=["#26A69A"]
)

st.plotly_chart(fig2, use_container_width=True)