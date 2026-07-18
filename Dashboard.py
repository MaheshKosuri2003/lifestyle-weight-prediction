import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Dataset
df = pd.read_csv("lifestyle weight tracker.csv")

st.title("📊 Dashboard")

st.write("### Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.write("---")

st.subheader("Dataset Preview")

st.dataframe(df.head(10), use_container_width=True)

st.write("---")

st.subheader("Summary Statistics")

st.dataframe(df.describe(), use_container_width=True)