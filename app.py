import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio
import os
import tempfile

st.set_page_config(layout="wide")
st.title("📊 Evolving Stock Return-Volume Clustering (Animated)")

# --- Sample NSE stock list (can be expanded) ---
nse_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS"
]

# --- Inputs ---
selected_ticker = st.selectbox("Select a stock", options=nse_stocks)
days = st.slider("Number of days to animate", min_value=30, max_value=180, value=60, step=10)

# --- Fetch data ---
@st.cache_data(show_spinner=False)
def fetch_data(ticker, days):
    df = yf.download(ticker, period=f"{days+1}d", auto_adjust=True)
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    return df

df = fetch_data(selected_ticker, days)

if df.empty:
    st.warning("No data available.")
else:
    st.write(f"Animating clustering over last **{days}** days for **{selected_ticker}**...")

    # --- Generate frames ---
    frames = []
    temp_dir = tempfile.mkdtemp()

    for i in range(10, days + 1):  # Start from 10 data points for clustering
        sub_df = df.iloc[:i]
        if sub_df.empty or len(sub_df) < 3:
            continue

        kmeans = KMeans(n_clusters=3, random_state=42)
        sub_df["Cluster"] = kmeans.fit_predict(sub_df[["Return", "Volume"]])

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(sub_df["Return"], sub_df["Volume"], c=sub_df["Cluster"], cmap="viridis", s=20)
        ax.set_title(f"{selected_ticker} Clustering - Day {i}")
        ax.set_xlabel("Return")
        ax.set_ylabel("Volume")

        frame_path = os.path.join(temp_dir, f"frame_{i}.png")
        fig.savefig(frame_path)
        frames.append(imageio.v2.imread(frame_path))
        plt.close(fig)

    # --- Save GIF ---
    gif_path = os.path.join(temp_dir, f"{selected_ticker}_animation.gif")
    imageio.mimsave(gif_path, frames, duration=0.3)  # duration per frame

    # --- Display GIF ---
    st.image(gif_path, caption=f"{selected_ticker} clustering over {days} days", use_container_width=500)
