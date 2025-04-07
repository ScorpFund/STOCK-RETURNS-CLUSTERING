import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Individual Stock Return-Volume Clustering")

# --- Inputs ---
tickers = st.text_input(
    "Enter up to 5 stock tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)",
    value="AAPL, MSFT, TSLA, GOOGL, AMZN"
)
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)
num_clusters = st.slider("Number of clusters per stock", min_value=2, max_value=6, value=3)

# --- Data Fetcher ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if df.empty or "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna().copy()
    return df

# --- Main Loop ---
cols = st.columns(len(tickers))

for i, ticker in enumerate(tickers):
    with cols[i]:
        st.subheader(ticker)
        df = fetch_stock_data(ticker, days)
        if df is None or df.empty or df.shape[0] < num_clusters:
            st.warning(f"Insufficient data for {ticker}")
            continue

        # Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[["Return", "Volume"]])

        # Clean Plot
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.scatterplot(data=df, x="Return", y="Volume", hue=df["Cluster"], palette="tab10", s=15, ax=ax)
        ax.set_title(f"{ticker} Clusters")
        st.pyplot(fig)

        # Cluster stats
        st.markdown("**Cluster Averages:**")
        st.dataframe(df.groupby("Cluster")[["Return", "Volume"]].mean().round(4))
