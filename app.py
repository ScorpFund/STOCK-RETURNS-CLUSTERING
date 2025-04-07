import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Combined Return-Volume Clustering for Multiple Stocks")

# --- Inputs ---
tickers = st.text_input("Enter up to 5 stock tickers (e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)", value="AAPL, MSFT, TSLA, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)
n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=10, value=4)

# --- Fetch & combine data ---
def fetch_and_prepare(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if "Close" not in df or "Volume" not in df:
        return pd.DataFrame()
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    df["Ticker"] = ticker
    return df

with st.spinner("Fetching data..."):
    combined_df = pd.concat([fetch_and_prepare(t, days) for t in tickers], ignore_index=True)

# --- Clustering ---
if not combined_df.empty:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    combined_df["Cluster"] = kmeans.fit_predict(combined_df[["Return", "Volume"]])

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=combined_df,
        x="Return",
        y="Volume",
        hue="Cluster",
        style="Ticker",
        palette="tab10",
        ax=ax,
        s=30
    )
    ax.set_title("Combined Clustering of Returns vs Volume", fontsize=16)
    ax.set_xlabel("1-Day Return")
    ax.set_ylabel("Volume")
    st.pyplot(fig)

    # --- Cluster summary ---
    st.subheader("🔍 Cluster Summary")
    summary = combined_df.groupby("Cluster").agg(
        Points=("Return", "count"),
        Avg_Return=("Return", "mean"),
        Avg_Volume=("Volume", "mean")
    ).round(4)
    st.dataframe(summary)

else:
    st.warning("⚠️ Data could not be fetched for the given tickers.")
