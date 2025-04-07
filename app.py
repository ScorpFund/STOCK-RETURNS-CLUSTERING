import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Stock Return-Volume Clustering Dashboard")

# --- Inputs ---
tickers = st.text_input("Enter 5 stock tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)", value="AAPL, MSFT, TSLA, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)
k = st.slider("Number of clusters", min_value=2, max_value=10, value=3)

# --- Function to fetch and prepare data ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    df["Ticker"] = ticker
    return df

# --- Collect all data ---
all_df = []
for ticker in tickers:
    df = fetch_stock_data(ticker, days)
    if df is not None and not df.empty:
        all_df.append(df)

data = pd.concat(all_df, axis=0).dropna()

if not data.empty:
    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data[["Return", "Volume"]])

    # Silhouette Score
    silhouette = silhouette_score(data[["Return", "Volume"]], data["Cluster"])
    st.sidebar.metric("Silhouette Score", f"{silhouette:.3f}")

    # Cluster Labels
    def label_cluster(row, cluster_stats):
        mean_ret = cluster_stats.loc[row.Cluster, "Return"]
        mean_vol = cluster_stats.loc[row.Cluster, "Volume"]
        if abs(mean_ret) > 0.02 and mean_vol > 1.5e7:
            return "High Volatility Spike"
        elif abs(mean_ret) < 0.005 and mean_vol < 5e6:
            return "Low Activity"
        elif mean_ret > 0.01:
            return "Positive Momentum"
        elif mean_ret < -0.01:
            return "Negative Pressure"
        else:
            return "Neutral"

    cluster_stats = data.groupby("Cluster")["Return", "Volume"].mean()
    data["Label"] = data.apply(lambda row: label_cluster(row, cluster_stats), axis=1)

    # Combined Scatter Plot
    st.subheader("🔀 Combined Cluster View")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scatter = ax1.scatter(data["Return"], data["Volume"], c=data["Cluster"], cmap="tab10", s=10)
    ax1.set_xlabel("1-Day Return")
    ax1.set_ylabel("Volume")
    ax1.set_title("All Stocks Clustered")
    st.pyplot(fig1)

    # Per-Ticker Summary and Cluster Timeline
    for ticker in tickers:
        df = data[data["Ticker"] == ticker].copy()
        if df.empty:
            continue

        st.subheader(f"📈 {ticker} Insights")

        # Cluster timeline
        fig2, ax2 = plt.subplots(figsize=(7, 2))
        ax2.plot(df.index, df["Cluster"], drawstyle='steps-post')
        ax2.set_title("Cluster Timeline")
        ax2.set_ylabel("Cluster")
        st.pyplot(fig2)

        # Cluster summary
        summary = df.groupby("Cluster").agg(
            Mean_Return=("Return", "mean"),
            Std_Return=("Return", "std"),
            Mean_Volume=("Volume", "mean"),
            Count=("Return", "count")
        )
        st.dataframe(summary)

        # Cluster labels bar chart
        label_counts = df["Label"].value_counts()
        st.bar_chart(label_counts)
else:
    st.warning("No data to show. Please check your tickers or data availability.")
