import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Stock Return-Volume Clustering Dashboard")

# --- Inputs ---
tickers = st.text_input(
    "Enter 5 stock tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)",
    value="AAPL, MSFT, TSLA, GOOGL, AMZN"
)
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)

num_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

# --- Function to fetch and prepare data ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if df.empty or "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna().copy()
    df["Ticker"] = ticker
    df.reset_index(drop=True, inplace=True)
    return df

# --- Collect and combine data ---
all_data = []
for ticker in tickers:
    df = fetch_stock_data(ticker, days)
    if df is not None:
        all_data.append(df)

if all_data:
    data = pd.concat(all_data, ignore_index=True)

    # Ensure the columns are clean and no NaNs
    data = data[["Return", "Volume", "Ticker"]].dropna()

    # Apply clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    data["Cluster"] = kmeans.fit_predict(data[["Return", "Volume"]])

    # --- Cluster Summary Stats ---
    st.subheader("📈 Cluster Summary (All Stocks Combined)")
    cluster_stats = data.groupby("Cluster")[["Return", "Volume"]].mean().round(4)
    st.dataframe(cluster_stats)

    # --- Cluster Distribution per Stock ---
    st.subheader("🔍 Cluster Distribution by Ticker")
    cluster_dist = data.groupby(["Ticker", "Cluster"]).size().unstack(fill_value=0)
    st.dataframe(cluster_dist)

    # --- Combined Plot ---
    st.subheader("🌐 Combined Cluster Plot")
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="Return",
        y="Volume",
        hue="Cluster",
        style="Ticker",
        palette="tab10",
        s=30,
        ax=ax_combined
    )
    ax_combined.set_title("All Stocks: Return vs Volume Clusters")
    st.pyplot(fig_combined)

    # --- Individual Plots ---
    st.subheader("📉 Individual Stock Clusters")
    cols = st.columns(len(tickers))
    for i, ticker in enumerate(tickers):
        with cols[i]:
            st.markdown(f"**{ticker}**")
            stock_df = data[data["Ticker"] == ticker]
            if not stock_df.empty:
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.scatterplot(
                    data=stock_df,
                    x="Return",
                    y="Volume",
                    hue="Cluster",
                    palette="tab10",
                    s=15,
                    ax=ax,
                    legend=False
                )
                ax.set_title(ticker)
                st.pyplot(fig)
else:
    st.error("⚠️ No valid stock data available for the entered tickers.")
