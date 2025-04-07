import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Stock Return-Volume Clustering")

# --- Inputs ---
tickers = st.text_input("Enter 5 stock tickers separated by commas (e.g., AAPL, MSFT, TSLA, GOOGL, AMZN)", value="AAPL, MSFT, TSLA, GOOGL, AMZN")
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)
n_clusters = st.slider("Number of clusters per stock", min_value=2, max_value=10, value=3)

# --- Function to fetch and prepare data ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    return df

# --- Clustering and Plotting ---
cols = st.columns(len(tickers))
combined_data = []

for i, ticker in enumerate(tickers):
    with cols[i]:
        st.subheader(ticker)
        df = fetch_stock_data(ticker, days)
        if df is None or df.empty:
            st.warning(f"Data not available for {ticker}")
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[["Return", "Volume"]])
        df["Ticker"] = ticker
        combined_data.append(df)

        # Plot
        fig, ax = plt.subplots(figsize=(4, 4))
        scatter = ax.scatter(df["Return"], df["Volume"], c=df["Cluster"], cmap="tab10", s=10)
        ax.set_xlabel("1-Day Return")
        ax.set_ylabel("Volume")
        ax.set_title(f"{ticker} Clusters")
        st.pyplot(fig)

        # Cluster summary table
        st.markdown("**Cluster Summary**")
        summary = df.groupby("Cluster").agg(
            Points=("Return", "count"),
            Avg_Return=("Return", "mean"),
            Avg_Volume=("Volume", "mean")
        ).round(4)
        st.dataframe(summary)

# --- Export all data ---
if combined_data:
    st.markdown("### 📁 Download Combined Clustered Data")
    all_df = pd.concat(combined_data).reset_index(drop=True)
    csv = all_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="clustered_stocks.csv", mime="text/csv")
