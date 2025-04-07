import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Multi-Stock Return & Volume Clustering")

# --- Inputs ---
tickers = st.text_input(
    "Enter up to 5 stock tickers (comma-separated):", 
    value="AAPL, MSFT, TSLA, GOOGL, AMZN"
)
tickers = [t.strip().upper() for t in tickers.split(",")][:5]

days = st.slider("Select number of days to fetch historical data", 30, 365, 180, 15)
n_clusters = st.slider("Select number of clusters", 2, 10, 4)

# --- Fetch data for each stock ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if df.empty or "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    df["Ticker"] = ticker
    return df

all_data = []

for ticker in tickers:
    df = fetch_stock_data(ticker, days)
    if df is not None and not df.empty:
        all_data.append(df)
    else:
        st.warning(f"⚠️ No data for {ticker}")

# --- Combine and cluster ---
if all_data:
    combined_df = pd.concat(all_data)
    combined_df = combined_df.dropna()

    if len(combined_df) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        combined_df["Cluster"] = kmeans.fit_predict(combined_df[["Return", "Volume"]])

        # Plot combined clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            combined_df["Return"], 
            combined_df["Volume"], 
            c=combined_df["Cluster"], 
            cmap="tab10", 
            s=10,
            alpha=0.8
        )
        ax.set_title("Clustered Return vs Volume (All Stocks)")
        ax.set_xlabel("1-Day Return")
        ax.set_ylabel("Volume")
        ax.grid(True)
        st.pyplot(fig)

        # Summary table
        st.markdown("### 📊 Cluster Summary")
        summary = combined_df.groupby("Cluster").agg(
            Count=("Return", "count"),
            Avg_Return=("Return", "mean"),
            Avg_Volume=("Volume", "mean")
        ).round(4)
        st.dataframe(summary)
    else:
        st.error("❌ Not enough data to form clusters. Try reducing cluster count or fetching more days.")
else:
    st.error("❌ No data available for selected tickers.")
