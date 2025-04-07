import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Stock Return-Volume Clustering (NSE Support)")

# --- Sample NSE stock list (replace with full list if needed) ---
nse_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS"
]

# --- User Inputs ---
st.markdown("### Select stocks from dropdown or enter manually")
selected_from_dropdown = st.multiselect("Pick up to 5 NSE stocks", options=nse_stocks)

manual_input = st.text_input(
    "Or enter up to 5 stock tickers manually (e.g., AAPL, MSFT, TSLA)",
    value="RELIANCE.NS, INFY.NS"
)
manual_tickers = [t.strip().upper() for t in manual_input.split(",") if t.strip()]

# Combine selections, prioritize dropdown, fallback to manual
tickers = selected_from_dropdown if selected_from_dropdown else manual_tickers
tickers = tickers[:5]

# Days of data
days = st.slider("Number of days of historical data to fetch", min_value=30, max_value=365, value=180, step=15)

# --- Function to fetch and prepare data ---
def fetch_stock_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    return df

# --- Clustering and Plotting ---
if not tickers:
    st.warning("Please select or enter at least one stock ticker.")
else:
    cols = st.columns(len(tickers))

    for i, ticker in enumerate(tickers):
        with cols[i]:
            st.subheader(ticker)
            df = fetch_stock_data(ticker, days)
            if df is None or df.empty:
                st.warning(f"Data not available for {ticker}")
                continue

            kmeans = KMeans(n_clusters=3, random_state=42)
            df["Cluster"] = kmeans.fit_predict(df[["Return", "Volume"]])

            fig, ax = plt.subplots(figsize=(4, 4))
            scatter = ax.scatter(df["Return"], df["Volume"], c=df["Cluster"], cmap="viridis", s=10)
            ax.set_xlabel("1-Day Return")
            ax.set_ylabel("Volume")
            ax.set_title(f"{ticker} Clusters")
            st.pyplot(fig)
