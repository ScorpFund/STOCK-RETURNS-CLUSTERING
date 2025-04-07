import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

st.set_page_config(page_title="📈 Multi-Stock Cluster Visualizer", layout="wide")

st.title("📊 Multi-Stock Return-Volume Cluster Visualizer")

tickers = st.text_input("Enter up to 5 stock tickers (comma-separated)", "MSFT,AAPL,GOOGL,AMZN,TSLA")
num_days = st.number_input("How many past days of data to fetch?", min_value=30, max_value=1000, value=365)
max_clusters = st.slider("Max Clusters (Auto detects best k)", min_value=2, max_value=10, value=5)

ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:5]
end_date = datetime.today()
start_date = end_date - timedelta(days=int(num_days))

def get_cluster_suggestion(X, max_k=10):
    best_k = 2
    best_score = -1
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

for ticker in ticker_list:
    st.subheader(f"📌 {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

# If the result is MultiIndex and has the ticker as column level
if isinstance(df.columns, pd.MultiIndex):
    df = df[ticker]  # select the data for the specific ticker

    df["Return"] = df["Adj Close"].pct_change()
    df = df.dropna()
    df["Volume"] = df["Volume"].astype(float)
    X = df[["Return", "Volume"]]

    suggested_k = get_cluster_suggestion(X, max_k=max_clusters)
    st.write(f"Suggested clusters: {suggested_k}")

    kmeans = KMeans(n_clusters=suggested_k, random_state=42).fit(X)
    df["Cluster"] = kmeans.labels_

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Volume", y="Return", hue="Cluster", palette="tab10", ax=ax)
    ax.set_title(f"{ticker} Return vs Volume Clusters")
    st.pyplot(fig)

    with st.expander("📈 Cluster Insights"):
        cluster_insights = df.groupby("Cluster").agg({
            "Return": ["mean", "count"],
            "Volume": "mean"
        }).round(4)
        st.dataframe(cluster_insights)
