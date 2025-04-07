import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("📊 Return-Volume Cluster Visualizer")

ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y")
    df["Return"] = df["Close"].pct_change()
    return df.dropna()

df = load_data(ticker)

X = df[["Return", "Volume"]].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df["Cluster"] = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df["Volume"], df["Return"], c=df["Cluster"], cmap="tab10", alpha=0.7)
ax.set_xlabel("Volume")
ax.set_ylabel("Daily Return")
ax.set_title(f"{ticker} Return vs Volume Clusters")
ax.grid(True)

st.pyplot(fig)

if st.checkbox("Show raw data"):
    st.dataframe(df)
