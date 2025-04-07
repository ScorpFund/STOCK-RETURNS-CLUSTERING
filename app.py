import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

st.set_page_config(layout="centered")
st.title("🔍 Store Stock Clustering Data in Vector DB")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

# --- User Input ---
ticker = st.text_input("Enter a NSE stock ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
days = st.slider("Number of days of historical data", min_value=30, max_value=3650, value=180)

# --- Fetch and process data ---
def fetch_clustered_data(ticker, days):
    df = yf.download(ticker, period=f"{days}d", auto_adjust=True)
    if df.empty or "Close" not in df or "Volume" not in df:
        return None
    df["Return"] = df["Close"].pct_change()
    df = df[["Return", "Volume"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df[["Return", "Volume"]])
    return df

# --- Store cluster data in vector DB ---
if st.button("Fetch and Store in Vector DB") and ticker:
    df = fetch_clustered_data(ticker, days)
    if df is not None:
        # Describe clusters to embed
        cluster_descriptions = []
        for c in df["Cluster"].unique():
            stats = df[df["Cluster"] == c].describe().loc[["mean", "std"]]
            desc = f"{ticker} cluster {c} - mean return: {stats.loc['mean', 'Return']:.4f}, " \
                   f"std return: {stats.loc['std', 'Return']:.4f}, mean volume: {stats.loc['mean', 'Volume']:.2f}"
            cluster_descriptions.append(desc)

        # Embed
        embeddings = embedder.encode(cluster_descriptions)

        # Create and store FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))

        st.success(f"Stored {len(embeddings)} cluster embeddings in FAISS index for {ticker}.")

        # Optionally persist the FAISS index to disk
        index_path = f"faiss_index_{ticker.replace('.', '_')}.index"
        faiss.write_index(index, index_path)
        st.info(f"FAISS index saved to: {index_path}")
    else:
        st.error("Failed to fetch or process data.")
