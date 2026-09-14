import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Clustering Deployment", layout="wide")

st.title("📊 Clustering Deployment App")
st.write("KMeans, Agglomerative (Complete & Single), PCA visualization, and Dendrograms")

# --------------------------------------------------
# LOAD MODELS (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "user": {
            "kmeans": joblib.load("kmeans_user_model.pkl"),
            "pca": joblib.load("pca_user_kmeans.pkl"),
            "agl_complete": joblib.load("agl_complete_user.pkl"),
            "agl_single": joblib.load("agl_single_user.pkl"),
        },
        "item": {
            "kmeans": joblib.load("kmeans_item_model.pkl"),
            "pca": joblib.load("pca_item_kmeans.pkl"),
            "agl_complete": joblib.load("agl_complete_item.pkl"),
            "agl_single": joblib.load("agl_single_item.pkl"),
        },
    }
    return models

models = load_models()

# --------------------------------------------------
# LOAD DATA (YOU MUST ADJUST THIS)
# --------------------------------------------------
@st.cache_data
def load_data():
    X_user = np.load("X_user.npy")
    X_item = np.load("X_item.npy")
    return X_user, X_item

X_user, X_item = load_data()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Controls")

entity = st.sidebar.selectbox("Select Entity", ["user", "item"])
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["KMeans", "Agglomerative - Complete", "Agglomerative - Single"],
)

show_dendrogram = st.sidebar.checkbox("Show Dendrogram", value=False)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------
st.subheader(f"PCA Visualization ({entity.capitalize()})")

pca_model = models[entity]["pca"]
X_scaled = X_user if entity == "user" else X_item
X_pca = pca_model.transform(X_scaled)

if algorithm == "KMeans":
    labels = models[entity]["kmeans"].labels_
elif algorithm == "Agglomerative - Complete":
    labels = models[entity]["agl_complete"].labels_
else:
    labels = models[entity]["agl_single"].labels_

fig, ax = plt.subplots(figsize=(7, 5))

for cluster in np.unique(labels):
    idx = labels == cluster
    ax.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        label=f"Cluster {cluster}",
        s=30
    )

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title(f"{algorithm} PCA Plot ({entity.capitalize()})")
ax.legend(title="Clusters")
ax.grid(True)

st.pyplot(fig)
# --------------------------------------------------
# DENDROGRAM (DIRECT CUT UP TO TRUE MAX)
# --------------------------------------------------
if show_dendrogram:
    st.subheader(f"Dendrogram ({entity.capitalize()})")

    method = "complete" if algorithm == "Agglomerative - Complete" else "single"

    # Sampling for performance
    
    X_sample = X_scaled

    # Compute linkage matrix
    Z = linkage(X_sample, method=method)

    # TRUE maximum cut distance
    max_distance = float(Z[:, 2].max())

    # Slider allows full valid range
    threshold = st.slider(
        "Select Dendrogram Cut (Distance)",
        min_value=0.0,
        max_value=max_distance,
        value=max_distance * 0.8,   # default near top
        step=0.5
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    dendrogram(
        Z,
        color_threshold=threshold,
        above_threshold_color="blue",
        ax=ax
    )

    # Draw cut line exactly at chosen value
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        label=f"Cut = {threshold:.2f}"
    )

    ax.set_title(
        f"{entity.capitalize()} Dendrogram ({method.capitalize()} Linkage)"
    )
    ax.set_ylabel("Distance")
    ax.legend()

    st.pyplot(fig)

    # Cluster count
    clusters = fcluster(Z, t=threshold, criterion="distance")
    st.write("Number of clusters formed:", len(np.unique(clusters)))

#Load lookup + similarity table
@st.cache_data
def load_item_tables():
    clusters = pd.read_csv("item_cluster_lookup.csv")
    similarity = pd.read_csv("item_similarity_df.csv", index_col=0)
    return clusters, similarity

item_clusters, item_similarity_df = load_item_tables()


#Product selector
st.sidebar.subheader("🔍 Product Lookup")

product_id = st.sidebar.selectbox(
    "Select Product ID",
    item_clusters["productid"].unique()
)



#Show cluster membership
row = item_clusters[item_clusters["productid"] == product_id].iloc[0]

st.subheader("📌 Cluster Membership")

st.write(f"KMeans Cluster: {row.kmeans_cluster}")
st.write(f"HAC Complete Cluster: {row.hac_complete_cluster}")
st.write(f"HAC Single Cluster: {row.hac_single_cluster}")


#Use YOUR existing similarity logic
def show_similar_items(product_id, similarity_df, top_n=10):
    if product_id not in similarity_df.columns:
        st.warning("No similarity data available")
        return

    sims = similarity_df[product_id].sort_values(ascending=False)
    sims = sims[sims > 0]

    if sims.empty:
        st.warning("No similar products found")
    else:
        #st.markdown("### 🔗 Similar Products")
        for i, (pid, score) in enumerate(sims.head(10).items(), start=1):
            st.write(f"{i}. {pid}")

#Display similar products
st.subheader("🔗 Similar Products (Item-Based CF)")
show_similar_items(product_id, item_similarity_df)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Deployed clustering app using Streamlit")
