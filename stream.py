import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# --- Title and Intro ---
st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")
st.markdown("""
Adjust clustering parameters below and explore interactive visualizations to understand customer segments and their characteristics.
""")

# --- Load Data ---
data = pd.read_csv("data/processed_data.csv")

# --- Sidebar for parameters ---
st.sidebar.header("Clustering Parameters")
features = st.sidebar.multiselect("Select features for clustering", list(data.columns), default=list(data.columns[:3]))
n_clusters = st.sidebar.slider("Number of segments", 2, 10, 3)

# --- Check selection ---
if len(features) < 2:
    st.warning("Please select at least 2 features for clustering.")
else:
    # --- Scale and cluster ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=69)
    data['Cluster'] = kmeans.fit_predict(scaled_data)

    # --- Summary Stats ---
    st.subheader("Segment Summary Statistics")
    cluster_summary = data.groupby('Cluster')[features].mean().round(2).sort_index()
    st.dataframe(cluster_summary)

    # --- Distribution Plot for Each Feature ---
    st.subheader("Feature Distribution by Segment")
    for feature in features:
        fig = px.box(data, x="Cluster", y=feature, color="Cluster", title=f"{feature} Distribution by Cluster")
        st.plotly_chart(fig, use_container_width=True)

    # --- 2D Scatter Plot ---
    if len(features) >= 2:
        st.subheader("2D Cluster Scatter Plot")
        fig2d = px.scatter(
            data,
            x=features[0],
            y=features[1],
            color='Cluster',
            title=f"2D Scatter: {features[0]} vs {features[1]}",
            opacity=0.7
        )
        st.plotly_chart(fig2d, use_container_width=True)

        # Seaborn jointplot
        st.markdown(f"### Joinplot for {features[0]} vs {features[1]}")
        g = sns.jointplot(
            x=features[0],
            y=features[1],
            hue=data["Cluster"],
            data=data,
            kind="kde",
            palette="Set2"
        )
        st.pyplot(g.figure)
        plt.close()

    # --- 3D Scatter Plot ---
    if len(features) >= 3:
        st.subheader("3D Cluster Scatter Plot")
        fig3d = px.scatter_3d(
            data,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Cluster',
            title="3D Cluster Visualization",
            opacity=0.7
        )
        st.plotly_chart(fig3d, use_container_width=True)

    # --- Insights Section ---
    st.subheader("Segment Insights")
    for cluster_id in sorted(data['Cluster'].unique()):
        cluster_data = data[data['Cluster'] == cluster_id]
        st.markdown(f"### Segment {cluster_id}")
        st.write(f"- **Count**: {len(cluster_data)} customers")
        top_features = cluster_summary.loc[cluster_id].sort_values(ascending=False)
        st.write("- **Key Traits**:")
        for feature, value in top_features.items():
            st.write(f"    • {feature}: {value:.2f}")


# --- Elbow Method Plot ---
st.subheader("Elbow Method for Optimal no. of Clusters")
st.markdown("Based on Elbow Method this data has 4 clusters")
fig, ax = plt.subplots()
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
ax.plot(range(1, 11), inertia, marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Inertia")
ax.axvline(x=4, color='r', linestyle='--', label='Optimal Clusters (4)')
ax.legend()
st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.caption("© 2025 Sigma | Customer Segmentation App")
