import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Loading data
df = pd.read_csv("customer_segmentation.csv")

#Selecting features for clustering
x = df[['Age', 'Income', 'Occupation', 'Settlement size']]

#Standardise features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Elbow method to find optimal K
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_scaled, kmeans.labels_))
    
#plot elbow curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.tight_layout()
plt.show()

#Optimal k
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(x_scaled)

df['Cluster'] = clusters

# Summarize the segments to understand the "Goal Achievement"
print("\n--- Cluster Characteristics ---")
segment_analysis = df.groupby('Cluster')[['Age', 'Income', 'Occupation', 'Settlement size']].mean()
print(segment_analysis)

#Vizualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title(f'Customer segments (k={optimal_k} clusters)')
plt.colorbar(scatter, label = 'Cluster')
plt.show()
