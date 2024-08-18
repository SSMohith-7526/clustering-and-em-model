import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'Iris.csv'
data = pd.read_csv(file_path)

# Dropping the 'Id' column if it exists, since it's not useful for clustering
data.drop(columns=['Id'], inplace=True)

# We will use only the features for clustering, drop the species column if present
X = data.drop(columns=['Species'], axis=1).values

# Optional: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.cluster import KMeans

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Optionally, print or plot the cluster centers
print("K-Means Cluster Centers:")
print(kmeans.cluster_centers_)
from sklearn.mixture import GaussianMixture

# Apply EM Algorithm using Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Optionally, print or plot the means of the GMM
print("GMM Means:")
print(gmm.means_)
from sklearn.metrics import adjusted_rand_score

# Compare the results using Adjusted Rand Index
ari = adjusted_rand_score(kmeans_labels, gmm_labels)
print(f"Adjusted Rand Index between K-Means and GMM clusters: {ari:.3f}")

# Optionally, visualize the clusters
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.title('K-Means Clustering')

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis', marker='o')
plt.title('GMM Clustering')

plt.show()
