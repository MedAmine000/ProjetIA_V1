import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from DataBasePrep import X_prepared  # Charger les données sans les étiquettes

# **1. Initialisation et entraînement de K-Means**
# # Méthode Elbow pour déterminer le bon nombre de clusters
inertia = []
for k in range(1, 100):  # Tester différents nombres de clusters
 kmeans = KMeans(n_clusters=k, random_state=42)
 kmeans.fit(X_prepared)
 inertia.append(kmeans.inertia_)
#
# Plot de la méthode Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 100), inertia, marker='o')
plt.title("Méthode Elbow pour déterminer le nombre optimal de clusters")
plt.xlabel("Nombre de Clusters")
plt.ylabel("Inertia")
plt.show()

# Nombre de clusters
n_clusters = 2

# Initialisation de K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_prepared)

# Résultats du clustering
clusters = kmeans.labels_

# **2. Évaluation de la qualité des clusters**
# Métriques courantes
silhouette_avg = silhouette_score(X_prepared, clusters)
davies_bouldin = davies_bouldin_score(X_prepared, clusters)

print(f"Score de silhouette : {silhouette_avg:.2f}")
print(f"Score de Davies-Bouldin : {davies_bouldin:.2f}")

# **3. Visualisation des clusters**
# Réduction de la dimensionnalité avec PCA pour visualisation
pca = PCA(n_components=2)  # Réduire à 2 dimensions
X_pca = pca.fit_transform(X_prepared)

# Scatter plot des clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title(f"Visualisation des Clusters (K-Means, k={n_clusters})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.show()

# **4. Sauvegarde des clusters pour analyse ultérieure**
# Ajout des clusters dans un DataFrame pour interprétation
df_clusters = pd.DataFrame(X_prepared, columns=[f"Feature_{i}" for i in range(X_prepared.shape[1])])
df_clusters["Cluster"] = clusters

# Sauvegarder le DataFrame
df_clusters.to_csv("kmeans_clusters.csv", index=False)
print("Clusters sauvegardés dans 'kmeans_clusters.csv'.")
