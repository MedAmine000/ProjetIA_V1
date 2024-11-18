import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from DataBasePrep import X, y  # Données d'origine
from ClusteringModel import clusters  # Étiquettes des clusters issues de K-Means

# **1. Ajouter les étiquettes de clusters aux données d'origine**
X_clustered = X.copy()
X_clustered["Cluster"] = clusters

# **2. Statistiques descriptives par cluster**
# Identifier les colonnes numériques uniquement
numeric_columns = X_clustered.select_dtypes(include=[np.number]).columns

# Calculer les statistiques descriptives uniquement sur les colonnes numériques
cluster_summary = X_clustered[numeric_columns].groupby(X_clustered["Cluster"]).agg(["mean", "median", "std", "count"])

# Afficher les statistiques par cluster
print("\n=== Statistiques descriptives par cluster ===")
print(cluster_summary)

# **3. Interprétation des clusters**
for cluster in sorted(X_clustered["Cluster"].unique()):
    print(f"\n=== Cluster {cluster} ===")
    print(X_clustered[X_clustered["Cluster"] == cluster][numeric_columns].describe())

# **4. Visualisation des clusters et des caractéristiques**
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.select_dtypes(include=[np.number]).drop(columns=["Cluster"], errors="ignore"))

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Clusters (Projection PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# **5. Analyse des proportions dans les clusters**
cluster_distribution = X_clustered["Cluster"].value_counts()
print("\n=== Répartition des points par cluster ===")
print(cluster_distribution)

# Sauvegarde des clusters pour analyse future
X_clustered.to_csv("cluster_analysis.csv", index=False)
print("\nLes données des clusters ont été sauvegardées dans 'cluster_analysis.csv'.")
