
```markdown
# Projet AI : Comparaison des Algorithmes d'Intelligence Artificielle

## Description
Ce projet compare les performances de trois algorithmes d’intelligence artificielle :
1. **Arbre de décision** : Utilisé pour des tâches de classification supervisée.
2. **Clustering K-Means** : Utilisé pour segmenter les données sans supervision.
3. **Réseau de neurones (Perceptron multicouche)** : Utilisé pour des tâches de classification supervisée avec une capacité d’apprentissage élevée.

---

## Structure du projet

```
ProjetAI/
│
├── cluster_analysis.csv           # Résultats du clustering K-Means
├── ClusteringModel.py             # Script pour entraîner K-Means
├── ClusteringModelAnalysis.py     # Analyse des clusters
├── DataBasePrep.py                # Préparation des données
├── decision_tree_model.pkl        # Modèle d'arbre de décision sauvegardé
├── DecisionTreeModel.py           # Script pour entraîner l'arbre de décision
├── kmeans_clusters.csv            # Cluster labels générés par K-Means
├── neural_network_model.pkl       # Modèle de réseau de neurones sauvegardé
├── NeuralNetworkModel.py          # Script pour entraîner le réseau de neurones
└── README.md                      # Documentation du projet
```

---

## Installation et exécution

### 1. **Cloner le dépôt**
Cloner le projet depuis le dépôt GitHub ou votre système local :
```bash
git clone https://github.com/username/ProjetAI.git
cd ProjetAI
```

### 2. **Installer les dépendances**
Assurez-vous d’avoir Python installé. Installez les bibliothèques nécessaires avec :
```bash
pip install -r requirements.txt
```

### 3. **Étapes d'exécution**

- **Préparation des données** :
  Préparer la base de données pour l'analyse :
  ```bash
  python DataBasePrep.py
  ```

- **Arbre de décision** :
  Entraîner et évaluer l'arbre de décision :
  ```bash
  python DecisionTreeModel.py
  ```

- **Clustering K-Means** :
  Exécuter le modèle K-Means et analyser les clusters :
  ```bash
  python ClusteringModel.py
  python ClusteringModelAnalysis.py
  ```

- **Réseau de neurones** :
  Entraîner et évaluer le réseau de neurones :
  ```bash
  python NeuralNetworkModel.py
  ```

---

## Résultats et analyse
Les performances des modèles et les résultats sont détaillés dans le fichier `rapport.md`. Ce document explique les forces et faiblesses de chaque algorithme et donne des recommandations.

---

## Auteur
Ce projet a été créé par Korniti MedAmine. N'hésitez pas à me contacter pour toute question ou amélioration.
```

