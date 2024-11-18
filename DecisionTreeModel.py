import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # Pour sauvegarder le modèle

# Charger les données préparées depuis DataBasePrep.py
from DataBasePrep import X_train, X_test, y_train, y_test

# **1. Initialisation et formation de l'arbre de décision**
# Configuration de l'arbre
tree_clf = DecisionTreeClassifier(
    criterion='gini',       # ou 'entropy'
    max_depth=5,            # Limiter la profondeur pour éviter le surapprentissage
    random_state=42         # Répétabilité des résultats
)

# Entraînement du modèle
tree_clf.fit(X_train, y_train)

# **2. Évaluation du modèle**
# Prédictions
y_pred_train = tree_clf.predict(X_train)
y_pred_test = tree_clf.predict(X_test)

# Métriques de performance
print("=== Performance sur l'ensemble d'entraînement ===")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train)}")
print("\nClassification Report:")
print(classification_report(y_train, y_pred_train))

print("\n=== Performance sur l'ensemble de test ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Matrice de confusion
print("\n=== Matrice de Confusion ===")
conf_matrix = confusion_matrix(y_test, y_pred_test)
print(conf_matrix)

# **3. Visualisation de l'arbre**
# Affichage de l'arbre sous forme textuelle
print("\n=== Arbre sous forme textuelle ===")
print(export_text(tree_clf))

# Visualisation graphique
plt.figure(figsize=(15, 8))
plot_tree(tree_clf, filled=True, feature_names=None, class_names=True)
plt.title("Arbre de décision")
plt.show()

# **4. Sauvegarde du modèle**
# Export du modèle pour une réutilisation future
joblib.dump(tree_clf, 'decision_tree_model.pkl')
print("Modèle sauvegardé sous le nom 'decision_tree_model.pkl'.")
