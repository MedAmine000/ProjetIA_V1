import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # Pour sauvegarder le modèle
from DataBasePrep import X_train, X_test, y_train, y_test  # Import des données préparées

# **1. Initialisation et construction du réseau de neurones**
# Configuration du réseau
nn_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 couches cachées : 100 neurones, puis 50
    activation='relu',            # Fonction d'activation : ReLU
    solver='adam',                # Optimiseur : Adam
    learning_rate_init=0.001,     # Taux d'apprentissage initial
    max_iter=200,                 # Nombre maximal d'itérations
    random_state=42               # Répétabilité
)

# **2. Entraînement du modèle**
# Convertir y_train et y_test en 1D
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print("\n=== Entraînement du réseau de neurones ===")
nn_clf.fit(X_train, y_train)
print("Modèle entraîné avec succès.")

# **3. Évaluation du modèle**
# Prédictions sur les ensembles d'entraînement et de test
y_pred_train = nn_clf.predict(X_train)
y_pred_test = nn_clf.predict(X_test)

# Performances sur l'ensemble d'entraînement
print("\n=== Performances sur l'ensemble d'entraînement ===")
print(f"Accuracy : {accuracy_score(y_train, y_pred_train)}")
print("\nClassification Report (Train) :")
print(classification_report(y_train, y_pred_train))

# Performances sur l'ensemble de test
print("\n=== Performances sur l'ensemble de test ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_test)}")
print("\nClassification Report (Test) :")
print(classification_report(y_test, y_pred_test))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\n=== Matrice de Confusion (Test) ===")
print(conf_matrix)

# **4. Visualisation de la convergence**
# Visualisation de la perte (loss) au cours des itérations
plt.figure(figsize=(8, 5))
plt.plot(nn_clf.loss_curve_)
plt.title("Courbe de perte au cours des itérations")
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.show()

# **5. Sauvegarde du modèle**
joblib.dump(nn_clf, 'neural_network_model.pkl')
print("\nModèle sauvegardé sous le nom 'neural_network_model.pkl'.")
