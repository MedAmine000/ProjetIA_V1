from ucimlrepo import fetch_ucirepo

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# # metadata
# print(bank_marketing.metadata)
#
# # variable information
# print(bank_marketing.variables)


##################################################################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# **1. Vérification des données**
print(X.head())  # Aperçu des premières lignes
print(X.info())  # Types des colonnes
print(X.describe())  # Statistiques descriptives

# **2. Traitement des valeurs manquantes**
# Variables avec valeurs manquantes
missing_cols = ['contact', 'pdays', 'poutcome']
imputer = SimpleImputer(strategy='most_frequent')  # Imputation par le mode
X.loc[:, missing_cols] = imputer.fit_transform(X[missing_cols])

# Vérification après imputation
print(X[missing_cols].isnull().sum())

# **3. Encodage des variables catégoriques**
# Identification des colonnes catégoriques
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome']
num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Encodage One-Hot pour les colonnes catégoriques
ohe = OneHotEncoder(drop='first', sparse_output=False)

# Création d’un pipeline pour la transformation des données
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),  # Standardisation des variables numériques
        ('cat', ohe, cat_cols)               # One-Hot Encoding pour les catégoriques
    ]
)

# Transformation des données
X_prepared = preprocessor.fit_transform(X)

# Vérification des dimensions après transformation
print(X_prepared.shape)

# **4. Division en ensembles d'entraînement et de test**
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.3, random_state=42)

# Résumé des dimensions
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
