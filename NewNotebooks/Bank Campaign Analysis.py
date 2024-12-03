import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Charger le jeu de données encodé
file_path = r"C:\Users\hergi\Documents\hergiDiangue\PGE3Fr\Coding for AI & Data Science\projetS1\Data\bank\bank.csv"
data = pd.read_csv(file_path, sep=';')

# Vérifier le nom de la colonne cible
print("Colonnes disponibles :", data.columns.tolist())

# Corriger les noms de colonnes en supprimant les espaces éventuels
data.columns = data.columns.str.strip()

# Séparer les variables explicatives (X) et la variable cible (y)
X = data.drop(columns=['y'])
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Séparer les caractéristiques numériques et catégorielles
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement : StandardScaler pour les colonnes numériques et OneHotEncoder pour les colonnes catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Liste des modèles à entraîner
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Dictionnaire pour stocker les performances
model_performance = {}

# Entraîner chaque modèle avec un pipeline et évaluer ses performances
for model_name, model in models.items():
    # Créer un pipeline qui inclut le prétraitement et le modèle
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Entraîner le pipeline
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculer la précision et le F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Stocker les performances
    model_performance[model_name] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "pipeline": pipeline  # Stocker le pipeline
    }

# Trouver le meilleur modèle basé sur le F1-score
best_model_name = max(model_performance, key=lambda x: model_performance[x]["f1_score"])
best_pipeline = model_performance[best_model_name]["pipeline"]

# Afficher les performances
print("Performances des modèles :")
for model_name, performance in model_performance.items():
    print(f"{model_name} -> Accuracy: {performance['accuracy']:.4f}, F1-Score: {performance['f1_score']:.4f}")

print(f"\nLe meilleur modèle est : {best_model_name}")

# Enregistrer le meilleur pipeline
joblib.dump(best_pipeline, "best_model_pipeline.pkl")
print(f"\nLe meilleur pipeline a été enregistré sous le nom 'best_model_pipeline.pkl'")

# Tester le meilleur modèle avec des cas spécifiques
# Cas de souscription modifié pour garantir la probabilité de souscription
test_cases = [
    {   # Cas de souscription (fortement amélioré pour maximiser la probabilité)
        'age': 60, 'job': 'management', 'marital': 'single', 'education': 'tertiary', 'default': 'no',
        'balance': 300000, 'housing': 'no', 'loan': 'no', 'contact': 'cellular', 'day': 10, 'month': 'aug',
        'duration': 6000, 'campaign': 1, 'pdays': 999, 'previous': 50, 'poutcome': 'success'
    },
    {   # Cas de non-souscription
        'age': 25, 'job': 'blue-collar', 'marital': 'married', 'education': 'primary', 'default': 'yes',
        'balance': 200, 'housing': 'yes', 'loan': 'yes', 'contact': 'unknown', 'day': 5, 'month': 'may',
        'duration': 50, 'campaign': 4, 'pdays': 999, 'previous': 0, 'poutcome': 'failure'
    }
]

# Convertir les cas de test en DataFrame
test_df = pd.DataFrame(test_cases)

# Faire des prédictions pour les cas de test
print("\nPrédictions des cas de test avec probabilités :")
test_predictions = best_pipeline.predict(test_df)
test_probabilities = best_pipeline.predict_proba(test_df)
for i, (prediction, probability) in enumerate(zip(test_predictions, test_probabilities)):
    result = 'Souscription' if prediction == 1 else 'Pas de souscription'
    probability_percent = probability[1] * 100  # Probabilité de souscription
    print(f"Cas de test {i+1} : {result} (Probabilité de souscription : {probability_percent:.2f}%)")
