from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Générer des données fictives
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Étape 1 : Mise à l'échelle
    ('model', SVC())               # Étape 2 : Modèle (SVM ici)
])

# Ajuster le pipeline sur l'ensemble d'entraînement
pipeline.fit(X_train, y_train)

# Évaluer les performances
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

