import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Charger les données
data = pd.read_pickle("processed/heart.pkl.gz")
# Prétraitement
data["sex"] = data["sex"].map({"M": 1, "F": 0})
x = data.drop(columns=["time"])
y = data["time"]  

# Division en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
print("MAE (Régression Logistique) :", mae_lr)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7, weights="uniform")
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
print("MAE (k-NN) :", mae_knn)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
print("MAE (Arbre de Décision) :", mae_dt)

import matplotlib.pyplot as plt

models = ["Logistic Regression", "k-NN", "Decision Tree"]
maes = [mae_lr, mae_knn, mae_dt]

plt.bar(models, maes, color=["blue", "orange", "green"])
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Comparison of Model Performances")
plt.savefig("results/task2.png")
plt.show()

from sklearn.model_selection import GridSearchCV

# Tuning des hyperparamètres pour KNeighborsRegressor
param_grid_knn = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
grid_search_knn = GridSearchCV(
    KNeighborsRegressor(), param_grid_knn, cv=5, scoring="neg_mean_absolute_error"
)
grid_search_knn.fit(x_train, y_train)

print("Meilleurs paramètres (k-NN) :", grid_search_knn.best_params_)
print("Meilleure MAE (k-NN) :", -grid_search_knn.best_score_)

# Tuning des hyperparamètres pour LogisticRegression
param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "lbfgs"]}
grid_search_lr = GridSearchCV(
    LogisticRegression(), param_grid_lr, cv=5, scoring="neg_mean_absolute_error"
)
grid_search_lr.fit(x_train, y_train)

print("Meilleurs paramètres (Régression Logistique) :", grid_search_lr.best_params_)
print("Meilleure MAE (Régression Logistique) :", -grid_search_lr.best_score_)

# Tuning des hyperparamètres pour DecisionTreeRegressor
param_grid_dt = {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]}
grid_search_dt = GridSearchCV(
    DecisionTreeRegressor(), param_grid_dt, cv=5, scoring="neg_mean_absolute_error"
)
grid_search_dt.fit(x_train, y_train)

print("Meilleurs paramètres (Arbre de Décision) :", grid_search_dt.best_params_)
print("Meilleure MAE (Arbre de Décision) :", -grid_search_dt.best_score_)
