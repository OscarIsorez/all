import pandas as pd

# Chargement des fichiers
appointments = pd.read_csv("data/appointments/appointments.txt", sep=r"\s+")
participants = pd.read_csv("data/appointments/participants.txt", sep=r"\s+")

# Fusionner les deux fichiers
data = appointments.merge(participants, on="participant")

# Filtrer les patients avec au moins 5 rendez-vous
data = data[data["count"] >= 5]


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, make_scorer, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay


# Identification des colonnes
categorical_columns = [
    "sms_received",
    "sex",
    "hipertension",
    "diabetes",
    "alcoholism",
    "weekday",
    "status",
]
numerical_columns = ["age", "advance", "day", "month", "count"]

numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)


X = data.drop(columns=["participant"])
y = data["status"].map({"fullfilled": 1, "no-show": 0})


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVM": SVC(probability=True),
}

param_grid = {
    "RandomForest": {"model__n_estimators": [100, 200], "model__max_depth": [5, 10]},
    "GradientBoosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
    },
    "SVM": {"model__C": [0.1, 1], "model__kernel": ["linear", "rbf"]},
}

best_estimators = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    grid_search = GridSearchCV(
        pipeline, param_grid[name], cv=cv, scoring="f1", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_estimators[name] = grid_search.best_estimator_

for name, estimator in best_estimators.items():
    y_scores = estimator.predict_proba(X_test)[:, 1]
    display = PrecisionRecallDisplay.from_predictions(y_test, y_scores)
    display.ax_.set_title(f"Precision-Recall Curve: {name}")
    plt.savefig(f"results/t3_precision_recall_cv_for{name}.png")
    plt.show()


import shap

# SHAP
explainer = shap.Explainer(best_estimators["GradientBoosting"].named_steps["model"])
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

from sklearn.ensemble import StackingClassifier

stacked_model = StackingClassifier(estimators=[
    ("rf", best_estimators["RandomForest"]),
    ("gb", best_estimators["GradientBoosting"]),
    ("svm", best_estimators["SVM"])
], final_estimator=GradientBoostingClassifier())

stacked_model.fit(X_train, y_train)


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_estimators["GradientBoosting"], X_train, y_train, cv=cv, scoring="f1"
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")
plt.legend()
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve")
plt.savefig("results/t3_learning_curve.png")
plt.show()
