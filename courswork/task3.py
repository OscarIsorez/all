import pandas as pd

# Chargement des fichiers
appointments = pd.read_csv("data/appointments/appointments.txt", sep=r"\s+")
participants = pd.read_csv("data/appointments/participants.txt", sep=r"\s+")

# Fusionner les deux fichiers
data = appointments.merge(participants, on="participant")

# Filtrer les patients avec au moins 5 rendez-vous
data = data[data["count"] >= 5]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

# Identification des colonnes
categorical_columns = [
    "sms_received",
    "sex",
    "hipertension",
    "diabetes",
    "alcoholism",
    "weekday",
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

X = data.drop(columns=["status"], axis=1)
y = data["status"].map({"fullfilled": 1, "no-show": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=127)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Pipeline for Logistic Regression
log_reg_pipeline = make_pipeline(
    preprocessor, LogisticRegression(max_iter=1000, random_state=127)
)

log_reg_scores = cross_val_score(
    log_reg_pipeline, X_train, y_train, scoring="f1", cv=cv
)
print("Logistic Regression F1 Score:", log_reg_scores.mean())

from sklearn.ensemble import RandomForestClassifier

rf_pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=127))
rf_scores = cross_val_score(rf_pipeline, X_train, y_train, scoring="f1", cv=cv)
print("Random Forest F1 Score:", rf_scores.mean())

from sklearn.ensemble import GradientBoostingClassifier

gb_pipeline = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=127))
gb_scores = cross_val_score(gb_pipeline, X_train, y_train, scoring="f1", cv=cv)
print("Gradient Boosting F1 Score:", gb_scores.mean())

from sklearn.metrics import precision_recall_curve

# Sélection du meilleur pipeline
if rf_scores.mean() > gb_scores.mean():
    best_pipeline = rf_pipeline
    feature_importance_attr = "feature_importances_"
else:
    best_pipeline = gb_pipeline
    feature_importance_attr = "feature_importances_"

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("results/t3_precision_recall_curve.png")
plt.close()

# Importance des caractéristiques
importances = getattr(
    best_pipeline.named_steps[best_pipeline.steps[-1][0]], feature_importance_attr
)
feature_names = best_pipeline.named_steps["columntransformer"].get_feature_names_out()
plt.figure(figsize=(10, 8))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("results/t3_feature_importance.png")
plt.close()

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_pipeline, X, y, cv=cv, scoring="f1"
)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test")
plt.xlabel("Training Examples")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig("results/t3_learning_curve.png")
plt.close()

from sklearn.ensemble import StackingClassifier

stacked_clf = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(random_state=127)),
        ("gb", GradientBoostingClassifier(random_state=127)),
    ],
    final_estimator=LogisticRegression(),
)

pipeline = make_pipeline(preprocessor, stacked_clf)

stacked_scores = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=cv)
print("Stacked Model F1 Score:", stacked_scores.mean())


outer_cv = StratifiedKFold(7, shuffle=True, random_state=127)
inner_cv = StratifiedKFold(6, shuffle=True, random_state=128)

from sklearn.model_selection import GridSearchCV

params = {"randomforestclassifier__max_depth": [5, 10, None]}
grid = GridSearchCV(rf_pipeline, params, cv=inner_cv, scoring="f1")
nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring="f1")
print("Nested CV Score:", nested_scores.mean())
