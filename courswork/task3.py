import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load data
appointments = pd.read_csv("data/appointments/appointments.txt", sep=r"\s+")
participants = pd.read_csv("data/appointments/participants.txt", sep=r"\s+")

# Merge datasets
data = appointments.merge(participants, on="participant")

# Filter patients with at least 5 appointments
data = data[data["count"] >= 5]

# Target and features
X = data.drop(columns=["status", "participant"])
y = (data["status"] == "no-show").astype(int)

# Preprocessing pipeline
numeric_features = ["age", "advance", "day", "month"]
categorical_features = ["sms_received", "sex", "hipertension", "diabetes", "alcoholism", "weekday"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Define model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced"))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Fit model
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot precision-recall curve
y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fullfilled", "No-show"], yticklabels=["Fullfilled", "No-show"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


rf = model.named_steps["classifier"]
X_test_transformed = model.named_steps["preprocessor"].transform(X_test)
shap_explainer = shap.TreeExplainer(rf)
shap_values = shap_explainer.shap_values(X_test_transformed)[1]

shap.summary_plot(shap_values, X_test_transformed, feature_names=X_test.columns)
plt.savefig("results/t3_shap_summary_plot.png")


# Learning curves
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=StratifiedKFold(n_splits=5), scoring="f1", train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training F1")
plt.plot(train_sizes, test_mean, label="Validation F1")
plt.xlabel("Training Set Size")
plt.ylabel("F1 Score")
plt.title("Learning Curves")
plt.legend()
plt.show()
