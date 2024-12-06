import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

appointment = pd.read_csv("data/appointments/appointments.txt", sep=r"\s+")
participants = pd.read_csv("data/appointments/participants.txt", sep=r"\s+")

data = pd.merge(appointment, participants, on="participant")
data = data[data["count"] > 5]

categorical_cols = [
    "participant",
    "sms_received",
    "day",
    "month",
    "weekday",
    "sex",
    "hipertension",
    "diabetes",
    "alcoholism",
]

numerical_cols = ["advance", "age", "count"]

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="passthrough",
)

logistic_regression = make_pipeline(preprocessor, LogisticRegression(max_iter=10000, random_state=42))
nearest_neighbors = make_pipeline(preprocessor, KNeighborsClassifier())
decision_tree = make_pipeline(preprocessor, DecisionTreeClassifier())

stacking_model = StackingClassifier(
    estimators=[
        ('log_reg', logistic_regression),
        ('knn', nearest_neighbors),
        ('decision_tree', decision_tree)
    ],
    final_estimator=LogisticRegression(max_iter=10000, random_state=42),  # Meta-model
    cv=5  
)

X = data.drop("status", axis=1)  # Replace 'status' with your target column
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

stacking_model.fit(X_train, y_train)

models = {'Logistic Regression': logistic_regression,
          'Nearest Neighbors': nearest_neighbors,
          'Decision Tree': decision_tree,
          'Stacking Model': stacking_model}

for name, model in models.items():
    if name != 'Stacking Model':
        model.fit(X_train, y_train)  # Fit base models independently
    y_pred = model.predict(X_test)
    print(f"Performance of {name}:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

y_binary = y.map({'fullfilled': 1, 'no-show': 0})

cv = StratifiedKFold(n_splits=5)
precision_list = []
recall_list = []
for train_idx, test_idx in cv.split(X, y_binary):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
    stacking_model.fit(X_train_fold, y_train_fold)
    y_pred_prob = stacking_model.predict_proba(X_test_fold)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_fold, y_pred_prob, pos_label=1)
    precision_list.append(precision)
    recall_list.append(recall)

plt.figure(figsize=(10, 6))
for i in range(len(precision_list)):
    plt.plot(recall_list[i], precision_list[i], lw=2, label=f'Fold {i+1}')
plt.xlabel('Recall')
plt.ylabel('Precision') 
plt.legend()
plt.title('Precision-Recall Curve for Stacking Model')
plt.savefig("results/t3_precision_recall_curve.png")
# plt.show()
