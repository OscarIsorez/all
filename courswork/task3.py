import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
appointment = pd.read_csv("data/appointments/appointments.txt", sep=r"\s+")
participants = pd.read_csv("data/appointments/participants.txt", sep=r"\s+")

# Merge data
data = pd.merge(appointment, participants, on="participant")
data = data[data["count"] > 5]

# Define categorical and numerical columns
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

# Define transformers
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="passthrough",
)

# Define Base Models
logistic_regression = make_pipeline(preprocessor, LogisticRegression(max_iter=10000, random_state=42))
nearest_neighbors = make_pipeline(preprocessor, KNeighborsClassifier())
decision_tree = make_pipeline(preprocessor, DecisionTreeClassifier())

# Define Stacking Model
stacking_model = StackingClassifier(
    estimators=[
        ('log_reg', logistic_regression),
        ('knn', nearest_neighbors),
        ('decision_tree', decision_tree)
    ],
    final_estimator=LogisticRegression(max_iter=10000, random_state=42),  # Meta-model
    cv=5  # Cross-validation for meta-model
)

# Prepare data for training
X = data.drop("status", axis=1)  # Replace 'target' with your target column
y = data["status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit Stacking Model
stacking_model.fit(X_train, y_train)

# Evaluate Models
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