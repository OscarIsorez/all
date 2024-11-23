


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
data =  pd.read_pickle("processed/heart.pkl.gz")


X = data.drop(columns="time")
y = data["time"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define and train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

# 5. Plot comparison of model performances
metrics_df = pd.DataFrame(results).T  # Transpose for easier plotting
metrics_df[['MAE', 'MSE', 'R2']].plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Metric Score")
plt.xticks(rotation=45)
plt.show()

# Boxplot for prediction errors
errors = {name: abs(y_test - model.predict(X_test)) for name, model in models.items()}
errors_df = pd.DataFrame(errors)
sns.boxplot(data=errors_df)
plt.title("Error Distribution by Model")
plt.ylabel("Absolute Error (days)")
plt.show()

# Scatter plot of predictions vs actual values
plt.figure(figsize=(12, 6))
for name, model in models.items():
    plt.scatter(y_test, model.predict(X_test), label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.title("Predicted vs Actual Days Before Death")
plt.xlabel("Actual Days")
plt.ylabel("Predicted Days")
plt.legend()
plt.show()
