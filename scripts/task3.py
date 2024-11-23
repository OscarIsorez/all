import pandas as pd
import numpy as np
from sklearn import model_selection, pipeline, preprocessing
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt

# Load the data
data = pd.read_pickle("processed/mushroom.pkl.gz")

# Encode categorical variables using get_dummies
categorical_columns = data.select_dtypes(
    include=["object", "category"]
).columns.tolist()
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Features and target
x = data_encoded.drop(columns=["kind_p"])
y = data_encoded["kind_p"]

# Initialize the SVC model with linear kernel
svc_linear = SVC(kernel="linear", class_weight="balanced", probability=True)
svc_linear.fit(x, y)
background = shap.sample(x, 100)

# SHAP values for the linear SVC model
explainer = shap.KernelExplainer(svc_linear.predict, background)
explanation = explainer.shap_values(background)
shap_values = explanation.values[0]
expected = explanation.expected_value
explanation = shap.Explanation(
    shap_values, expected, data=x.values, feature_names=x.columns
)
shap.plots.bar(explanation, show=False)
plt.savefig("results/figure.png", dpi=250, bbox_inches="tight")
plt.close("all")
shap.plots.beeswarm(explanation, show=False)
shap.plots.heatmap(explanation, max_display=10, show=False)


# Create a pipeline with StandardScaler and SVC
p_linear = pipeline.make_pipeline(preprocessing.StandardScaler(), svc_linear)

# Cross-validation
cv = model_selection.StratifiedKFold(n_splits=5)
scores_linear = model_selection.cross_val_score(p_linear, x, y, scoring="f1", cv=cv)
print("F1 Score (SVC) kernel=linear:", np.mean(scores_linear))


# Initialize the SVC model with RBF kernel
svc_rbf = SVC(
    kernel="rbf", gamma=1 / x.shape[1], class_weight="balanced", probability=True
)
svc_rbf.fit(x, y)


# SHAP values for the RBF SVC model
explainer = shap.KernelExplainer(svc_rbf.predict, background)
explanation = explainer.shap_values(background)
shap_values = explanation.values[0]
expected = explanation.expected_value
explanation = shap.Explanation(
    shap_values, expected, data=x.values, feature_names=x.columns
)
shap.plots.bar(explanation, show=False)
plt.savefig("results/figure.png", dpi=250, bbox_inches="tight")
plt.close("all")
shap.plots.beeswarm(explanation, show=False)
shap.plots.heatmap(explanation, max_display=10, show=False)

# Create a pipeline with StandardScaler and SVC
p_rbf = pipeline.make_pipeline(preprocessing.StandardScaler(), svc_rbf)

# Cross-validation
scores_rbf = model_selection.cross_val_score(p_rbf, x, y, scoring="f1", cv=cv)
print("F1 Score (SVC) kernel=rbf:", np.mean(scores_rbf))
