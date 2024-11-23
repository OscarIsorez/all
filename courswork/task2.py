from matplotlib.ticker import FixedLocator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from scipy.stats import sem
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator

n = 4


# def create_visits(n):
#     for i in range(1, n + 1):
#         exec(f'global visit{i}; visit{i} = pd.read_csv("data/visits/visits/visit-{i}.csv")')
# create_visits(n)
# visits  = [f"visit{i}" for i in range(1, n+1) ]


# Load the data
visit1 = pd.read_csv("data/visits/visits/visit-1.csv")
visit2 = pd.read_csv("data/visits/visits/visit-2.csv")
visit3 = pd.read_csv("data/visits/visits/visit-3.csv")
visit4 = pd.read_csv("data/visits/visits/visit-4.csv")

""" 
print the head, info and describe of each visit
"""


def get_infos(n):
    if n == 1:
        print(visit1.head())
        print(visit1.info())
        print(visit1.describe())
    elif n == 2:
        print(visit2.head())
        print(visit2.info())
        print(visit2.describe())
    elif n == 3:
        print(visit3.head())
        print(visit3.info())
        print(visit3.describe())
    elif n == 4:
        print(visit4.head())
        print(visit4.info())
        print(visit4.describe())
    else:
        print("Invalid visit number")


""" 
Visually inspect the data for missing values
 """


def plot_missing_values_initial():
    visits = [visit1, visit2, visit3, visit4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, v in enumerate(visits):

        ax = axes[i // 2, i % 2]
        v.isnull().sum().plot(kind="bar", ax=ax)
        ax.set_title(f"Visit {i+1}")

    plt.tight_layout()
    plt.show()
    plt.close()


# Impute missing for SES in visit 1
visit1["SES"] = visit1["SES"].fillna(visit1["SES"].median())
visit1["sex"] = visit1["sex"].map({"F": 0, "M": 1})
visit1["hand"] = visit1["hand"].map({"R": 0, "L": 1})


visit1["ASF"] = visit1["ASF"].str.replace(",", ".")
visit1["ASF"] = visit1["ASF"].astype("float64")


visit1["CDR"] = visit1["CDR"].map(
    {
        "none": 0,
        "very mild": 1,
        "mild": 2,
        "very midl": 1,
        "vry mild": 1,
        "very miId": 1,
    }
)

""" - patient ID
- MRI exam ID
- visit
- age
- sex: M/F
- hand: R/L
- delay: MR delay time
- YOE: years of education
- SES: socioeconomic status (5 categories)
- MMSE: Mini Mental State Examination (0-30 points)
- CDR: Clinical Dementia Rating (none, very mild, mild, moderate, severe)
- eTIV: estimated Total Intracranial Volume
- nWBV: normalised Whole Brain Volume
- ASF: Atlas Scaling Factor """
categorical_columns = ["ID", "MRI_ID", "sex", "hand", "SES", "CDR"]
for col in categorical_columns:
    visit1[col] = visit1[col].astype("category")

# get_infos(1)


# Plot the distribution of the target variable


def plot_target_distribution():
    plt.figure(figsize=(6, 4))
    sns.countplot(x="CDR", data=visit1)
    plt.xlabel("CDR")
    plt.ylabel("Count")
    plt.show()
    plt.close()


visit1_cleaned = visit1

# Keep only ID and CDR for follow-up visits
visit2_cleaned = visit2[["ID", "CDR"]].rename(columns={"CDR": "cdr_visit2"})
visit3_cleaned = visit3[["ID", "CDR"]].rename(columns={"CDR": "cdr_visit3"})
visit4_cleaned = visit4[["ID", "CDR"]].rename(columns={"CDR": "cdr_visit4"})

# # Merge datasets on 'id'
merged_data = visit1_cleaned.merge(visit2_cleaned, on="ID", how="left")
merged_data = merged_data.merge(visit3_cleaned, on="ID", how="left")
merged_data = merged_data.merge(visit4_cleaned, on="ID", how="left")


# rename CDR in visit 1 to CDR_visit1

merged_data = merged_data.rename(columns={"CDR": "cdr_visit1"})


# transform the cdr columns to categorical
cdr_columns = ["cdr_visit1", "cdr_visit2", "cdr_visit3", "cdr_visit4"]
for col in cdr_columns[1:]:
    merged_data[col] = merged_data[col].map(
        {
            "none": 0,
            "very mild": 1,
            "mild": 2,
            "moderate": 3,
            "severe": 4,
            "midl": 1,
            "very miId": 1,
        }
    )
    merged_data[col] = merged_data[col].astype("category")


# plot the missing values for the cdr columns
def plot_missing_values_cdr():

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, col in enumerate(cdr_columns):
        ax = axes[i]
        merged_data[col].isnull().value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"{col}")
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_corr_matrix_merged_data():
    plt.figure(figsize=(10, 8))
    merged_data_without_MRI = merged_data.drop(columns=["MRI_ID"])
    sns.heatmap(merged_data_without_MRI.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    plt.close()


merged_data = merged_data.drop(columns=["MRI_ID"])


def plot_disease_progression():
    visits = ["cdr_visit1", "cdr_visit2", "cdr_visit3", "cdr_visit4"]

    mean_cdr = merged_data[visits].apply(pd.to_numeric).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(visits, mean_cdr, marker="o", linestyle="-", color="b")
    plt.xticks(
        ticks=range(len(visits)), labels=[f"Visit {i+1}" for i in range(len(visits))]
    )
    plt.xlabel("Visits")
    plt.ylabel("Mean CDR Score")
    plt.grid(True)
    plt.show()


# Define the columns with missing values
cdr_columns = ["cdr_visit2", "cdr_visit3", "cdr_visit4"]

# Prepare the data
y = merged_data["cdr_visit1"]
X = merged_data.drop(columns=["cdr_visit1", "ID"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline for handling missing values
missing_indicator = MissingIndicator(features="missing-only")
imputer = SimpleImputer(strategy="most_frequent")

preprocessor = ColumnTransformer(
    transformers=[
        ("missing_indicator", missing_indicator, cdr_columns),
        ("imputer", imputer, cdr_columns),
    ],
    remainder="passthrough",
)

log_reg = LogisticRegression(max_iter=10000, random_state=42)

# Create a pipeline with the preprocessor and the logistic regression model
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", log_reg)])

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

mean_score = scores.mean()
confidence_interval = sem(scores) * 1.96
print(f"Mean accuracy: {mean_score:.2f} +/- {confidence_interval:.2f}")
print(
    f"95% confidence interval: {mean_score - confidence_interval:.2f} - {mean_score + confidence_interval:.2f}"
)


def train_and_evaluate_logistic_regression(penalty):
    log_reg = LogisticRegression(max_iter=10000, random_state=42,solver="liblinear", penalty=penalty)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", log_reg)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    confidence_interval = sem(scores) * 1.96
    print(f"Mean accuracy ({penalty}): {mean_score:.2f} +/- {confidence_interval:.2f}")
    print(f"95% confidence interval: {mean_score - confidence_interval:.2f} - {mean_score + confidence_interval:.2f}")

train_and_evaluate_logistic_regression("l1")
train_and_evaluate_logistic_regression("l2")